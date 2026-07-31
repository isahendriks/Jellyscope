#%% ================================================================
#  Overnight Enclosure Leak / Humidity / Temperature Logger
# ====================================================================
"""
Same underlying idea as the short pump-down test script, stretched to
run unattended for many hours:

  1. Close the housing, pump it down with the hand vacuum pump.
  2. Start this script, then click "Mark: pump-down complete" once the
     pumping-induced pressure/temperature transient has settled.
  3. Leave it running overnight (see the tmux/systemd note at the
     bottom of this docstring - a bare SSH session will die when your
     connection drops, taking this script with it unless you protect
     it).
  4. In the morning, open the web page (or just look at the CSV) and
     you'll see the full-night pressure/humidity/temperature trace,
     plus a fitted leak-rate slope and PASS/FAIL verdict.

Every sample is appended to a CSV file on disk as it arrives (not just
held in memory), so you don't lose the whole night's data if the
script, the SSH session, or the Flask server hiccups at 3am.

-----------------------------------------------------------------------
THE PHYSICS - PRESSURE (same model as the short-test script)
-----------------------------------------------------------------------
Ideal gas law for the fixed enclosure volume V:  P*V = n*R*T
=> n = P*V/(R*T). n (moles of air trapped inside) only changes if air
is actually crossing the boundary - that's the leak we want. P alone
isn't a reliable stand-in for n because P also drifts with temperature
even with zero air loss, and overnight ambient temperature WILL drift
(house cooling down, etc). So every internal-pressure sample is first
corrected to the reference temperature at test start:

    P_corrected(t) = P(t) * (T_ref / T(t))         <- T in KELVIN

This is "what the pressure would read if temperature had stayed at
T_ref all night" - its trend is driven only by real air loss/gain.

SATURATION-AWARE FIT: unlike a short pump-down test, an overnight test
easily runs long enough for the internal pressure to relax most of the
way back to ambient. As it does, the physical driving force behind the
leak (the pressure differential across the wall) shrinks, so the leak
*rate* naturally shrinks too - the trace bends over instead of staying
a straight line. Modelling that: let D(t) = P_ext(t) - P_corrected(t)
(the remaining deficit vs. ambient). If the leak is a simple
orifice/gap leak, D relaxes roughly exponentially: D(t) = D0*exp(-t/tau).
We fit that exponential to the early, high-signal portion, and report
the *initial* slope dP/dt|_0 = D0/tau - the conservative, worst-case
rate right after pump-down, which is what the PASS/FAIL threshold is
meant to compare against.

Once D has decayed below FIT_DEFICIT_CUTOFF_FRAC of its post-settle
value, it's dropped from the fit: near ambient, the "signal" is
dominated by sensor noise and by the small permanent calibration
mismatch between the internal (BME280) and external (Bar3XT) pressure
dies - not by the leak - and letting that noisy tail into the fit
would just corrupt the slope estimate.

-----------------------------------------------------------------------
THE PHYSICS - HUMIDITY (new for this script)
-----------------------------------------------------------------------
Relative humidity (%RH) is relative to temperature - it rises just from
the housing cooling down overnight, with zero water actually getting
in. To get a trend that only moves when real moisture is entering, we
convert %RH + BME280's own temperature to ABSOLUTE humidity (grams of
water vapor per m^3 of air) using the Magnus-Tetens approximation:

    e_s(T) = 6.112 * exp( 17.62*T / (243.12+T) )   [hPa]  saturation vapor pressure
    e(T)   = RH/100 * e_s(T)                        [hPa]  actual vapor pressure
    a      = 216.7 * e(T) / (T + 273.15)            [g/m^3] absolute humidity

`a` only changes when the actual mass of water vapor inside changes -
a flat `a` overnight is a clean pass; a climbing `a` means water is
getting in, however it's getting in (O-ring, cable wicking, whatever -
which is exactly why this is a fairer, mechanism-agnostic check than
the cable-wicking-blind pressure slope).

We also compute dew point (again via Magnus-Tetens, inverted), because
that's the number that matters for lens/optics fogging risk: if the
coldest internal surface (usually the lens/window, in contact with
cold water) drops below the dew point, you'll get condensation on it
regardless of what the bulk RH% says.

IMPORTANT: this requires the humidity reading and the temperature
reading to be compensated with respect to EACH OTHER - i.e. the BME280
firmware must call readTemperature() before readHumidity() each cycle
so the humidity compensation uses this cycle's t_fine, not the
previous cycle's. (Flagged separately - fix it in firmware if not
already done.)

-----------------------------------------------------------------------
RUNNING THIS OVERNIGHT WITHOUT LOSING IT WHEN SSH DROPS
-----------------------------------------------------------------------
A plain `python3 overnight_leak_monitor.py` over SSH dies the moment
your SSH session disconnects. Use one of:

    tmux new -s leaktest
    python3 overnight_leak_monitor.py
    # Ctrl-b, d to detach - reattach in the morning with: tmux attach -t leaktest

or:

    nohup python3 overnight_leak_monitor.py > leak_test.log 2>&1 &
d
Either way, the CSV file keeps recording regardless of whether you're
still looking at the web page.
-----------------------------------------------------------------------
"""

import sys
import io
import csv
import time
import threading
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import serial
import matplotlib
matplotlib.use("Agg")  # headless - no display attached over SSH
import matplotlib.pyplot as plt
from flask import Flask, Response, jsonify

MONITOR_DIR = Path(__file__).resolve().parent.parent / "Monitor"
if str(MONITOR_DIR) not in sys.path:
    sys.path.insert(0, str(MONITOR_DIR))
import config

#%% ---------------- USER-CONFIGURABLE PARAMETERS -------------------
ENCLOSURE_VOLUME_L = 20.4          # internal air volume of YOUR housing [L] - CHECK THIS
MAX_HISTORY_HOURS = 24              # safety cap on in-memory history (CSV keeps everything regardless)
SAMPLE_MIN_INTERVAL_S = 2.0         # store/log at most one sample every N seconds (keeps an
                                    # overnight run's memory/CSV size sane; raw serial rate is
                                    # much faster than we need for an hours-long trend)
SETTLE_TIME_S = 5 * 60              # ignore this long right after "mark pump-down complete" -
                                    # longer than the short-test script, since post-pumping
                                    # thermal transients (housing/gas warmed by your hand and
                                    # by pump work) take a few minutes to fully die out
FIT_DEFICIT_CUTOFF_FRAC = 0.3       # stop feeding samples into the leak-rate fit once the
                                    # internal/external pressure differential has decayed
                                    # below this fraction of its post-settle value - see
                                    # "SATURATION-AWARE FIT" in the module docstring
LEAK_THRESHOLD_INHG_PER_MIN = 1.0 / 15.0    # ~0.067 inHg/min, Blue Robotics-style starting point -
                                    # rescale for your enclosure volume, see our discussion
INTERNAL_TEMP_FIELD = "b"           # "b" = BME280's own temperature - co-located with the BME280
                                    # pressure AND humidity sensor, so it's the right temperature
                                    # to correct both with. ("t"=DS18B20, "T"=Bar3XT/external -
                                    # only use those if BME280 temp is unavailable.)
INTERNAL_TEMP_SANE_RANGE_C = (-10.0, 60.0)   # samples outside this are discarded before use
                                    # (covers DS18B20 error sentinels / garbled serial lines;
                                    # BME280 shouldn't hit these, but cheap insurance)
STREAM_PORT = 8081                  # browse http://<this-host>:8081/ (or SSH -L 8081:localhost:8081)
LOG_DIR = Path(__file__).resolve().parent / "leak_test_logs"

serial_port = config.METRO_M0_SERIAL_PORT
baud_rate = config.BAUD_RATE
serial_timeout_s = config.SERIAL_TIMEOUT_S

#%% ---------------- Physical constants / unit helpers ---------------
R = 8.314462618          # J/(mol K), universal gas constant
PA_PER_INHG = 3386.389   # Pa per inch of mercury
T_KELVIN_OFFSET = 273.15


def c_to_k(t_c):
    return t_c + T_KELVIN_OFFSET


def pa_to_inhg(pa):
    return pa / PA_PER_INHG


def saturation_vapor_pressure_hpa(t_c):
    """Magnus-Tetens approximation, valid roughly -45..60 C."""
    return 6.112 * np.exp((17.62 * t_c) / (243.12 + t_c))


def absolute_humidity_g_m3(rh_pct, t_c):
    e_hpa = (rh_pct / 100.0) * saturation_vapor_pressure_hpa(t_c)
    return 216.7 * e_hpa / c_to_k(t_c)


def dew_point_c(rh_pct, t_c):
    e_hpa = np.maximum((rh_pct / 100.0) * saturation_vapor_pressure_hpa(t_c), 1e-6)
    ln_term = np.log(e_hpa / 6.112)
    return (243.12 * ln_term) / (17.62 - ln_term)


#%% ---------------- Shared data store (written by serial thread) ----
data_lock = threading.Lock()
# each entry: (t_rel_s, p_int_pa, p_ext_pa, temp_int_c, humidity_pct, leak_flag)
history = deque()
latest_sample = {}
t0_wall = None              # time.time() at first received sample
test_start_t = None         # t_rel [s] when "mark pump-down complete" was pressed
stop_event = threading.Event()

LOG_DIR.mkdir(exist_ok=True)
csv_path = LOG_DIR / f"leak_test_{datetime.now():%Y%m%d_%H%M%S}.csv"
csv_lock = threading.Lock()
csv_file = open(csv_path, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["iso_time", "t_rel_s", "p_int_pa", "p_ext_pa",
                      "temp_int_c", "humidity_pct", "leak_flag"])
csv_file.flush()


def parse_line(line):
    """Parse 'p:...,h:...,b:...,P:...,d:...,T:...,l:...,t:...' -> dict of floats."""
    fields = {}
    for part in line.split(","):
        if ":" not in part:
            continue
        key, _, val = part.partition(":")
        try:
            fields[key.strip()] = float(val)
        except ValueError:
            pass
    return fields


def serial_reader_thread(ser):
    global t0_wall
    last_stored_t = None

    while not stop_event.is_set():
        try:
            raw = ser.readline().decode("utf-8", errors="ignore").rstrip()
        except serial.SerialException as exc:
            print(f"Serial read error: {exc}")
            time.sleep(1)
            continue

        if not raw:
            continue

        fields = parse_line(raw)
        if "p" not in fields or "P" not in fields:
            continue  # incomplete/garbled line - skip it

        now = time.time()
        if t0_wall is None:
            t0_wall = now
        t_rel = now - t0_wall

        if last_stored_t is not None and (t_rel - last_stored_t) < SAMPLE_MIN_INTERVAL_S:
            continue  # throttle - overnight run doesn't need every raw line
        last_stored_t = t_rel

        p_int_pa = fields["p"]                       # internal enclosure air, BME280
        p_ext_pa = fields["P"] * 100.0                # Bar3XT reports mbar -> convert to Pa
        temp_c = fields.get(INTERNAL_TEMP_FIELD, np.nan)
        humidity_pct = fields.get("h", np.nan)
        leak_flag = fields.get("l", 0.0)

        with data_lock:
            history.append((t_rel, p_int_pa, p_ext_pa, temp_c, humidity_pct, leak_flag))
            max_age = MAX_HISTORY_HOURS * 3600.0
            while history and (t_rel - history[0][0]) > max_age:
                history.popleft()
            latest_sample.update(
                t=t_rel, p_int_pa=p_int_pa, p_ext_pa=p_ext_pa,
                temp_c=temp_c, humidity_pct=humidity_pct, leak_flag=leak_flag,
            )

        with csv_lock:
            csv_writer.writerow([datetime.now().isoformat(timespec="seconds"),
                                  f"{t_rel:.2f}", p_int_pa, p_ext_pa,
                                  temp_c, humidity_pct, leak_flag])
            csv_file.flush()   # cheap at this sample rate, and means the file
                                # is always readable/recoverable mid-run


#%% ---------------- Leak-rate computation (pressure physics, applied) ----
def compute_leak_rate():
    """Same saturation-aware exponential-relaxation fit as the short-test
    script - see the module docstring for the full derivation."""
    with data_lock:
        snapshot = list(history)

    if test_start_t is None:
        return {"verdict": "NOT_STARTED", "n_points": len(snapshot)}

    candidates = [(t, p, temp, p_ext) for (t, p, p_ext, temp, _, _) in snapshot
                  if t >= test_start_t + SETTLE_TIME_S and not np.isnan(temp)]

    temp_lo, temp_hi = INTERNAL_TEMP_SANE_RANGE_C
    pts = [pt for pt in candidates if temp_lo <= pt[2] <= temp_hi]
    n_rejected_temp = len(candidates) - len(pts)

    if len(pts) < 5:
        return {"verdict": "COLLECTING", "n_points": len(pts),
                "n_rejected_temp": n_rejected_temp}

    t_arr = np.array([p[0] for p in pts])
    p_arr = np.array([p[1] for p in pts])
    temp_arr = np.array([p[2] for p in pts])
    p_ext_arr = np.array([p[3] for p in pts])

    T_ref_c = temp_arr[0]
    T_ref_k = c_to_k(T_ref_c)
    p_corrected = p_arr * (T_ref_k / c_to_k(temp_arr))   # ideal-gas temperature correction

    t_shift = t_arr - t_arr[0]
    deficit = p_ext_arr - p_corrected   # positive: internal still below ambient

    d0_ref = deficit[0]
    if d0_ref > 0:
        fit_mask = deficit >= FIT_DEFICIT_CUTOFF_FRAC * d0_ref
    else:
        fit_mask = np.ones_like(deficit, dtype=bool)

    b = None
    if d0_ref > 0 and fit_mask.sum() >= 5:
        b, ln_d0 = np.polyfit(t_shift[fit_mask], np.log(deficit[fit_mask]), 1)

    if b is not None and b < 0:
        fit_kind = "exponential"
        d0 = np.exp(ln_d0)
        slope_pa_per_s = d0 * (-b)
        fit_t_arr = t_arr[fit_mask]
        fit_curve = p_ext_arr[fit_mask] - d0 * np.exp(b * t_shift[fit_mask])
    else:
        fit_kind = "linear"
        slope_pa_per_s, intercept = np.polyfit(t_shift[fit_mask], p_corrected[fit_mask], 1)
        fit_t_arr = t_arr[fit_mask]
        fit_curve = slope_pa_per_s * t_shift[fit_mask] + intercept

    slope_pa_per_min = slope_pa_per_s * 60.0
    slope_inhg_per_min = pa_to_inhg(slope_pa_per_min)

    V_m3 = ENCLOSURE_VOLUME_L / 1000.0
    dn_dt_mol_per_min = (V_m3 / (R * T_ref_k)) * slope_pa_per_min

    P_std, T_std = 101325.0, c_to_k(20.0)
    vol_flow_cm3_per_min = dn_dt_mol_per_min * R * T_std / P_std * 1.0e6

    passed = abs(slope_inhg_per_min) <= LEAK_THRESHOLD_INHG_PER_MIN

    return {
        "verdict": "PASS" if passed else "FAIL",
        "n_points": len(pts),
        "n_rejected_temp": n_rejected_temp,
        "n_in_fit_window": int(fit_mask.sum()),
        "fit_kind": fit_kind,
        "test_duration_hr": (t_arr[-1] - test_start_t) / 3600.0,
        "T_ref_c": T_ref_c,
        "slope_pa_per_min": slope_pa_per_min,
        "slope_inhg_per_min": slope_inhg_per_min,
        "molar_leak_mol_per_min": dn_dt_mol_per_min,
        "equiv_leak_cm3_per_min": vol_flow_cm3_per_min,
        "threshold_inhg_per_min": LEAK_THRESHOLD_INHG_PER_MIN,
        "_t_arr": t_arr, "_p_corrected": p_corrected,
        "_fit_t_arr": fit_t_arr, "_fit_curve": fit_curve,
    }


#%% ---------------- Flask web app (viewable over SSH) ----------------
app = Flask(__name__)

INDEX_HTML = """
<html><body style="margin:0;background:#111;color:#eee;
                    font-family:sans-serif;padding:20px">
  <h2>Overnight Enclosure Test Monitor</h2>
  <p style="font-size:13px;color:#aaa">Logging to: {csv_path}</p>
  <img id="plot" src="/plot.png" style="max-width:95vw;border:1px solid #444">
  <div>
    <button onclick="markStart()" style="padding:10px 18px;font-size:15px;
            margin:12px 0;background:#2980b9;color:#fff;border:none;
            border-radius:6px;cursor:pointer">
      Mark: Pump-down complete / start test
    </button>
    <button onclick="resetTest()" style="padding:10px 18px;font-size:15px;
            margin:12px 0 12px 8px;background:#7f8c8d;color:#fff;border:none;
            border-radius:6px;cursor:pointer">
      Restart test
    </button>
  </div>
  <pre id="status" style="font-size:14px;line-height:1.5;
         background:#000;padding:12px;border-radius:6px;
         max-width:95vw;white-space:pre-wrap"></pre>
  <script>
    function markStart() { fetch('/mark_start', {method: 'POST'}); }
    function resetTest() { fetch('/reset', {method: 'POST'}); }
    async function refresh() {
      try {
        const d = await (await fetch('/status')).json();
        document.getElementById('status').textContent = JSON.stringify(d, null, 2);
        document.getElementById('plot').src = '/plot.png?' + Date.now();
      } catch (e) {}
    }
    refresh();
    setInterval(refresh, 10000);
  </script>
</body></html>
""".replace("{csv_path}", str(csv_path))


@app.route("/")
def index():
    return INDEX_HTML


@app.route("/mark_start", methods=["POST"])
def mark_start():
    global test_start_t
    with data_lock:
        test_start_t = latest_sample.get("t", 0.0)
    return "", 204


@app.route("/reset", methods=["POST"])
def reset():
    global test_start_t
    with data_lock:
        test_start_t = None
    return "", 204


@app.route("/status")
def status():
    result = compute_leak_rate()
    out = {k: v for k, v in result.items() if not k.startswith("_")}
    with data_lock:
        out["latest"] = dict(latest_sample)
        out["history_len"] = len(history)
        out["test_marked"] = test_start_t is not None
    out["csv_path"] = str(csv_path)
    return jsonify(out)


@app.route("/plot.png")
def plot_png():
    with data_lock:
        snapshot = list(history)

    fig, (ax_p, ax_h, ax_t) = plt.subplots(
        3, 1, figsize=(10, 10), sharex=True,
        gridspec_kw={"height_ratios": [2, 1.3, 1.3]},
    )

    if snapshot:
        t_arr = np.array([s[0] for s in snapshot])
        p_int_hpa = np.array([s[1] for s in snapshot]) / 100.0
        p_ext_hpa = np.array([s[2] for s in snapshot]) / 100.0
        temp_arr = np.array([s[3] for s in snapshot])
        hum_arr = np.array([s[4] for s in snapshot])
        t_hr = (t_arr - t_arr[0]) / 3600.0   # hours since first sample

        # -------- Pressure panel --------
        ax_p.plot(t_hr, p_int_hpa, color="#3498db", label="Internal pressure (raw)")
        ax_p.plot(t_hr, p_ext_hpa, color="#95a5a6", linewidth=1,
                  label="External pressure (Bar3XT, ambient ref.)")

        result = compute_leak_rate()
        if "_t_arr" in result:
            t_fit_hr = (result["_t_arr"] - t_arr[0]) / 3600.0
            ax_p.plot(t_fit_hr, result["_p_corrected"] / 100.0, color="#e67e22",
                      linestyle="--", linewidth=1, label="Temp.-corrected")
            t_fitline_hr = (result["_fit_t_arr"] - t_arr[0]) / 3600.0
            ax_p.plot(t_fitline_hr, result["_fit_curve"] / 100.0, color="#2ecc71",
                      linewidth=2,
                      label=f"Fit ({result['fit_kind']}): {result['slope_inhg_per_min']:.4f} "
                            f"inHg/min ({result['verdict']})")

        if test_start_t is not None:
            ax_p.axvline((test_start_t - t_arr[0]) / 3600.0, color="#c0392b",
                         linestyle=":", linewidth=1.5, label="Test start")

        ax_p.set_ylabel("Pressure [hPa]")
        ax_p.set_title("Overnight enclosure test - pressure / humidity / temperature")
        ax_p.legend(loc="best", fontsize=8)
        ax_p.grid(alpha=1)

        # -------- Humidity panel (RH%, absolute humidity, dew point) --------
        valid = ~np.isnan(hum_arr) & ~np.isnan(temp_arr)
        ax_h.plot(t_hr[valid], hum_arr[valid], color="#9b59b6", label="Relative humidity [%]")
        ax_h.set_ylabel("RH [%]", color="#9b59b6")
        ax_h.tick_params(axis="y", labelcolor="#9b59b6")
        ax_h.grid(alpha=1)

        if valid.any():
            abs_hum = absolute_humidity_g_m3(hum_arr[valid], temp_arr[valid])
            ax_h2 = ax_h.twinx()
            ax_h2.plot(t_hr[valid], abs_hum, color="#1abc9c", linewidth=1.5,
                       label="Absolute humidity [g/m^3]")
            ax_h2.set_ylabel("Absolute humidity [g/m^3]", color="#1abc9c")
            ax_h2.tick_params(axis="y", labelcolor="#1abc9c")

        # -------- Temperature panel --------
        ax_t.plot(t_hr[valid], temp_arr[valid], color="#e74c3c", label="Internal temp (BME280) [C]")
        if valid.any():
            dp = dew_point_c(hum_arr[valid], temp_arr[valid])
            ax_t.plot(t_hr[valid], dp, color="#3498db", linestyle="--",
                      label="Dew point [C]")
        ax_t.set_ylabel("Temperature [C]")
        ax_t.set_xlabel("Time since start [hours]")
        ax_t.legend(loc="best", fontsize=8)
        ax_t.grid(alpha=1)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return Response(buf.getvalue(), mimetype="image/png")


#%% ---------------- Main --------------------------------------------
if __name__ == "__main__":
    ser = serial.Serial(serial_port, baud_rate, timeout=serial_timeout_s)
    ser.write(b"C")

    reader = threading.Thread(target=serial_reader_thread, args=(ser,), daemon=True)
    reader.start()

    print(f"Logging to: {csv_path}")
    print(f"Serving overnight monitor at http://<this-host-ip>:{STREAM_PORT}/")
    print("(over SSH: ssh -L 8081:localhost:8081 user@host, then open "
          "http://localhost:8081/ locally)")
    print("Run this under tmux/screen/nohup so it survives an SSH disconnect - see "
          "the module docstring.")
    try:
        app.run(host="0.0.0.0", port=STREAM_PORT, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        ser.close()
        with csv_lock:
            csv_file.close()

# %%