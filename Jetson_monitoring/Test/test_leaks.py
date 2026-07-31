#%% ================================================================
#  Enclosure Vacuum / Leak-Rate Monitor
# ====================================================================
"""
Reads the internal enclosure pressure (BME280/BMP280, field 'p', in Pa)
and the external/ambient pressure (Bar3XT via Keller LD, field 'P', in
mbar) from the Metro M0 over serial, and serves a live web page
(viewable over SSH via port-forwarding, same idea as the Jetson camera
live-stream script) showing:

  - internal pressure over the last 30 minutes (or since start),
  - a leak-rate estimate once you mark "pump-down complete",
  - a PASS / FAIL verdict against a configurable threshold.

Workflow matching what you described:
  1. Start this script.
  2. Pump the housing down with your hand vacuum pump.
  3. Once you stop pumping and the needle/reading has settled, click
     "Mark: Pump-down complete" on the web page.
  4. Watch the plot + leak-rate number. After a few minutes you'll
     have a statistically meaningful slope to judge PASS/FAIL.

-----------------------------------------------------------------------
THE PHYSICS (please read before trusting the numbers)
-----------------------------------------------------------------------
The enclosure is a fixed volume V of trapped air. Ideal gas law:

    P * V = n * R * T        =>        n = P * V / (R * T)

n = moles of air actually inside the enclosure. If the enclosure were
perfectly sealed, n is constant no matter what P or T do individually.
A REAL leak is air physically crossing the boundary, i.e. n changing
over time. That's the thing we actually want to measure - not P.

The problem: P by itself is NOT a reliable stand-in for n, because P
also changes with temperature even with zero air loss (V is ~fixed by
the rigid housing). Handling the housing, sun on the hull, or even the
hand pump doing work on the gas during pump-down can shift T by a
degree or more, and at a modest vacuum that alone can look like a
"leak" of similar magnitude to the real thing.

To remove that artifact, every sample is corrected to a fixed
reference temperature T_ref (the temperature recorded at the moment
you click "mark pump-down complete"):

    P_corrected(t) = P(t) * (T_ref / T(t))      <- T in KELVIN

P_corrected(t) is "what the pressure would read if temperature had
stayed at T_ref the whole time". Its slope over time is therefore
driven only by n(t) actually changing:

    dP_corrected/dt  ~  (R * T_ref / V) * dn/dt

We fit a straight line (least-squares) to P_corrected vs. time after
the mark. The resulting slope is reported in:

  - Pa/min and inHg/min, so you can compare directly against the
    Blue Robotics-style acceptance criterion we discussed
    (~1 inHg drop per 15 min, i.e. ~0.067 inHg/min, is their
    documented "acceptable" ballpark for an O-ring-sealed enclosure);
  - an equivalent molar leak rate dn/dt = (V / (R * T_ref)) * slope,
    which is then re-expressed as a volumetric flow at 1 atm / 20 C
    (cm^3/min) purely so the number is intuitive - it's the flow you
    would measure with a bubble flow meter at the surface, not the
    actual in-situ flow (which is smaller, since it's happening at a
    partial vacuum).

IMPORTANT ASSUMPTION - check this for your build:
    The temperature-correction above needs the temperature of the AIR
    INSIDE the sealed volume. Two temperature channels are available
    over serial: 't' (DS18B20) and 'T' (Bar3XT). The Bar3XT is
    conventionally the *external*/water-side sensor, so this script
    defaults to using the DS18B20 ('t') as the internal-air proxy.
    If your DS18B20 is actually mounted outside, or the Bar3XT die is
    what's actually sitting inside your sealed volume, flip
    INTERNAL_TEMP_FIELD below to "T" instead of "t".
-----------------------------------------------------------------------
"""

import sys
import io
import time
import threading
from collections import deque
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
HISTORY_WINDOW_S = 120 * 60         # how much history to keep & plot [s]
SETTLE_TIME_S = 15                 # ignore this many seconds right after
                                    # "mark pump-down complete" (pump vibration
                                    # / thermal transient from the pumping work)
LEAK_THRESHOLD_INHG_PER_MIN = 1.0 / 15.0   # ~0.067 inHg/min, Blue Robotics-style
FIT_DEFICIT_CUTOFF_FRAC = 0.3      # stop feeding samples into the leak-rate fit once
                                    # the internal/external pressure differential has
                                    # decayed below this fraction of its post-settle
                                    # value - closer to ambient than that, the signal
                                    # is mostly sensor noise + inter-sensor calibration
                                    # mismatch, not the leak (see compute_leak_rate())
INTERNAL_TEMP_FIELD = "t"          # "t" = DS18B20, "T" = Bar3XT - see assumption above
INTERNAL_TEMP_SANE_RANGE_C = (-10.0, 60.0)   # samples outside this are discarded before
                                    # the leak-rate fit. Covers DS18B20's known error
                                    # sentinels (-127 hard-fault, 85 power-on-reset default)
                                    # and any other garbled serial line - any of these would
                                    # otherwise blow up the T_ref/T(t) correction and wreck
                                    # both the fitted slope and the plot's y-scale.
STREAM_PORT = 8081                 # browse to http://<this-host>:8081/ (or SSH -L 8081:localhost:8081)

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


#%% ---------------- Shared data store (written by serial thread) ----
data_lock = threading.Lock()
history = deque()          # each entry: (t_rel_s, p_int_pa, p_ext_pa, temp_c, leak_flag)
latest_sample = {}
t0_wall = None              # time.time() at first received sample
test_start_t = None         # t_rel [s] when "mark pump-down complete" was pressed
stop_event = threading.Event()


def parse_line(line):
    """Parse 'p:...,h:...,P:...,T:...,d:...,l:...,t:...' -> dict of floats."""
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

        p_int_pa = fields["p"]                       # internal enclosure air, BME280/BMP280
        p_ext_pa = fields["P"] * 100.0                # Bar3XT reports mbar -> convert to Pa
        temp_c = fields.get(INTERNAL_TEMP_FIELD, np.nan)
        leak_flag = fields.get("l", 0.0)

        with data_lock:
            history.append((t_rel, p_int_pa, p_ext_pa, temp_c, leak_flag))
            while history and (t_rel - history[0][0]) > HISTORY_WINDOW_S:
                history.popleft()
            latest_sample.update(
                t=t_rel, p_int_pa=p_int_pa, p_ext_pa=p_ext_pa,
                temp_c=temp_c, leak_flag=leak_flag,
            )


#%% ---------------- Leak-rate computation (the physics, applied) ----
def compute_leak_rate():
    """
    Returns a dict describing the leak rate since test_start_t, using the
    temperature-corrected pressure trace described in the module docstring.

    The trace is also fit against the ambient/external pressure (Bar3XT,
    already sampled every line) rather than forced through a straight line:
    as the enclosure re-equilibrates with ambient, the pressure differential
    driving the leak shrinks, so the *rate* shrinks too - the trace bends
    over ("saturates") instead of staying linear, and a straight-line fit
    across that whole bend understates the true leak rate. This models it as
    first-order relaxation towards ambient, D(t) = D0 * exp(-t/tau) where
    D = p_ext - p_corrected, and reports the *initial* (t=0, largest
    differential, worst-case) rate dP/dt|_0 = D0/tau - the conservative
    number the PASS/FAIL threshold is meant to catch.

    Once D has decayed close to zero, it's no longer a reliable signal - it's
    dominated by measurement noise and by the internal/external sensors' own
    calibration mismatch (two different physical pressure sensors, never in
    perfect agreement even at true equilibrium), not by the leak. So samples
    are dropped from the fit once D falls below FIT_DEFICIT_CUTOFF_FRAC of its
    post-settle value, instead of letting that noisy near-ambient tail pull
    the fit off of the steeper, more informative early portion. Falls back to
    a plain linear fit over the whole window if there's no usable initial
    differential to anchor the cutoff to.
    """
    with data_lock:
        snapshot = list(history)

    if test_start_t is None:
        return {"verdict": "NOT_STARTED", "n_points": len(snapshot)}

    candidates = [(t, p, temp, p_ext) for (t, p, p_ext, temp, _) in snapshot
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

    d0_ref = deficit[0]   # differential right after settle - the reference "full" signal
    if d0_ref > 0:
        fit_mask = deficit >= FIT_DEFICIT_CUTOFF_FRAC * d0_ref
    else:
        fit_mask = np.ones_like(deficit, dtype=bool)   # no usable differential to anchor a cutoff to

    b = None
    if d0_ref > 0 and fit_mask.sum() >= 5:
        b, ln_d0 = np.polyfit(t_shift[fit_mask], np.log(deficit[fit_mask]), 1)

    if b is not None and b < 0:
        fit_kind = "exponential"
        d0 = np.exp(ln_d0)
        slope_pa_per_s = d0 * (-b)                        # dP/dt at t=0 (worst case)
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

    # Re-express as an equivalent surface flow (1 atm, 20 C) - intuitive only,
    # not the true in-situ flow rate (which is smaller, at partial vacuum).
    P_std, T_std = 101325.0, c_to_k(20.0)
    vol_flow_cm3_per_min = dn_dt_mol_per_min * R * T_std / P_std * 1.0e6

    passed = abs(slope_inhg_per_min) <= LEAK_THRESHOLD_INHG_PER_MIN

    return {
        "verdict": "PASS" if passed else "FAIL",
        "n_points": len(pts),
        "n_rejected_temp": n_rejected_temp,
        "n_in_fit_window": int(fit_mask.sum()),
        "fit_kind": fit_kind,
        "test_duration_min": (t_arr[-1] - test_start_t) / 60.0,
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
  <h2>Enclosure Leak Test Monitor</h2>
  <img id="plot" src="/plot.png" style="max-width:95vw;border:1px solid #444">
  <div>
    <button onclick="markStart()" style="padding:10px 18px;font-size:15px;
            margin:12px 0;background:#2980b9;color:#fff;border:none;
            border-radius:6px;cursor:pointer">
      Mark: Pump-down complete / start leak test
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
    setInterval(refresh, 3000);
  </script>
</body></html>
"""


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
    return jsonify(out)


@app.route("/plot.png")
def plot_png():
    with data_lock:
        snapshot = list(history)

    fig, ax = plt.subplots(figsize=(9, 4.5))

    if snapshot:
        t_arr = np.array([s[0] for s in snapshot])
        p_hpa = np.array([s[1] for s in snapshot]) / 100.0  # Pa -> hPa
        t_min = (t_arr - t_arr[-1]) / 60.0  # minutes relative to "now"

        # thin out for plotting performance if history is dense
        max_pts = 3000
        step = max(1, len(t_min) // max_pts)
        ax.plot(t_min[::step], p_hpa[::step], color="#3498db",
                 label="Internal pressure (raw)")

        result = compute_leak_rate()
        if "_t_arr" in result:
            t_fit_min = (result["_t_arr"] - t_arr[-1]) / 60.0
            ax.plot(t_fit_min, result["_p_corrected"] / 100.0, color="#e67e22",
                     linestyle="--", linewidth=1, label="Temp.-corrected")
            # fit line is drawn only over its own (possibly shorter) window - it
            # stops once the differential to ambient gets too small to trust, see
            # FIT_DEFICIT_CUTOFF_FRAC / compute_leak_rate() docstring
            t_fitline_min = (result["_fit_t_arr"] - t_arr[-1]) / 60.0
            ax.plot(t_fitline_min, result["_fit_curve"] / 100.0, color="#2ecc71", linewidth=2,
                     label=f"Fit ({result['fit_kind']}): {result['slope_inhg_per_min']:.4f} "
                           f"inHg/min ({result['verdict']})")

        if test_start_t is not None:
            ax.axvline((test_start_t - t_arr[-1]) / 60.0, color="#c0392b",
                       linestyle=":", linewidth=1.5, label="Test start")

    ax.set_xlabel("Time [min] (0 = now)")
    ax.set_ylabel("Internal pressure [hPa]")
    ax.set_title(f"Internal enclosure pressure - last {HISTORY_WINDOW_S // 60} min")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
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

    print(f"Serving leak monitor at http://<this-host-ip>:{STREAM_PORT}/")
    print("(over SSH: ssh -L 8081:localhost:8081 user@host, then open "
          "http://localhost:8081/ locally)")
    try:
        app.run(host="0.0.0.0", port=STREAM_PORT, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        ser.close()

# %%