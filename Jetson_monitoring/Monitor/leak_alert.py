"""Leak detection and Slack alerting: the direct M0 leak sensor (CRITICAL,
unambiguous) plus two indirect BME280-derived signals -- dew point trend and
enclosure pressure trend (WARNING, noisier, so debounced/combined).

Why dew point, not raw %RH: relative humidity is temperature-dependent -- a
sunny afternoon can shift %RH several points with zero water ingress, purely
because warmer air holds more moisture at the same absolute water content.
Dew point (derived from temp+humidity via the Magnus-Tetens approximation)
tracks actual moisture content in the enclosure air, independent of that
temperature-driven noise, so it's a much cleaner "did water get in" signal.

THRESHOLDS ARE PLACEHOLDERS (see config.py) -- there's no baseline data from
this deployment yet to calibrate against. bme280_temp_c/bme280_humidity_pct/
bme280_pressure_mbar are already visible on the live-stream's Device chapter;
watch dew_point_c (computed here, not otherwise displayed yet) over a few
days of normal operation before trusting these numbers. A leak alert system
that cries wolf gets ignored, which defeats the point.

Called once per metadata.py sample tick; keeps its own rolling history and
debounce/cooldown state as module-level globals (same pattern as
metro_m0.py/strobe.py's persistent-connection state) since there's exactly
one metadata.py process calling this.
"""

import math
import time
from collections import deque

import requests

import config
import webhook_secrets

# (timestamp, dew_point_c, bme280_pressure_mbar) tuples, trimmed to config.LEAK_WARNING_WINDOW_S
_history: deque = deque()

_consecutive_leak_reads = 0
_last_alert_unix: dict[str, float] = {}  # tier -> last time THAT tier fired, for cooldown


def dew_point_c(temp_c: float | None, humidity_pct: float | None) -> float | None:
    """Magnus-Tetens approximation, valid for typical 0-60C / >0% RH ranges --
    plenty for an enclosure that should never see anything close to those
    extremes in normal operation."""
    if temp_c is None or humidity_pct is None or humidity_pct <= 0:
        return None
    b, c = 17.62, 243.12
    alpha = math.log(humidity_pct / 100.0) + (b * temp_c) / (c + temp_c)
    return (c * alpha) / (b - alpha)


def send_slack_alert(text: str) -> bool:
    """Best-effort -- never raises. If the webhook itself is unreachable (same
    flaky-5G/Tailscale link everything else here contends with), logs and
    moves on rather than blocking metadata.py's sample loop."""
    if not webhook_secrets.SLACK_WEBHOOK_URL:
        print("[leak_alert] SLACK_WEBHOOK_URL not configured, skipping alert -- see webhook_secrets_example.py")
        return False
    try:
        resp = requests.post(webhook_secrets.SLACK_WEBHOOK_URL, json={"text": text}, timeout=10)
        if resp.status_code != 200:
            print(f"[leak_alert] Slack webhook returned {resp.status_code}: {resp.text[:200]}")
            return False
        return True
    except requests.RequestException as exc:
        print(f"[leak_alert] Failed to send Slack alert: {exc}")
        return False


def _cooldown_ok(tier: str) -> bool:
    last = _last_alert_unix.get(tier)
    return last is None or (time.time() - last) >= config.LEAK_ALERT_COOLDOWN_S


def check_for_leak(device_sample: dict) -> None:
    """Call once per metadata.py device sample. Updates internal debounce/trend
    state and fires a Slack alert if a leak condition is newly detected (or
    the same tier re-qualifies past its cooldown)."""
    global _consecutive_leak_reads

    now = time.time()
    dp = dew_point_c(device_sample.get("bme280_temp_c"), device_sample.get("bme280_humidity_pct"))
    pressure = device_sample.get("bme280_pressure_mbar")
    _history.append((now, dp, pressure))
    while _history and now - _history[0][0] > config.LEAK_WARNING_WINDOW_S:
        _history.popleft()

    # --- Tier 1: direct sensor, debounced by consecutive reads only -- the
    # signal itself isn't in doubt, this just rules out a single electrical glitch.
    leak_detected = device_sample.get("leak_detected")
    if leak_detected:
        _consecutive_leak_reads += 1
    else:
        _consecutive_leak_reads = 0

    if _consecutive_leak_reads >= config.LEAK_SENSOR_CONSECUTIVE_READS and _cooldown_ok("critical"):
        send_slack_alert(
            ":rotating_light: *CRITICAL: Leak sensor triggered* :rotating_light:\n"
            f"Enclosure leak detector has read positive for "
            f"{_consecutive_leak_reads} consecutive samples. Retrieve the camera."
        )
        _last_alert_unix["critical"] = now
        return  # supersedes -- no need to also evaluate the noisier WARNING signals

    # --- Tier 2: indirect BME280 trend -- requires BOTH dew point rise AND a
    # pressure shift over a sustained window, not either alone, to cut down
    # single-metric false positives (see module docstring).
    if len(_history) < 2 or _history[-1][0] - _history[0][0] < config.LEAK_WARNING_WINDOW_S * 0.5:
        return  # not enough history yet to trust a trend
    oldest_dp = next((h[1] for h in _history if h[1] is not None), None)
    oldest_pressure = next((h[2] for h in _history if h[2] is not None), None)
    if dp is None or oldest_dp is None or pressure is None or oldest_pressure is None:
        return

    dp_rise = dp - oldest_dp
    pressure_shift = abs(pressure - oldest_pressure)
    if (dp_rise >= config.LEAK_DEW_POINT_RISE_THRESHOLD_C
            and pressure_shift >= config.LEAK_PRESSURE_DEVIATION_THRESHOLD_MBAR
            and _cooldown_ok("warning")):
        window_min = config.LEAK_WARNING_WINDOW_S / 60
        send_slack_alert(
            ":warning: *WARNING: possible leak (indirect signals)*\n"
            f"Enclosure dew point rose {dp_rise:.1f}C and pressure shifted "
            f"{pressure_shift:.1f} mbar over the last ~{window_min:.0f} min. "
            "Leak sensor itself hasn't triggered -- worth checking soon, not necessarily urgent."
        )
        _last_alert_unix["warning"] = now
