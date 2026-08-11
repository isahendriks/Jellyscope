"""Pipeline-stage-down Slack alerting: watches record.py/analyse.py/send.py's
own heartbeat files (already read into metadata.py's device_sample every tick
via heartbeat_status()) and fires a Slack alert when a stage goes stale, then
again when it recovers.

Deliberately does NOT watch metadata.py itself -- a process can't reliably
report its own death. That gap is meant to be covered by an external dead-
man's-switch (e.g. a cron job on the remote server checking upload recency),
not by anything running on the Jetson.

Called once per metadata.py device sample. Same debounce/cooldown/module-
global-state pattern as leak_alert.py (one metadata.py process calling this).
"""

import time

import config
from leak_alert import send_slack_alert

STAGES = ("record", "analyse", "send")

_consecutive_stale: dict[str, int] = {name: 0 for name in STAGES}
_currently_down: dict[str, bool] = {name: False for name in STAGES}
_last_down_alert_unix: dict[str, float] = {}


def _cooldown_ok(stage: str) -> bool:
    last = _last_down_alert_unix.get(stage)
    return last is None or (time.time() - last) >= config.PIPELINE_ALERT_COOLDOWN_S


def check_pipeline_health(device_sample: dict) -> None:
    """device_sample is collect_sample()'s return value -- device_sample[stage]
    is whatever heartbeat_status(stage) produced, already carrying "alive" /
    "last_update_age_s"."""
    for stage in STAGES:
        status = device_sample.get(stage) or {}
        alive = status.get("alive", False)

        if alive:
            _consecutive_stale[stage] = 0
            if _currently_down[stage]:
                # Recovery always announces immediately -- it's a one-shot event per
                # outage, not something that needs cooldown throttling like repeated
                # "still down" alerts do.
                send_slack_alert(f":white_check_mark: *{stage}.py back up* -- heartbeat resumed.")
                _currently_down[stage] = False
            continue

        _consecutive_stale[stage] += 1
        if _consecutive_stale[stage] < config.PIPELINE_ALERT_STALE_READS:
            continue  # rules out one missed/racy heartbeat-file read, not a real outage
        if _currently_down[stage]:
            continue  # already alerted for this outage -- don't spam every tick while it stays down
        if not _cooldown_ok(stage):
            continue  # flapped down again too soon after the last down-alert -- stay quiet

        age = status.get("last_update_age_s")
        age_txt = f"{age:.0f}s" if age is not None else "unknown"
        send_slack_alert(
            f":x: *{stage}.py appears down* -- heartbeat stale for {age_txt}.\n"
            "Jetson itself is still reachable (this alert came from the device) -- "
            f"check supervisor.sh's tmux session / logs/{stage}.log."
        )
        _currently_down[stage] = True
        _last_down_alert_unix[stage] = time.time()
