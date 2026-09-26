#!/usr/bin/env bash
set -uo pipefail

# Keep Tailscale recoverable when the control connection goes stale. The node's
# state is persisted by tailscaled, so restarting the daemon does not require a
# new auth key or an SSH session.
CHECK_INTERVAL_S="${CHECK_INTERVAL_S:-30}"
RESTART_COOLDOWN_S="${RESTART_COOLDOWN_S:-120}"
TAILSCALE="/usr/bin/tailscale"

status_is_healthy() {
    local status_json backend_state in_network_map health
    status_json="$(${TAILSCALE} status --json 2>/dev/null)" || return 1

    backend_state="$(printf '%s' "${status_json}" | /usr/bin/python3 -c '
import json, sys
try:
    status = json.load(sys.stdin)
    self_status = status.get("Self", {})
    print(status.get("BackendState", ""))
    print("true" if self_status.get("InNetworkMap") else "false")
    for item in status.get("Health", []):
        print(item)
except (ValueError, TypeError):
    raise SystemExit(2)
' )" || return 1

    if [[ "$(printf '%s\n' "${backend_state}" | sed -n '1p')" != "Running" ]]; then
        return 1
    fi
    if [[ "$(printf '%s\n' "${backend_state}" | sed -n '2p')" != "true" ]]; then
        return 1
    fi

    health="$(printf '%s\n' "${backend_state}" | tail -n +3)"
    # The current Jetson image reports an unrelated iptables/nftables warning
    # permanently. Only reconnect for loss of the coordination/control path.
    if printf '%s\n' "${health}" | grep -Eiq \
        'coordination server|network map|map poll|controlplane|control.*connection'; then
        return 1
    fi
    return 0
}

restart_tailscale() {
    echo "$(date --iso-8601=seconds) Tailscale unhealthy; restarting tailscaled"
    /usr/bin/systemctl restart tailscaled
}

last_restart=0
while true; do
    now="$(date +%s)"
    if ! status_is_healthy; then
        if (( now - last_restart >= RESTART_COOLDOWN_S )); then
            restart_tailscale
            last_restart="${now}"
        else
            echo "$(date --iso-8601=seconds) Tailscale still unhealthy; restart cooldown active"
        fi
    fi
    sleep "${CHECK_INTERVAL_S}"
done