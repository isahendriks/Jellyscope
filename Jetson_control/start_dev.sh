#!/usr/bin/env bash
# Finds the camera and the strobe controller regardless of which USB-C
# ethernet dongle each is currently plugged into, then makes each
# interface's static IP persistent via NetworkManager (avoids the
# "Invalid scope value" tug-of-war between raw `ip` commands and NM
# re-applying its own stored profile).

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

STROBE_IP="192.168.0.32"
STROBE_HOST_ADDR="192.168.0.10/24"
CAMERA_HOST_ADDR="169.254.141.1/24"
CAMERA_MTU="9000" # jumbo frame, must be >= the camera's GevSCPSPacketSize (~3992 in acquire_images.py)
CAMERA_RX_RING="4096" # NIC's advertised max; default of 100 is too small for bursty GVSP streaming
RMEM_SIZE="16777216" # UDP receive socket buffer, default 212992 overflows under load
SYSCTL_FILE="/etc/sysctl.d/99-gige-camera.conf"

# enxc84d4427a643 is dedicated to the 5G router (static 192.168.1.11, set up
# separately and persistently via NetworkManager) -- it's the only remote
# access path once deployed, so this script must never touch its IP config.
ROUTER_IFACE="enxc84d4427a643"

# USB ethernet dongles + the onboard port, minus the router's interface.
mapfile -t IFACES < <(
  { ip -o link show | awk -F': ' '{print $2}' | grep '^enx'; echo end0; } \
    | grep -vx "${ROUTER_IFACE}"
)

if [ "${#IFACES[@]}" -eq 0 ]; then
  echo "No candidate interfaces (enx*/end0) found. Is anything plugged in?" >&2
  exit 1
fi

echo "==> Candidate USB dongles: ${IFACES[*]}"

nm_connection_for() {
  nmcli -t -f GENERAL.CONNECTION device show "$1" 2>/dev/null | cut -d: -f2
}

wait_for_link() {
  local iface="$1" timeout="${2:-8}"
  for ((i = 0; i < timeout; i++)); do
    ethtool "${iface}" 2>/dev/null | grep -q "Link detected: yes" && return 0
    sleep 1
  done
  return 1
}

# nvethernet (Jetson's onboard end0) rejects MTU changes while the device is
# admin-up ("must be stopped to change its MTU" in dmesg). `ip link set down`
# and NetworkManager's live reapply both return before the driver has actually
# quiesced the device, so we poll the kernel's own IFF_UP flag rather than
# assuming the transition is instant.
wait_for_admin_down() {
  local iface="$1" timeout="${2:-5}" flags
  for ((i = 0; i < timeout * 10; i++)); do
    flags="$(cat "/sys/class/net/${iface}/flags" 2>/dev/null)" || return 1
    (( flags & 0x1 )) || return 0
    sleep 0.1
  done
  return 1
}

set_static_ip() {
  local iface="$1" addr="$2" mtu="${3:-}"
  local con current_mtu
  current_mtu="$(ip -o link show dev "${iface}" | grep -oP 'mtu \K[0-9]+')"
  con="$(nm_connection_for "${iface}")"

  if [ -z "${con}" ]; then
    if ip -4 addr show dev "${iface}" | grep -qF "${addr}" \
       && { [ -z "${mtu}" ] || [ "${current_mtu}" = "${mtu}" ]; }; then
      echo "    ${iface} already has ${addr} and correct MTU, skipping reconfiguration" >&2
      sudo ip link set dev "${iface}" up
      return
    fi
    echo "    No NetworkManager connection bound to ${iface}, using raw ip commands" >&2
    sudo ip addr flush dev "${iface}"
    if [ -n "${mtu}" ] && [ "${current_mtu}" != "${mtu}" ]; then
      # Some drivers (e.g. nvethernet on end0) refuse MTU changes while admin-up.
      sudo ip link set dev "${iface}" down
      wait_for_admin_down "${iface}" \
        || echo "    WARNING: ${iface} didn't go admin-down in time, MTU change may be rejected" >&2
      sudo ip link set dev "${iface}" mtu "${mtu}"
    fi
    sudo ip link set dev "${iface}" up
    wait_for_link "${iface}" || echo "    WARNING: ${iface} link didn't come back up in time" >&2
    sudo ip addr add "${addr}" dev "${iface}"
    return
  fi
  sudo nmcli con mod "${con}" ipv4.method manual ipv4.addresses "${addr}" ipv4.gateway ""
  if [ -n "${mtu}" ] && [ "${current_mtu}" != "${mtu}" ]; then
    sudo nmcli con mod "${con}" 802-3-ethernet.mtu "${mtu}"
    # nmcli can reapply MTU live on an already-active connection without a real
    # down/up cycle; force the kernel-level down nvethernet actually requires.
    sudo ip link set dev "${iface}" down
    wait_for_admin_down "${iface}" \
      || echo "    WARNING: ${iface} didn't go admin-down in time, MTU change may be rejected" >&2
  fi
  sudo nmcli con up "${con}" >/dev/null
  wait_for_link "${iface}" || echo "    WARNING: ${iface} link didn't come back up in time" >&2
}

STROBE_IFACE=""
CAMERA_IFACE=""

echo "==> Looking for the strobe controller (${STROBE_IP})..."
for iface in "${IFACES[@]}"; do
  sudo ip link set "${iface}" up
  sudo ip addr add "${STROBE_HOST_ADDR}" dev "${iface}" 2>/dev/null
  if ping -c 1 -W 1 -I "${iface}" "${STROBE_IP}" >/dev/null 2>&1; then
    echo "    Strobe controller found on ${iface}"
    STROBE_IFACE="${iface}"
  else
    sudo ip addr del "${STROBE_HOST_ADDR}" dev "${iface}" 2>/dev/null
  fi
done

echo "==> Looking for the camera (GVCP discovery)..."
source "${SCRIPT_DIR}/.venv-jellyscope/bin/activate"
for iface in "${IFACES[@]}"; do
  [ "${iface}" == "${STROBE_IFACE}" ] && continue
  sudo ip link set "${iface}" up
  sudo ip addr add "${CAMERA_HOST_ADDR}" dev "${iface}" 2>/dev/null
  if python "${SCRIPT_DIR}/list_cameras.py" >/dev/null 2>&1; then
    echo "    Camera found on ${iface}"
    CAMERA_IFACE="${iface}"
    break
  else
    sudo ip addr del "${CAMERA_HOST_ADDR}" dev "${iface}" 2>/dev/null
  fi
done

if [ -z "${STROBE_IFACE}" ]; then
  echo "WARNING: strobe controller not found on any interface." >&2
fi
if [ -z "${CAMERA_IFACE}" ]; then
  echo "ERROR: camera not found on any interface." >&2
  exit 1
fi

echo "==> Making the config persistent via NetworkManager..."
[ -n "${STROBE_IFACE}" ] && set_static_ip "${STROBE_IFACE}" "${STROBE_HOST_ADDR}"
set_static_ip "${CAMERA_IFACE}" "${CAMERA_HOST_ADDR}" "${CAMERA_MTU}"

ACTUAL_MTU="$(ip -o link show "${CAMERA_IFACE}" | grep -oP 'mtu \K[0-9]+')"
echo "    ${CAMERA_IFACE} MTU is now ${ACTUAL_MTU} (requested ${CAMERA_MTU})"
if [ "${ACTUAL_MTU}" -lt "${CAMERA_MTU}" ]; then
  echo "WARNING: adapter/driver clamped MTU below the requested jumbo size. GVSP packets (~3992 bytes) may still get dropped." >&2
fi

echo "==> Tuning host network buffers for camera streaming..."
sudo ethtool -G "${CAMERA_IFACE}" rx "${CAMERA_RX_RING}" 2>/dev/null \
  || echo "    (RX ring buffer bump not supported by this adapter, skipping)" >&2

if [ ! -f "${SYSCTL_FILE}" ] || ! grep -q "rmem_max=${RMEM_SIZE}" "${SYSCTL_FILE}" 2>/dev/null; then
  printf 'net.core.rmem_max=%s\nnet.core.rmem_default=%s\n' "${RMEM_SIZE}" "${RMEM_SIZE}" \
    | sudo tee "${SYSCTL_FILE}" >/dev/null
fi
sudo sysctl -p "${SYSCTL_FILE}" >/dev/null

echo "==> Final camera listing:"
python "${SCRIPT_DIR}/list_cameras.py"
