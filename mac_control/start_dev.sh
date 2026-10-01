#!/bin/bash
# macOS version of windows_control/start_dev.ps1 - run with: ./start_dev.sh
# (uses sudo, changing IP settings needs it)
# Set SERVICE to the name of the Ethernet port/adapter the camera is plugged into
# (list them with: networksetup -listallnetworkservices)
SERVICE="USB 10/100/1000 LAN"
# The camera (ITA204, serial 602295) has a fixed IP of 192.168.50.10/24, so the Mac goes in the same subnet
MAC_IP="192.168.50.1"
SUBNET_MASK="255.255.255.0"

# Set-NetIPInterface -Dhcp Disabled + New-NetIPAddress
sudo networksetup -setmanual "$SERVICE" "$MAC_IP" "$SUBNET_MASK"

# Larger socket buffers so full frames don't drop packets (reset on reboot)
sudo sysctl -w kern.ipc.maxsockbuf=16777216 > /dev/null

# list the cameras (arv-tool comes with brew install aravis)
sleep 3 # give the adapter a moment to come up
arv-tool-0.8 || echo "No camera found - check the cable, SERVICE name and that 'brew install aravis' was run"
