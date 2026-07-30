#!/usr/bin/env python3
"""List Itala cameras visible on the network, with IP/access-mode info."""

from itala import itala

def list_cameras():
    system = itala.create_system()
    devices = system.enumerate_devices(500)

    if devices.size() == 0:
        print("No devices found.")
        system.dispose()
        return 1

    print(f"{devices.size()} device(s) found.")
    for device in devices:
        print(f"Device: {device.serial_number}")
        print(f"\tDisplay name:  {device.display_name}")
        print(f"\tModel:         {device.model}")
        print(f"\tMAC address:   {device.mac_address_as_string}")
        print(f"\tIP address:    {device.ip_address_as_string}")
        print(f"\tAccess status: {device.access_status}")

    system.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(list_cameras())
