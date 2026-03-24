### This script connects to a camera using the GenICam interface,
### retrieves its nodemap, and lists all available camera features.

from itala import itala

system = itala.create_system()
devices_info = system.enumerate_devices(500)

if len(devices_info) == 0:
    print("No devices found. Exiting.")
    exit(1)
if devices_info[0].access_status != itala.DeviceAccessStatus_AvailableReadWrite:
    print("Device not accessible in RW mode. Exiting.")
    exit(1)


device = system.create_device(devices_info[0])

datastream_nodemap = device.datastream_node_map
datastream_nodemap.StreamBufferHandlingMode.from_string("NewestOnly")

print("Device initialized.")
nodemap = device.node_map
print("=== Available camera features ===")
for node in nodemap.nodes:
    print(node.
