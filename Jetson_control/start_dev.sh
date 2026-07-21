sudo ip addr flush dev enx00e04ca32607
sudo ip addr add 169.254.141.1/16 dev enx00e04ca32607
sudo ip link set enx00e04ca32607 up
/opt/itala-sdk/bin/itala-ipconfig list