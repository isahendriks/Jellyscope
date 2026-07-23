sudo ip addr flush dev enp1s0
sudo ip addr add 169.254.141.1/16 dev enp1s0
sudo ip link set enp1s0 up
/opt/itala-sdk/bin/itala-ipconfig list
