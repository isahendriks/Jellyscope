# mac_control

macOS version of `windows_control/monitoring`. The Itala SDK only exists for Windows and Linux,
so here the camera (a GigE Vision device) is read with [Aravis](https://github.com/AravisProject/aravis).
Camera settings and image processing are the same as the Windows `record.py`.

## One-time setup

```bash
brew install aravis gobject-introspection pkg-config cairo
cd Jellyscope
python3 -m venv .venv
source .venv/bin/activate
pip install -r mac_control/requirements-mac.txt
chmod +x mac_control/start_dev.sh
```

Check it works: `python -c "import gi; gi.require_version('Aravis', '0.8')"`.

## Each session

1. Plug in the camera, then run `./mac_control/start_dev.sh`. This gives the Mac IP 192.168.50.1 on the camera's
   network and lists the camera. Set `SERVICE` in the script to your Ethernet adapter's name
   (see `networksetup -listallnetworkservices`).
2. Optional: record a new dark frame with `python mac_control/monitoring/record_dark_frame.py`. The `bkg.png`
   from the Windows folder is copied in already.
3. Set `SAVE_PATH` in `mac_control/monitoring/record.py`, then run `python mac_control/monitoring/record.py`.

## If frames come in incomplete

- If the Mac's firewall asks whether Python may accept incoming connections, click Allow. The image data arrives as UDP.
- Use a wired adapter directly to the camera, not a hub.
- To use jumbo frames, run `sudo networksetup -setMTU <device> 9000` (find the device with `networksetup -listallhardwareports`).
  Then raise `packet_size_target` in `aravis_camera.py`, for example to 8000.
