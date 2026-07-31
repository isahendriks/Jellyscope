#%% import packages
import sys
from pathlib import Path
import serial 

MONITOR_DIR = Path(__file__).resolve().parent.parent / "Monitor"
if str(MONITOR_DIR) not in sys.path:
    sys.path.insert(0, str(MONITOR_DIR))

import config

#%%  Set parameters for serial communication
serial_port = config.METRO_M0_SERIAL_PORT
baud_rate = config.BAUD_RATE
serial_timeout_s = config.SERIAL_TIMEOUT_S

ser = serial.Serial(serial_port, baud_rate, timeout=serial_timeout_s)  # open serial port
ser.write(b"C")

#%% Run serial reading loop
try:
    while True:
        line = ser.readline().decode("utf-8").rstrip()  # read a '\n' terminated line and decode to string
        print(line)
except KeyboardInterrupt:
    ser.close()  # close serial port

# %%
