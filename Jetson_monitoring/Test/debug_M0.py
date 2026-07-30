
import serial, time
ser = serial.Serial("/dev/ttyACM0", 115200, timeout=2)
print("opened, dtr=", ser.dtr)
ser.write(b"C")
print("write ok")
print(ser.readline())
