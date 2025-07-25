import serial

ser = serial.Serial('/dev/ttyUSB0', 115200)

def get_imu_data():
    try:
        line = ser.readline().decode('utf-8').rstrip()
        ax, ay, az, gx, gy, gz = map(float, line.split(","))

        return ax, ay, az, gx, gy, gz
    
    except Exception as e:
        print(f"センサーデータの読み取り中にエラーが発生しました： {e}")
        return None, None, None, None, None, None
