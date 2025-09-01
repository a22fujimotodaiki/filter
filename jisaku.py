import serial
import numpy as np
import matplotlib.pyplot as plt
import time

def get_imu_data(ser):
    """シリアルポートから1行読み込み、IMUデータを返す"""
    try:
        line = ser.readline().decode('utf-8').strip()
        if line:
            ax, ay, az, gx, gy, gz = map(float, line.split(","))
            return np.array([ax, ay, az]), np.array([gx, gy, gz])
    except Exception as e:
        print(f"データ読み取りエラー: {e}")
    return None, None

def euler_from_accel(accel):
    roll = np.artan2(accel[1], accel[2])
    pitch = np.arctan2(-accel[0], np.sqrt(accel[1]**2 + accel[2]**2))
    return np.array([roll, pitch])

def main():
    try:
        ser = serial.Serial('/dev/ttyS3', 115200, timeout=1)
        print("シリアルに接続")
    except serial.SerialException as e:
        print(f"シリアル接続エラー": {e}")
        return
    
    timestamps = []
    accel_angles_history = []
    gyro_angles_history = []
    path_history = []

    gyro_integrated = np.zeros(3)

    position = np.zeros(2)
    velocity = np.zeros(2)
    path_history.append(position.copy())
    

    