import serial
import numpy as np
import matplotlib.pyplot as plt
import time

# --- センサーデータ取得関数 (変更なし) ---
def get_imu_data(ser):
    """シリアルポートから1行読み込み、IMUデータを返す"""
    try:
        line = ser.readline().decode('utf-8').strip()
        if line:
            # 受信するデータが6軸(ax,ay,az,gx,gy,gz)であることを想定
            ax, ay, az, gx, gy, gz = map(float, line.split(","))
            return np.array([ax, ay, az]), np.array([gx, gy, gz])
    except Exception as e:
        print(f"データ読み取りエラー: {e}")
    return None, None

# --- 加速度から角度を計算する関数 (変更なし) ---
def euler_from_accel(accel):
    """加速度計の値からロール、ピッチ角を計算"""
    # roll: y軸とz軸周りの回転
    roll = np.arctan2(accel[1], accel[2])
    # pitch: x軸とsqrt(y^2+z^2)軸周りの回転
    pitch = np.arctan2(-accel[0], np.sqrt(accel[1]**2 + accel[2]**2))
    return np.array([roll, pitch])

def main():
    # --- シリアルポートの設定 ---
    try:
        # ポート名は環境に合わせて変更してください (例: 'COM3' on Windows)
        ser = serial.Serial('/dev/ttyS3', 115200, timeout=1)
        print(f"{ser.name} に接続しました。データの読み取りを開始します... (停止するには Ctrl+C を押してください)")
    except serial.SerialException as e:
        print(f"エラー: シリアルポートに接続できません。 {e}")
        return

    # --- データ保存用リスト ---
    timestamps = []
    accel_angles_history = []  # 加速度から計算した角度 [roll, pitch]
    gyro_angles_history = []   # ジャイロを積分した角度 [roll, pitch, yaw]

    # --- ジャイロ積分用の角度変数を初期化 ---
    gyro_integrated_angle = np.zeros(3)  # [roll, pitch, yaw]

    # --- 時間計測の開始 ---
    start_time = time.time()
    last_time = start_time

    try:
        while True:
            # センサーデータを取得
            accel, gyro = get_imu_data(ser)
            
            if accel is not None and gyro is not None:
                current_time = time.time()
                dt = current_time - last_time
                if dt <= 0: # 時間間隔が0以下の場合はスキップ
                    continue
                last_time = current_time
                
                # --- 1. 加速度データから角度を計算 ---
                accel_angle = euler_from_accel(accel)
                
                # --- 2. ジャイロデータを単純に積分 ---
                # gyroの単位が rad/s の場合
                gyro_integrated_angle += gyro * dt
                
                # --- データを保存 ---
                timestamps.append(current_time - start_time)
                accel_angles_history.append(accel_angle.copy())
                gyro_angles_history.append(gyro_integrated_angle.copy())
                
                # --- ターミナルに現在の角度を表示 ---
                print(
                    f"Accel[Roll,Pitch]: [{np.rad2deg(accel_angle[0]):6.2f}, {np.rad2deg(accel_angle[1]):6.2f}] | "
                    f"Gyro [Roll,Pitch,Yaw]: [{np.rad2deg(gyro_integrated_angle[0]):6.2f}, {np.rad2deg(gyro_integrated_angle[1]):6.2f}, {np.rad2deg(gyro_integrated_angle[2]):6.2f}]",
                    end='\r'
                )

    except KeyboardInterrupt:
        print("\nプログラムを停止します。グラフを生成しています...")
    finally:
        ser.close()
        print("シリアルポートを閉じました。")

    # --- グラフの描画 ---
    if not timestamps:
        print("データが収集されなかったため、グラフは生成されません。")
        return
        
    # リストをNumPy配列に変換し、ラジアンから度に変換
    accel_deg = np.rad2deg(np.array(accel_angles_history))
    gyro_deg = np.rad2deg(np.array(gyro_angles_history))

    fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    fig.suptitle('IMU Raw Sensor Data Comparison', fontsize=16)

    # --- Rollのグラフ ---
    axs[0].plot(timestamps, accel_deg[:, 0], 'r-', label='Roll from Accelerometer', linewidth=1.5, alpha=0.7)
    axs[0].plot(timestamps, gyro_deg[:, 0], 'b-', label='Roll from Gyroscope (Integrated)', linewidth=1.5)
    axs[0].set_ylabel('Roll Angle [deg]')
    axs[0].legend()
    axs[0].grid(True)

    # --- Pitchのグラフ ---
    axs[1].plot(timestamps, accel_deg[:, 1], 'r-', label='Pitch from Accelerometer', linewidth=1.5, alpha=0.7)
    axs[1].plot(timestamps, gyro_deg[:, 1], 'b-', label='Pitch from Gyroscope (Integrated)', linewidth=1.5)
    axs[1].set_ylabel('Pitch Angle [deg]')
    axs[1].legend()
    axs[1].grid(True)
    
    # --- Yawのグラフ (ジャイロのみ) ---
    axs[2].plot(timestamps, gyro_deg[:, 2], 'g-', label='Yaw from Gyroscope (Integrated)', linewidth=1.5)
    axs[2].set_ylabel('Yaw Angle [deg]')
    axs[2].set_xlabel('Time [s]')
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('imu_raw_data.png')
    print("グラフを 'imu_raw_data.png' として保存しました。")

if __name__ == "__main__":
    main()