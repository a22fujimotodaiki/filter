import serial
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.spatial.transform import Rotation as R
import csv

# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# ★変更：バイアス推定を行わない1次元のカルマンフィルタ
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
class KalmanFilter:
    def __init__(self):
        # 状態変数 [角度] の1次元（スカラー）
        self.x = 0.0
        
        # 状態の不確かさ（スカラー）
        self.P = 1.0
        
        # プロセスノイズ（スカラー）
        # ジャイロの角速度の時間変化の不確かさ
        self.Q = 0.001
        
        # 観測ノイズ（スカラー）
        # 加速度センサから計算した角度の不確かさ
        self.R = 0.03

    def predict(self, gyro_rate, dt):
        """予測ステップ"""
        # 状態方程式に基づいて次状態を予測
        self.x = self.x + gyro_rate * dt
        
        # 誤差の不確かさを更新 (P = P + Q)
        self.P = self.P + self.Q
        
        return self.x # 予測した角度を返す

    def update(self, accel_angle):
        """更新ステップ"""
        # カルマンゲインを計算
        K = self.P / (self.P + self.R)
        
        # 観測値で状態を更新
        self.x = self.x + K * (accel_angle - self.x)
        
        # 誤差の不確かさを更新
        self.P = (1 - K) * self.P
        
        return self.x # 更新後の角度を返す

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

def main():
    SERIAL_PORT = 'COM4' # ご自身の環境に合わせて変更してください
    BAUD_RATE = 115200

    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    except serial.SerialException as e:
        print(f"エラー: シリアルポートに接続できません。 {e}")
        return

    print("--- センサーの初期化とキャリブレーション ---")
    print("3秒間、センサーを静止させてください...")
    
    accel_samples = []
    gyro_samples = []
    start_time = time.time()
    while time.time() - start_time < 3.0:
        accel, gyro = get_imu_data(ser)
        if accel is not None and gyro is not None:
            accel_samples.append(accel)
            gyro_samples.append(gyro)
    
    if not accel_samples or not gyro_samples:
        print("\nエラー: キャリブレーションデータを取得できませんでした。")
        ser.close()
        return

    accel_avg = np.mean(accel_samples, axis=0)
    initial_roll = np.arctan2(accel_avg[1], accel_avg[2])
    initial_pitch = np.arctan2(-accel_avg[0], np.sqrt(accel_avg[1]**2 + accel_avg[2]**2))
    
    gyro_offset = np.mean(gyro_samples, axis=0)
    
    print("キャリブレーション完了。")
    print(f"ジャイロオフセット: {gyro_offset}")
    print(f"初期角度(Roll, Pitch): {np.rad2deg(initial_roll):.2f}, {np.rad2deg(initial_pitch):.2f}")

    # --- 変数の初期化 ---
    position = np.zeros(2)
    velocity = np.zeros(2)
    path_history = [position.copy()]
    timestamps = [0.0]

    # カルマンフィルタのインスタンスを作成
    kf_roll = KalmanFilter()
    kf_pitch = KalmanFilter()
    # 初期角度を設定
    kf_roll.x = initial_roll
    kf_pitch.x = initial_pitch
    
    roll, pitch, yaw = initial_roll, initial_pitch, 0.0

    print("\n--- 計測を開始します ---")
    measurement_start_time = time.time()
    last_time = measurement_start_time
    
    try:
        while True:
            accel, gyro = get_imu_data(ser)
            if accel is not None and gyro is not None:
                current_time = time.time()
                dt = current_time - last_time
                if dt <= 0: continue
                last_time = current_time
                
                # ★重要：ここでバイアスを引くことで、フィルタはバイアス補正済みの値を受け取る
                gyro -= gyro_offset

                # --- 1. カルマンフィルタで姿勢を計算 ---
                # 加速度センサーから角度を計算
                accel_roll = np.arctan2(accel[1], accel[2])
                accel_pitch = np.arctan2(-accel[0], np.sqrt(accel[1]**2 + accel[2]**2))
                
                # カルマンフィルタの予測と更新ステップを実行
                roll_predicted = kf_roll.predict(gyro[0], dt)
                roll = kf_roll.update(accel_roll)
                
                pitch_predicted = kf_pitch.predict(gyro[1], dt)
                pitch = kf_pitch.update(accel_pitch)

                # ヨー角はジャイロの積分のみ（ドリフトします）
                yaw += gyro[2] * dt
                
                # --- 2. 姿勢から重力成分を除去 ---
                r = R.from_euler('xyz', [roll, pitch, yaw], degrees=False)
                accel_world = r.apply(accel)
                gravity_world = np.array([0, 0, -1.0])
                linear_accel_world = accel_world - gravity_world

                # --- 3. 経路を計算 ---
                velocity += linear_accel_world[:2] * dt
                position += velocity * dt
                path_history.append(position.copy())
                timestamps.append(current_time - measurement_start_time)

                print(f"Roll:{np.rad2deg(roll):6.1f}, Pitch:{np.rad2deg(pitch):6.1f}, Yaw:{np.rad2deg(yaw):6.1f} | Pos:(x={position[0]:.2f}, y={position[1]:.2f})", end='\r')

    except KeyboardInterrupt:
        print("\n\n計測を停止しました。")
        # （CSV保存処理は省略）
        if len(path_history) > 1:
            csv_filename = 'path_data_kalman.csv'
            print(f"位置データを '{csv_filename}' に保存しています...")
            try:
                with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Timestamp (s)', 'X Position (m)', 'Y Position (m)'])
                    for ts, pos in zip(timestamps, path_history):
                        writer.writerow([ts, pos[0], pos[1]])
                print("保存が完了しました。")
            except Exception as e:
                print(f"CSVファイルの保存中にエラーが発生しました: {e}")

    finally:
        ser.close()

    # --- 結果の可視化 ---
    if len(path_history) > 1:
        path = np.array(path_history)
        plt.figure(figsize=(8, 8))
        plt.plot(path[:, 0], path[:, 1], 'c-', label='Movement Path (Kalman)')
        plt.plot(path[0, 0], path[0, 1], 'go', markersize=10, label='Start')
        plt.plot(path[-1, 0], path[-1, 1], 'rs', markersize=10, label='End')
        plt.title('Estimated 2D Path (Kalman Filter)')
        plt.xlabel('X Position [m]')
        plt.ylabel('Y Position [m]')
        plt.legend()
        plt.grid(True)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.savefig('path_with_kalman_filter.png')
        plt.show()
        
if __name__ == "__main__":
    main()