import serial
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.spatial.transform import Rotation as R

def get_imu_data(ser):
    """シリアルポートから1行読み込み、IMUデータを返す"""
    try:
        line = ser.readline().decode('utf-8').strip()
        if line:
            ax, ay, az, gx, gy, gz = map(float, line.split(","))
            # ジャイロの単位が deg/s の場合はラジアンに変換
            # return np.array([ax, ay, az]), np.deg2rad(np.array([gx, gy, gz]))
            return np.array([ax, ay, az]), np.array([gx, gy, gz]) # 単位が rad/s の場合
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

    # 静止時の加速度の平均値から初期の傾きを計算
    accel_avg = np.mean(accel_samples, axis=0)
    initial_roll = np.arctan2(accel_avg[1], accel_avg[2])
    initial_pitch = np.arctan2(-accel_avg[0], np.sqrt(accel_avg[1]**2 + accel_avg[2]**2))
    
    # ジャイロのオフセット（バイアス）を計算
    gyro_offset = np.mean(gyro_samples, axis=0)
    
    print("キャリブレーション完了。")
    print(f"ジャイロオフセット: {gyro_offset}")
    print(f"初期角度(Roll, Pitch): {np.rad2deg(initial_roll):.2f}, {np.rad2deg(initial_pitch):.2f}")

    # --- 変数の初期化 ---
    position = np.zeros(2)    # 2次元位置 [x, y]
    velocity = np.zeros(2)    # 2次元速度
    path_history = [position.copy()]

    # 相補フィルター用の変数
    roll, pitch, yaw = initial_roll, initial_pitch, 0.0
    alpha = 0.98  # ジャイロを信頼する割合 (大きいほどジャイロ重視)

    last_time = time.time()
    print("\n--- 計測を開始します ---")
    
    try:
        while True:
            accel, gyro = get_imu_data(ser)
            if accel is not None and gyro is not None:
                current_time = time.time()
                dt = current_time - last_time
                if dt <= 0: continue
                last_time = current_time
                
                # ジャイロのオフセットを引く
                gyro -= gyro_offset

                # --- 1. 相補フィルターで姿勢を計算 ---
                # 加速度センサーから計算した角度
                accel_roll = np.arctan2(accel[1], accel[2])
                accel_pitch = np.arctan2(-accel[0], np.sqrt(accel[1]**2 + accel[2]**2))

                # ジャイロを積分した角度と、加速度から求めた角度を合成
                roll = alpha * (roll + gyro[0] * dt) + (1 - alpha) * accel_roll
                pitch = alpha * (pitch + gyro[1] * dt) + (1 - alpha) * accel_pitch
                yaw += gyro[2] * dt # ヨー角はジャイロの積分のみ（ドリフトします）

                # --- 2. 姿勢から重力成分を除去 ---
                # 回転オブジェクトを作成
                r = R.from_euler('xyz', [roll, pitch, yaw], degrees=False)
                # センサー座標系の加速度をワールド座標系に変換
                accel_world = r.apply(accel)
                # ワールド座標系の重力ベクトル (仮に[0, 0, -9.8]とする)
                gravity_world = np.array([0, 0, -1.0]) # 正規化されていると仮定
                # 純粋な移動加速度
                linear_accel_world = accel_world - gravity_world

                # --- 3. 経路を計算 ---
                velocity += linear_accel_world[:2] * dt
                position += velocity * dt
                path_history.append(position.copy())

                print(f"Roll:{np.rad2deg(roll):6.1f}, Pitch:{np.rad2deg(pitch):6.1f}, Yaw:{np.rad2deg(yaw):6.1f} | Pos:(x={position[0]:.2f}, y={position[1]:.2f})", end='\r')

    except KeyboardInterrupt:
        print("\n\n計測を停止しました。")
    finally:
        ser.close()

    # --- 結果の可視化 ---
    if len(path_history) > 1:
        path = np.array(path_history)
        plt.figure(figsize=(8, 8))
        plt.plot(path[:, 0], path[:, 1], 'm-', label='Movement Path')
        plt.plot(path[0, 0], path[0, 1], 'go', markersize=10, label='Start')
        plt.plot(path[-1, 0], path[-1, 1], 'rs', markersize=10, label='End')
        plt.title('Estimated 2D Path (Complementary Filter)')
        plt.xlabel('X Position [m]')
        plt.ylabel('Y Position [m]')
        plt.legend()
        plt.grid(True)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.savefig('path_with_complementary_filter.png')
        plt.show()

if __name__ == "__main__":
    main()