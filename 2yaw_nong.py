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

def main():
    # --- 1. 実行前の準備 ---
    SERIAL_PORT = 'COM4' # ご自身の環境に合わせて変更してください
    BAUD_RATE = 115200

    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"シリアルポート {ser.name} に接続しました。")
    except serial.SerialException as e:
        print(f"エラー: シリアルポートに接続できません。 {e}")
        return

    # --- 2. 重力キャリブレーション (最初の3秒間) ---
    print("\n--- 重力キャリブレーションを開始します ---")
    print("3秒間、センサーを完全に静止させてください...")
    
    cal_time = 10.0
    calibration_samples = []
    start_time = time.time()
    while time.time() - start_time < cal_time:
        accel, gyro = get_imu_data(ser)
        if accel is not None:
            calibration_samples.append(accel)
            print(f"キャリブレーション中...残り {cal_time - (time.time() - start_time):.1f} 秒", end='\r')
    
    if not calibration_samples:
        print("\nエラー: キャリブレーション中にデータを取得できませんでした。")
        ser.close()
        return

    # 平均値を基準の重力ベクトルとする
    gravity_vector = np.mean(calibration_samples, axis=0)
    print("\nキャリブレーション完了。")
    print(f"計測された重力ベクトル: {gravity_vector}")

    # --- 3. 変数の初期化 (計測用) ---
    position = np.zeros(3)  # 3次元位置 [x, y, z]
    velocity = np.zeros(3)  # 3次元速度
    rotation_matrix = np.eye(3) # 回転行列 (初期姿勢は単位行列)
    path_history = [position[:2].copy()] # 2Dの経路を保存
    
    print("\n--- 計測を開始します ---")
    print("センサーを動かしてください... (停止するには Ctrl+C を押してください)")
    last_time = time.time()

    # --- 4. メインループ (計測) ---
    try:
        while True:
            accel, gyro = get_imu_data(ser)
            if accel is not None and gyro is not None:
                current_time = time.time()
                dt = current_time - last_time
                if dt <= 0: continue
                last_time = current_time

                # (1) ジャイロで姿勢(回転行列)を更新
                angle_change = gyro * dt
                angle_magnitude = np.linalg.norm(angle_change)
                if angle_magnitude > 0:
                    axis = angle_change / angle_magnitude
                    # ロドリゲスの回転公式を用いて回転行列を更新
                    K = np.array([[0, -axis[2], axis[1]],
                                  [axis[2], 0, -axis[0]],
                                  [-axis[1], axis[0], 0]])
                    delta_R = np.eye(3) + np.sin(angle_magnitude) * K + (1 - np.cos(angle_magnitude)) * (K @ K)
                    rotation_matrix = rotation_matrix @ delta_R

                # (2) 現在の姿勢における重力ベクトルを計算
                # 基準の重力ベクトルを現在の姿勢に合わせて回転させる
                gravity_in_body_frame = rotation_matrix.T @ gravity_vector

                # (3) 加速度から重力成分を除去
                linear_accel_body = accel - gravity_in_body_frame

                # (4) ワールド座標系での加速度に変換し、経路を計算
                linear_accel_world = rotation_matrix @ linear_accel_body
                velocity += linear_accel_world * dt
                position += velocity * dt
                
                path_history.append(position[:2].copy()) # X-Y平面の経路を保存

                # 現在の状態を表示
                print(f"Position: (x={position[0]:.2f}, y={position[1]:.2f}) [m]", end='\r')

    except KeyboardInterrupt:
        print("\n\n計測を停止しました。")
    finally:
        ser.close()
        print("シリアルポートを閉じました。")

    # --- 5. 結果の可視化 ---
    if len(path_history) > 1:
        path = np.array(path_history)
        plt.figure(figsize=(8, 8))
        plt.plot(path[:, 0], path[:, 1], 'm-', label='Movement Path', linewidth=2)
        plt.plot(path[0, 0], path[0, 1], 'go', markersize=10, label='Start')
        plt.plot(path[-1, 0], path[-1, 1], 'rs', markersize=10, label='End')
        plt.title('Estimated 2D Movement Path (Gravity Compensated)')
        plt.xlabel('X Position [m]')
        plt.ylabel('Y Position [m]')
        plt.legend()
        plt.grid(True)
        plt.gca().set_aspect('equal', adjustable='box')
        
        filename = 'path_result_calibrated.png'
        plt.savefig(filename)
        print(f"グラフを '{filename}' として保存しました。")
        plt.show()

if __name__ == "__main__":
    main()