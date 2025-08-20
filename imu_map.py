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
    """加速度計の値からロール、ピッチ角を計算"""
    roll = np.arctan2(accel[1], accel[2])
    pitch = np.arctan2(-accel[0], np.sqrt(accel[1]**2 + accel[2]**2))
    return np.array([roll, pitch])

def main():
    # --- シリアルポートの設定 ---
    try:
        # ご自身の環境に合わせてポート名を変更してください (例: 'COM3')
        ser = serial.Serial('/dev/ttyS3', 115200, timeout=1)
        print(f"{ser.name} に接続しました。データの読み取りを開始します... (停止するには Ctrl+C を押してください)")
    except serial.SerialException as e:
        print(f"エラー: シリアルポートに接続できません。 {e}")
        return

    # --- データ保存用リスト ---
    timestamps = []
    accel_angles_history = []
    gyro_angles_history = []
    path_history = []

    # --- ジャイロ積分用の角度変数を初期化 ---
    gyro_integrated_angle = np.zeros(3)  # [roll, pitch, yaw]

    # --- 【変更点】位置推定用の変数を初期化 ---
    position = np.zeros(2)  # [x, y]
    velocity = np.zeros(2)  # [vx, vy]
    path_history.append(position.copy()) # 初期位置を保存

    # --- 時間計測の開始 ---
    start_time = time.time()
    last_time = start_time

    try:
        while True:
            accel, gyro = get_imu_data(ser)
            
            if accel is not None and gyro is not None:
                current_time = time.time()
                dt = current_time - last_time
                if dt <= 0:
                    continue
                last_time = current_time
                
                # --- 1. 加速度から角度を計算 ---
                accel_angle = euler_from_accel(accel)
                
                # --- 2. ジャイロを積分して角度を更新 ---
                gyro_integrated_angle += gyro * dt
                
                # --- 3. 【追加】移動経路を計算 (条件2) ---
                # ヨー角を使って、センサー座標系の加速度(x,y)をワールド座標系に変換
                # ドリフトは無視し、重力加速度の影響も単純化して計算します
                yaw = gyro_integrated_angle[2]
                world_accel_x = accel[0] * np.cos(yaw) - accel[1] * np.sin(yaw)
                world_accel_y = accel[0] * np.sin(yaw) + accel[1] * np.cos(yaw)
                
                # 速度と位置を積分して更新
                velocity += np.array([world_accel_x, world_accel_y]) * dt
                position += velocity * dt
                
                # --- データを保存 ---
                timestamps.append(current_time - start_time)
                accel_angles_history.append(accel_angle.copy())
                gyro_angles_history.append(gyro_integrated_angle.copy())
                path_history.append(position.copy()) # 計算した位置を保存
                
                # --- 【変更点】ターミナルにジャイロの角度のみ表示 (条件1) ---
                print(
                    f"Gyro [Roll,Pitch,Yaw]: [{np.rad2deg(gyro_integrated_angle[0]):6.2f}, {np.rad2deg(gyro_integrated_angle[1]):6.2f}, {np.rad2deg(gyro_integrated_angle[2]):6.2f}]",
                    end='\r'
                )

    except KeyboardInterrupt:
        print("\nプログラムを停止します。グラフを生成しています...")
    finally:
        ser.close()
        print("シリアルポートを閉じました。")

    # --- 【変更点】グラフの描画 (条件3) ---
    if not timestamps:
        print("データが収集されなかったため、グラフは生成されません。")
        return
        
    accel_deg = np.rad2deg(np.array(accel_angles_history))
    gyro_deg = np.rad2deg(np.array(gyro_angles_history))
    path = np.array(path_history)

    # グラフのレイアウトを設定 (角度グラフとマップを1枚に描画)
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, 2, width_ratios=[2, 3]) # 3行2列、グラフエリアの幅の比率を2:3に
    fig.suptitle('IMU Data Analysis: Angle Variation & Estimated Path', fontsize=18)

    # --- 左側: 角度グラフ ---
    ax_roll = fig.add_subplot(gs[0, 0])
    ax_pitch = fig.add_subplot(gs[1, 0], sharex=ax_roll)
    ax_yaw = fig.add_subplot(gs[2, 0], sharex=ax_roll)
    
    # Roll
    ax_roll.plot(timestamps, accel_deg[:, 0], 'r-', label='Roll (Accel)', linewidth=1.5, alpha=0.7)
    ax_roll.plot(timestamps, gyro_deg[:, 0], 'b-', label='Roll (Gyro)', linewidth=1.5)
    ax_roll.set_ylabel('Roll Angle [deg]')
    ax_roll.set_title('Angle Variation')
    ax_roll.legend()
    ax_roll.grid(True)
    plt.setp(ax_roll.get_xticklabels(), visible=False)

    # Pitch
    ax_pitch.plot(timestamps, accel_deg[:, 1], 'r-', label='Pitch (Accel)', linewidth=1.5, alpha=0.7)
    ax_pitch.plot(timestamps, gyro_deg[:, 1], 'b-', label='Pitch (Gyro)', linewidth=1.5)
    ax_pitch.set_ylabel('Pitch Angle [deg]')
    ax_pitch.legend()
    ax_pitch.grid(True)
    plt.setp(ax_pitch.get_xticklabels(), visible=False)

    # Yaw
    ax_yaw.plot(timestamps, gyro_deg[:, 2], 'g-', label='Yaw (Gyro)', linewidth=1.5)
    ax_yaw.set_ylabel('Yaw Angle [deg]')
    ax_yaw.set_xlabel('Time [s]')
    ax_yaw.legend()
    ax_yaw.grid(True)

    # --- 右側: 移動経路マップ ---
    ax_map = fig.add_subplot(gs[:, 1])
    ax_map.plot(path[:, 0], path[:, 1], 'm-', label='Estimated Path', linewidth=2)
    ax_map.plot(path[0, 0], path[0, 1], 'go', markersize=10, label='Start') # 開始地点
    ax_map.plot(path[-1, 0], path[-1, 1], 'rs', markersize=10, label='End') # 終了地点
    ax_map.set_title('Estimated 2D Path (Drift Ignored)')
    ax_map.set_xlabel('X Position [m]')
    ax_map.set_ylabel('Y Position [m]')
    ax_map.legend()
    ax_map.grid(True)
    ax_map.set_aspect('equal', adjustable='box') # X軸とY軸のスケールを合わせる

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('imu_analysis_result.png')
    print("グラフを 'imu_analysis_result.png' として保存しました。")

if __name__ == "__main__":
    main()