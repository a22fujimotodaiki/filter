import serial
import numpy as np
import matplotlib.pyplot as plt
import time

# --- センサーデータ取得関数 ---
def get_imu_data(ser):
    """シリアルポートから1行読み込み、IMUデータを返す"""
    try:
        # データの読み取りとデコード
        line = ser.readline().decode('utf-8').strip()
        
        # データが空でないことを確認
        if line:
            # カンマで分割し、float型に変換
            ax, ay, az, gx, gy, gz = map(float, line.split(","))
            return np.array([ax, ay, az]), np.array([gx, gy, gz])
            
    except Exception as e:
        # エラー内容を表示するが、プログラムは続行
        print(f"データ読み取りエラー: {e}")

    return None, None

# --- 加速度から角度を計算 ---
def euler_from_accel(accel):
    """加速度計の値からロール、ピッチ角を計算"""
    roll = np.arctan2(accel[1], accel[2])
    pitch = np.arctan2(-accel[0], np.sqrt(accel[1]**2 + accel[2]**2))
    return np.array([roll, pitch])

def main():
    # --- シリアルポートの設定 ---
    try:
        # ★ご自身の環境に合わせてポート名 ('/dev/ttyUSB0' や 'COM3' など) を変更してください
        #ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
        ser = serial.Serial('/dev/ttyS3', 115200, timeout=1)

        print(f"{ser.name} に接続しました。データの読み取りを開始します... (停止するには Ctrl+C を押してください)")
    except serial.SerialException as e:
        print(f"エラー: シリアルポートに接続できません。 {e}")
        return

    # --- EKFの初期設定 ---
    # 状態 [roll, pitch, bias_x, bias_y]
    state = np.zeros(4) 
    P = np.eye(4) * 0.1
    Q = np.diag([0.001, 0.001, 0.0001, 0.0001])
    R = np.diag([0.03, 0.03])
    H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    
    # --- データ保存用リスト ---
    timestamps = []
    estimated_states = []
    raw_angles = []
    
    # 時間計測の開始
    start_time = time.time()
    last_time = start_time

    try:
        while True:
            # --- センサーデータの取得 ---
            accel, gyro = get_imu_data(ser)
            
            # データが正常に取得できた場合のみ処理
            if accel is not None and gyro is not None:
                current_time = time.time()
                dt = current_time - last_time
                last_time = current_time
                
                # タイムスタンプを保存
                timestamps.append(current_time - start_time)

                # --- 予測(Predict)ステップ ---
                phi, theta = state[0], state[1]
                bias_x, bias_y = state[2], state[3]
                p, q = gyro[:2] - np.array([bias_x, bias_y]) # gyro_zは使わない

                F = np.eye(4)
                F[0,1] = p * np.cos(phi) * np.tan(theta) - q * np.sin(phi) * np.tan(theta)
                F[0,2] = -dt
                F[0,3] = -dt * np.sin(phi) * np.tan(theta)
                F[1,3] = -dt * np.cos(phi)

                phi_dot = p + q * np.sin(phi) * np.tan(theta)
                theta_dot = q * np.cos(phi)

                state[0] += dt * phi_dot
                state[1] += dt * theta_dot
                
                P = F @ P @ F.T + Q

                # --- 更新(Update)ステップ ---
                z = euler_from_accel(accel)
                raw_angles.append(z)
                
                y = z - H @ state[:4] # 状態と観測の次元を合わせる
                S = H @ P @ H.T + R
                K = P @ H.T @ np.linalg.inv(S)
                
                state = state + K @ y
                P = (np.eye(4) - K @ H) @ P
                
                estimated_states.append(state.copy())
                
                # ターミナルに現在の推定角度を表示（任意）
                print(f"Roll: {np.rad2deg(state[0]):.2f} deg, Pitch: {np.rad2deg(state[1]):.2f} deg", end='\r')

    except KeyboardInterrupt:
        print("\nプログラムを停止します。グラフを生成しています...")
    finally:
        # 必ずシリアルポートを閉じる
        ser.close()
        print("シリアルポートを閉じました。")

    # --- グラフの描画 ---
    if not estimated_states:
        print("データが収集されなかったため、グラフは生成されません。")
        return
        
    estimated_states = np.array(estimated_states)
    raw_angles = np.array(raw_angles)
    estimated_deg = np.rad2deg(estimated_states[:, :2])
    raw_deg = np.rad2deg(raw_angles)
    
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle('Real-time 6-axis IMU Attitude Estimation (EKF)', fontsize=16)

    axs[0].plot(timestamps, raw_deg[:, 0], 'r-', label='Raw Roll (from Accel)', alpha=0.5)
    axs[0].plot(timestamps, estimated_deg[:, 0], 'b-', label='EKF Roll', linewidth=2)
    axs[0].set_ylabel('Roll [deg]')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(timestamps, raw_deg[:, 1], 'r-', label='Raw Pitch (from Accel)', alpha=0.5)
    axs[1].plot(timestamps, estimated_deg[:, 1], 'b-', label='EKF Pitch', linewidth=2)
    axs[1].set_ylabel('Pitch [deg]')
    axs[1].legend()
    axs[1].grid(True)
    
    axs[1].set_xlabel('Time [s]')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    plt.savefig('realtime_filter_result.png')
    print("グラフを 'realtime_filter_result.png' として保存しました。")

if __name__ == "__main__":
    main()