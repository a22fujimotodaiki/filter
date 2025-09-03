import serial
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import deque

# --- センサーデータ取得関数 ---
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

# --- 加速度から角度を計算 ---
def euler_from_accel(accel):
    """加速度計の値からロール、ピッチ角を計算"""
    roll = np.arctan2(accel[1], accel[2])
    pitch = np.arctan2(-accel[0], np.sqrt(accel[1]**2 + accel[2]**2))
    return np.array([roll, pitch])

def main():
    # --- シリアルポートの設定 ---
    try:
        ser = serial.Serial('COM4', 115200, timeout=1)
        #ser = serial.Serial('/dev/ttyS3', 115200, timeout=1)
        print(f"{ser.name} に接続しました。データの読み取りを開始します... (停止するには Ctrl+C を押してください)")
    except serial.SerialException as e:
        print(f"エラー: シリアルポートに接続できません。 {e}")
        return

    # --- EKFの初期設定 (ロール・ピッチ) ---
    state = np.zeros(4) 
    P = np.eye(4) * 0.1
    Q = np.diag([0.001, 0.001, 0.0001, 0.0001])
    R = np.diag([0.02, 0.02]) 
    H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    
    # --- ヨー角の初期化 ---
    yaw_angle_rad = 0.0

    # --- ★適応化のための追加パラメータ★ ---
    WINDOW_SIZE = 5
    ADAPTATION_FACTOR = 0.05
    innovation_history = deque(maxlen=WINDOW_SIZE)

    # --- データ保存用リスト ---
    timestamps = []
    estimated_states = []
    r_history = []
    raw_angles = [] ### 変更・追加 ###: フィルタ前の生データを保存するリスト
    ### 変更・追加 ###: 角速度積分値を保存するリスト
    gyro_integrated_angles = []

    ### 変更・追加 ###: 角速度を積分するための角度変数を初期化
    gyro_integrated_angle = np.zeros(2) # [roll, pitch]
    start_time = time.time()
    last_time = start_time

    try:
        while True:
            accel, gyro = get_imu_data(ser)
            
            if accel is not None and gyro is not None:
                current_time = time.time()
                dt = current_time - last_time
                last_time = current_time
                
                timestamps.append(current_time - start_time)

                ### 変更・追加 ###: 角速度を単純に積分して「生の値」を計算
                # gyro[:2] は [gx, gy] (ロールとピッチの角速度)
                gyro_integrated_angle += gyro[:2] * dt
                gyro_integrated_angles.append(gyro_integrated_angle.copy())
                # --- 【ロール・ピッチの計算 (EKF)】 ---
                # --- 予測(Predict)ステップ ---
                phi, theta = state[0], state[1]
                bias_x, bias_y = state[2], state[3]
                p, q = gyro[:2] - np.array([bias_x, bias_y])

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
                raw_angles.append(z) ### 変更・追加 ###: 生データをリストに保存
                
                y = z - H @ state[:4]

                # --- ★Rの適応更新ロジック★ ---
                innovation_history.append(y)
                if len(innovation_history) == WINDOW_SIZE:
                    innov_array = np.array(innovation_history)
                    empirical_cov = np.mean([np.outer(inn, inn) for inn in innov_array], axis=0)
                    R = (1 - ADAPTATION_FACTOR) * R + ADAPTATION_FACTOR * empirical_cov
                
                S = H @ P @ H.T + R
                K = P @ H.T @ np.linalg.inv(S)
                
                state = state + K @ y
                P = (np.eye(4) - K @ H) @ P
                
                estimated_states.append(state.copy())
                r_history.append(np.diag(R).copy())
                
                # --- 【ヨー角の計算 (単純積分)】 ---
                gz = gyro[2] 
                yaw_angle_rad += gz * dt
                
                # --- ターミナル表示 (Rの値も表示) ---
                print(f"Roll: {np.rad2deg(state[0]):.2f}, Pitch: {np.rad2deg(state[1]):.2f}, R_roll: {R[0,0]:.4f}, R_pitch: {R[1,1]:.4f}", end='\r')


    except KeyboardInterrupt:
        print("\nプログラムを停止します。グラフを生成しています...")
    finally:
        ser.close()
        print("シリアルポートを閉じました。")

    # --- グラフの描画 ---
    if not estimated_states:
        print("データが収集されなかったため、グラフは生成されません。")
        return
        
    estimated_states = np.array(estimated_states)
    estimated_deg = np.rad2deg(estimated_states[:, :2])
    r_history = np.array(r_history)
    ### 変更・追加 ###: 保存した角速度積分値をNumPy配列に変換し、度に変換
    gyro_integrated_angles = np.array(gyro_integrated_angles)
    gyro_deg = np.rad2deg(gyro_integrated_angles)

    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.suptitle('Adaptive EKF for IMU Attitude Estimation', fontsize=16)

    ### 変更・追加 ###: フィルタ前のデータをグラフに追加
    # RollとPitchのグラフ
    # フィルタ後のデータ（実線）
    axs[0].plot(timestamps, estimated_deg[:, 0], 'b-', label='EKF Roll', linewidth=2)
    axs[0].plot(timestamps, estimated_deg[:, 1], 'g-', label='EKF Pitch', linewidth=2)
    # フィルタ前のデータ（点線）
    axs[1].plot(timestamps, gyro_deg[:, 0], 'c-', label='Raw Roll (from Gyro)', linewidth=1)
    axs[1].plot(timestamps, gyro_deg[:, 1], 'm-', label='Raw Pitch (from Gyro)', linewidth=1)
    axs[1].set_ylabel('Angle [deg]')
    axs[1].legend()
    axs[1].grid(True)

    # Rの値のグラフ
    """
    axs[1].plot(timestamps, r_history[:, 0], 'r-', label='Adaptive R for Roll')
    axs[1].plot(timestamps, r_history[:, 1], 'orange', linestyle='-', label='Adaptive R for Pitch')
    axs[1].set_ylabel('Measurement Noise Covariance (R)')
    axs[1].set_xlabel('Time [s]')
    axs[1].set_yscale('log')
    axs[1].legend()
    axs[1].grid(True, which="both")
    """

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('adaptive_ekf_result_with_raw.png')
    print("グラフを 'adaptive_ekf_result_with_raw.png' として保存しました。")

if __name__ == "__main__":
    main()