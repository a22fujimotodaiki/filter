import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def euler_from_accel(accel):
    """加速度計の値からロール、ピッチ角を計算"""
    roll = np.arctan2(accel[1], accel[2])
    pitch = np.arctan2(-accel[0], np.sqrt(accel[1]**2 + accel[2]**2))
    return np.array([roll, pitch])

def main():
    # --- データの準備 ---
    try:
        df = pd.read_csv('imu_data2.csv', encoding='shift_jis')
    except FileNotFoundError:
        print("エラー: 'imu_data.csv' が見つかりません。")
        return

    # --- EKFの初期設定 ---
    # 状態ベクトル [roll, pitch, bias_x, bias_y]
    state = np.zeros(4) 
    P = np.eye(4) * 0.1

    # プロセスノイズ Q (ジャイロのドリフトなど)
    Q = np.diag([0.001, 0.001, 0.0001, 0.0001])
    
    # 観測ノイズ R (加速度計のノイズ)
    R = np.diag([0.03, 0.03])

    # 観測行列 H
    H = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0]
    ])
    
    # 結果を保存するリスト
    timestamps = df['timestamp'].values
    estimated_states = []
    raw_angles = []

    # --- ループ処理でフィルタを実行 ---
    for i in range(len(df)):
        dt = timestamps[i] - timestamps[i-1] if i > 0 else timestamps[1] - timestamps[0]
        
        # --- 予測(Predict)ステップ ---
        gyro = df[['gyro_x', 'gyro_y']].iloc[i].values
        phi, theta = state[0], state[1]
        bias_x, bias_y = state[2], state[3]
        
        p, q = gyro - np.array([bias_x, bias_y])

        # 状態遷移行列 F (ヤコビアン)
        F = np.eye(4)
        F[0,1] = p * np.cos(phi) * np.tan(theta) - q * np.sin(phi) * np.tan(theta)
        F[0,2] = -dt
        F[0,3] = -dt * np.sin(phi) * np.tan(theta)
        F[1,3] = -dt * np.cos(phi)

        # 角度の時間変化率
        phi_dot = p + q * np.sin(phi) * np.tan(theta)
        theta_dot = q * np.cos(phi)

        # 状態を更新
        state[0] += dt * phi_dot
        state[1] += dt * theta_dot
        
        P = F @ P @ F.T + Q

        # --- 更新(Update)ステップ ---
        accel = df[['acc_x', 'acc_y', 'acc_z']].iloc[i].values
        z = euler_from_accel(accel)
        raw_angles.append(z)
        
        y = z - H @ state
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        
        state = state + K @ y
        P = (np.eye(4) - K @ H) @ P
        
        estimated_states.append(state.copy())

    # --- 結果のプロット準備 ---
    estimated_states = np.array(estimated_states)
    raw_angles = np.array(raw_angles)
    estimated_deg = np.rad2deg(estimated_states[:, :2])
    raw_deg = np.rad2deg(raw_angles)
    
    # --- グラフ描画 ---
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle('6-axis IMU Attitude Estimation (EKF)', fontsize=16)

    # ロール角
    axs[0].plot(timestamps, raw_deg[:, 0], 'r-', label='Raw Roll', alpha=0.5)
    axs[0].plot(timestamps, estimated_deg[:, 0], 'b-', label='EKF Roll', linewidth=2)
    axs[0].set_ylabel('Roll [deg]')
    axs[0].legend()
    axs[0].grid(True)

    # ピッチ角
    axs[1].plot(timestamps, raw_deg[:, 1], 'r-', label='Raw Pitch', alpha=0.5)
    axs[1].plot(timestamps, estimated_deg[:, 1], 'b-', label='EKF Pitch', linewidth=2)
    axs[1].set_ylabel('Pitch [deg]')
    axs[1].legend()
    axs[1].grid(True)
    
    axs[1].set_xlabel('Time [s]')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    plt.savefig('ahrs_6axis_result.png')
    print("グラフを 'ahrs_6axis_result.png' として保存しました。")


if __name__ == "__main__":
    main()