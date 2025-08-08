import numpy as np
import matplotlib.pyplot as plt

def main():
    # --- シミュレーション設定 ---
    dt = 1.0  # タイムステップ (1秒)
    total_steps = 60 # 合計60秒
    
    # 実際の動き（真値）を生成
    true_accel = np.zeros(total_steps)
    true_accel[5:15] = 0.2  # 5-15秒の間、0.2 m/s^2で加速
    true_accel[30:35] = -0.4 # 30-35秒の間、-0.4 m/s^2で減速

    true_velocity = np.zeros(total_steps)
    true_position = np.zeros(total_steps)
    for i in range(1, total_steps):
        true_velocity[i] = true_velocity[i-1] + true_accel[i-1] * dt
        true_position[i] = true_position[i-1] + true_velocity[i-1] * dt + 0.5 * true_accel[i-1] * dt**2

    # --- センサーデータのシミュレーション ---
    # 加速度計のノイズ
    accel_noise_sigma = 0.05
    measured_accel = true_accel + np.random.randn(total_steps) * accel_noise_sigma
    
    # GPSのノイズ（5秒に1回だけ測位する）
    gps_noise_sigma = 3.0
    gps_measurement = np.full(total_steps, np.nan) # データがない場所はNaN
    for i in range(0, total_steps, 5):
        gps_measurement[i] = true_position[i] + np.random.randn() * gps_noise_sigma

    # --- カルマンフィルターの初期設定 ---
    # 状態 [位置, 速度]
    state = np.array([0.0, 0.0]) 
    P = np.eye(2) * 500  # 初期誤差は大きいと仮定
    
    # 予測モデル
    F = np.array([[1, dt], [0, 1]]) # 状態遷移行列
    B = np.array([0.5 * dt**2, dt]) # 制御入力行列
    
    # ノイズの行列
    q_val = 0.1
    Q = np.array([[0.25*dt**4, 0.5*dt**3], [0.5*dt**3, dt**2]]) * q_val # プロセスノイズ
    R = np.array([[gps_noise_sigma**2]]) # 観測ノイズ

    # 観測モデル
    H = np.array([[1, 0]]) # 位置のみを観測

    # --- データ保存用リスト ---
    estimated_positions = []
    estimated_velocities = []

    # --- フィルター処理ループ ---
    for i in range(total_steps):
        # --- 予測(Predict)ステップ ---
        # 入力uは加速度
        u = measured_accel[i]
        state = F @ state + B * u
        P = F @ P @ F.T + Q
        
        # --- 更新(Update)ステップ ---
        # GPSデータがある時のみ実行
        if not np.isnan(gps_measurement[i]):
            z = gps_measurement[i]
            y = z - H @ state
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            
            state = state + K @ y
            P = (np.eye(2) - K @ H) @ P

        # 結果を保存
        estimated_positions.append(state[0])
        estimated_velocities.append(state[1])
        
    # --- グラフの描画 ---
    time_axis = np.arange(total_steps) * dt
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.suptitle('Kalman Filter for Position/Velocity Estimation (1D)', fontsize=16)

    # 位置のグラフ
    axs[0].plot(time_axis, true_position, 'g-', label='True Position')
    axs[0].plot(time_axis, estimated_positions, 'b-', label='KF Estimated Position', linewidth=2)
    axs[0].scatter(time_axis, gps_measurement, c='r', marker='x', label='GPS Measurement', s=100)
    axs[0].set_ylabel('Position [m]')
    axs[0].legend()
    axs[0].grid(True)

    # 速度のグラフ
    axs[1].plot(time_axis, true_velocity, 'g-', label='True Velocity')
    axs[1].plot(time_axis, estimated_velocities, 'b-', label='KF Estimated Velocity', linewidth=2)
    axs[1].set_ylabel('Velocity [m/s]')
    axs[1].set_xlabel('Time [s]')
    axs[1].legend()
    axs[1].grid(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('kf_position_velocity.png')
    plt.show()
    print("グラフを 'kf_position_velocity.png' として保存しました。")

if __name__ == "__main__":
    main()