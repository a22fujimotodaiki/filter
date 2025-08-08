import numpy as np
import matplotlib.pyplot as plt
import time

# --- シミュレーションデータ生成関数 ---
def generate_simulated_imu_data(t, total_time):
    """
    シミュレーション用のIMUデータを生成する。
    - 2秒後から4秒後まで前方に1.0m/s^2で加速
    - ジャイロには常にわずかなドリフト（ヨー方向）を持たせる
    """
    g = 9.81
    accel = np.array([0.0, 0.0, g])
    gyro = np.array([0.0, 0.0, 0.0])

    if 2.0 < t < 4.0:
        accel[0] = 1.0

    accel_noise = np.random.randn(3) * 0.05
    gyro_noise = np.random.randn(3) * 0.005
    gyro_drift = np.array([0.0, 0.0, 0.02])

    return accel + accel_noise, gyro + gyro_noise + gyro_drift

# --- 加速度から角度を計算 ---
def euler_from_accel(accel):
    roll = np.arctan2(accel[1], accel[2])
    pitch = np.arctan2(-accel[0], np.sqrt(accel[1]**2 + accel[2]**2))
    return np.array([roll, pitch])

def main():
    # --- EKF等の初期設定 (省略、内容は前回と同じ) ---
    state = np.zeros(4) 
    P = np.eye(4) * 0.1
    Q = np.diag([0.001, 0.001, 0.0001, 0.0001])
    R = np.diag([0.03, 0.03])
    H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    yaw_angle_rad = 0.0
    position = np.array([0.0, 0.0])
    velocity = np.array([0.0, 0.0])
    g = 9.81
    timestamps, estimated_states, raw_angles, estimated_yaw, path_data = [], [], [], [], []

    # --- シミュレーションループ (省略、内容は前回と同じ) ---
    dt = 0.01
    total_time = 10.0
    for t in np.arange(0, total_time, dt):
        accel, gyro = generate_simulated_imu_data(t, total_time)
        timestamps.append(t)
        phi, theta = state[0], state[1]
        bias_x, bias_y = state[2], state[3]
        p, q = gyro[:2] - np.array([bias_x, bias_y])
        F = np.eye(4)
        F[0,1] = p*np.cos(phi)*np.tan(theta) - q*np.sin(phi)*np.tan(theta)
        F[0,2] = -dt
        F[0,3] = -dt*np.sin(phi)*np.tan(theta)
        F[1,3] = -dt*np.cos(phi)
        phi_dot = p + q * np.sin(phi) * np.tan(theta)
        theta_dot = q * np.cos(phi)
        state[0] += dt * phi_dot
        state[1] += dt * theta_dot
        P = F @ P @ F.T + Q
        z = euler_from_accel(accel)
        raw_angles.append(z)
        y = z - H @ state[:4]
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        state = state + K @ y
        P = (np.eye(4) - K @ H) @ P
        estimated_states.append(state.copy())
        gz = gyro[2] 
        yaw_angle_rad += gz * dt
        estimated_yaw.append(yaw_angle_rad)
        phi, theta, psi = state[0], state[1], yaw_angle_rad
        c_phi,s_phi=np.cos(phi),np.sin(phi)
        c_th,s_th=np.cos(theta),np.sin(theta)
        c_psi,s_psi=np.cos(psi),np.sin(psi)
        R_x=np.array([[1,0,0],[0,c_phi,-s_phi],[0,s_phi,c_phi]])
        R_y=np.array([[c_th,0,s_th],[0,1,0],[-s_th,0,c_th]])
        R_z=np.array([[c_psi,-s_psi,0],[s_psi,c_psi,0],[0,0,1]])
        R_mat = R_z @ R_y @ R_x
        accel_world = R_mat @ accel
        motion_accel_world = accel_world - np.array([0, 0, g])
        velocity += motion_accel_world[:2] * dt
        position += velocity * dt
        path_data.append(position.copy())

    print("シミュレーション終了。グラフを生成しています...")

    # --- グラフの描画 ---
    estimated_states = np.array(estimated_states)
    raw_angles = np.array(raw_angles)
    path_data = np.array(path_data)
    estimated_deg = np.rad2deg(estimated_states[:, :2])
    raw_deg = np.rad2deg(raw_angles)
    estimated_yaw_deg = np.rad2deg(np.array(estimated_yaw))

    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('IMU Attitude & Path Estimation (Simulation)', fontsize=16)

    # (Roll, Pitch, Yawのプロットは前回と同じ)
    axs[0, 0].plot(timestamps, raw_deg[:, 0], 'r-', label='Raw Roll', alpha=0.5); axs[0, 0].plot(timestamps, estimated_deg[:, 0], 'b-', label='EKF Roll'); axs[0, 0].set_title('Roll'); axs[0, 0].legend(); axs[0, 0].grid(True)
    axs[0, 1].plot(timestamps, raw_deg[:, 1], 'r-', label='Raw Pitch', alpha=0.5); axs[0, 1].plot(timestamps, estimated_deg[:, 1], 'b-', label='EKF Pitch'); axs[0, 1].set_title('Pitch'); axs[0, 1].legend(); axs[0, 1].grid(True)
    axs[1, 0].plot(timestamps, estimated_yaw_deg, 'g-', label='Yaw'); axs[1, 0].set_title('Yaw'); axs[1, 0].legend(); axs[1, 0].grid(True); axs[1, 0].set_xlabel('Time [s]')

    # --- 【★ここからが変更点】 ---
    # 2D経路マップ
    axs[1, 1].plot(path_data[:, 0], path_data[:, 1], 'm-', label='Estimated Path')
    axs[1, 1].plot(path_data[0,0], path_data[0,1], 'go', markersize=10, label='Start')
    axs[1, 1].plot(path_data[-1,0], path_data[-1,1], 'ro', markersize=10, label='End')
    axs[1, 1].set_title('Estimated 2D Path (10cm Grid)')
    axs[1, 1].set_xlabel('X Position [m]')
    axs[1, 1].set_ylabel('Y Position [m]')
    axs[1, 1].legend()
    axs[1, 1].grid(True)
    
    # 推定経路の中心点を計算
    if len(path_data) > 0:
        center_x = (path_data[:, 0].min() + path_data[:, 0].max()) / 2
        center_y = (path_data[:, 1].min() + path_data[:, 1].max()) / 2
    else:
        center_x, center_y = 0, 0

    # 中止点から+/- 5cm (0.05m) の範囲で表示範囲を設定
    axs[1, 1].set_xlim(center_x - 0.05, center_x + 0.05)
    axs[1, 1].set_ylim(center_y - 0.05, center_y + 0.05)
    
    # アスペクト比を1に設定して正方形にする
    axs[1, 1].set_aspect('equal', adjustable='box')
    # --- 【★変更点ここまで】 ---

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('fixed_10cm_path_estimation.png')
    print("グラフを 'fixed_10cm_path_estimation.png' として保存しました。")

if __name__ == "__main__":
    main()