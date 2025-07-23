import numpy as np
from scipy.spatial.transform import Rotation as R_quat # クォータニオンを扱うためのSciPyのライブラリ

# --- 1. カルマンフィルタのパラメータ設定 ---

# 時間ステップ (秒)
dt = 0.01 # 例: 100HzのIMUデータ

# 状態の次元数 (クォータニオン4 + ジャイロバイアス3 = 7)
num_states = 7

# 観測の次元数 (加速度3 + ジャイロ3 = 6)
num_observations = 6

# 初期状態推定値 (x_hat)
# q = [qw, qx, qy, qz] (単位クォータニオン、初期は回転なし)
# bias_gyro = [bx, by, bz] (ジャイロバイアスの初期値、通常は0)
initial_state = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# 初期誤差共分散行列 (P)
# 姿勢推定は初期が不確か、バイアスも不確かとして大きめに設定
initial_covariance = np.diag([0.1, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01]) # 対角成分のみ

# プロセスノイズ共分散行列 (Q)
# 状態予測の不確かさ。姿勢やバイアスのドリフト、モデルの不完全性を表す
# クォータニオン部分のノイズ（姿勢のランダムウォーク）
q_noise_val = 0.01
# ジャイロバイアス部分のノイズ（バイアスのランダムウォーク）
bias_noise_val = 0.001
process_noise_covariance = np.diag([q_noise_val]*4 + [bias_noise_val]*3)

# 観測ノイズ共分散行列 (R)
# センサーのノイズの大きさ。加速度計とジャイロ計の計測ノイズ
# 加速度センサーのノイズ（例: 0.1 m/s^2）
accel_noise_val = 0.1
# ジャイロセンサーのノイズ（例: 0.01 rad/s）
gyro_noise_val = 0.01
observation_noise_covariance = np.diag([accel_noise_val]*3 + [gyro_noise_val]*3)

# 重力ベクトル (基準として使用)
# 地球座標系での重力方向（下向き）
gravity_vector_ref = np.array([0.0, 0.0, 9.81]) # Z軸が上向きなら -9.81

# --- 2. 状態遷移関数と観測関数（非線形） ---

# クォータニオンの乗算関数
def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

# 状態遷移関数 f(x, dt, gyro_obs)
# 現在の推定状態 x_hat とジャイロ観測値を使って次の状態を予測
def state_transition_function(x_hat, dt, gyro_obs_raw):
    q_prev = x_hat[0:4] # 前の姿勢クォータニオン
    bias_gyro = x_hat[4:7] # 前のジャイロバイアス

    # ジャイロ観測値からバイアスを差し引く
    # 正しい角速度 = 観測された角速度 - バイアス
    omega = gyro_obs_raw - bias_gyro

    # 角速度からクォータニオンの変化量を計算
    # 角速度ベクトル [wx, wy, wz] からクォータニオン変化量 dq = [0, wx, wy, wz] * 0.5 * dt
    # q_dot = 0.5 * q_prev * [0, omega_x, omega_y, omega_z]
    # クォータニオンの運動方程式 (dq/dt = 0.5 * q * omega_quaternion)
    omega_quat = np.array([0.0, omega[0], omega[1], omega[2]])
    dq = 0.5 * quat_mult(q_prev, omega_quat) * dt
    
    q_pred = q_prev + dq
    q_pred = q_pred / np.linalg.norm(q_pred) # クォータニオンを正規化

    # ジャイロバイアスはそのまま（ゆっくり変化すると仮定）
    bias_gyro_pred = bias_gyro

    return np.concatenate((q_pred, bias_gyro_pred))

# 観測関数 h(x_pred)
# 予測された状態からセンサーが何を観測するはずかを計算
def observation_function(x_pred):
    q_pred = x_pred[0:4] # 予測された姿勢クォータニオン

    # クォータニオンから回転行列を計算
    rot_matrix = R_quat.from_quat(q_pred[1:4].tolist() + [q_pred[0]]).as_matrix() # scipyのクォータニオンは [x,y,z,w] 順

    # 重力ベクトルを回転させて、IMU座標系での重力成分を予測
    # 実際にはIMUは加速度センサーで重力と線形加速度を計測するが、ここでは重力のみを観測できると仮定（静止状態を想定）
    # センサ座標系での重力ベクトル = 回転行列の逆転置（または単に転置） * 地球座標系での重力ベクトル
    predicted_gravity_in_imu_frame = rot_matrix.T @ gravity_vector_ref
    
    # 予測されるジャイロ観測値は、バイアスが0（ここでは状態推定でバイアスを推定するので、純粋な角速度は0とする）
    # 注意: ここではIMUが静止しており、線形加速度がないと仮定している。
    #      もし線形加速度も状態に含める場合、h_x_predの加速度部分にそれを追加する必要がある。
    predicted_gyro_obs = np.array([0.0, 0.0, 0.0]) # 静止状態なので角速度は0と予測

    # 予測される加速度とジャイロ観測値を結合
    return np.concatenate((predicted_gravity_in_imu_frame, predicted_gyro_obs))


# --- 3. ヤコビアン行列の計算（手動または自動微分ライブラリ） ---
# EKFで最も複雑な部分。ここでは簡略化のため、概略のみ示す。
# 実際のシステムでは、これらは状態x_hatの周りで評価される。

# F_k: 状態遷移行列 f のヤコビアン ( df/dx )
def compute_F_k(x_hat, dt, gyro_obs_raw):
    # これは非常に複雑な行列になるため、実装は省略。
    # クォータニオンの微分、バイアスの微分などが含まれる。
    # 状態クォータニオンとバイアスに対する偏微分
    # 一般的には、解析的に導出するか、自動微分ライブラリを使用する。
    
    # 簡易的な例として、単位行列に近いものを返す（実際にはもっと複雑）
    F = np.eye(num_states) 
    # クォータニオン部分のF行列は、角速度に依存する。
    # ジャイロバイアスに対するクォータニオンの変化の偏微分も考慮される。
    # バイアスがそのまま予測されると仮定すると、バイアス部分の対角は1。
    # F[0:4, 4:7] = ... (q_dot のバイアスに対する偏微分)
    
    # より正確には、R_quat.from_rotvec(omega * dt).as_quat() の微分や、クォータニオン微分のヤコビアンが必要
    # 例えば、q_dot = 0.5 * [ -qx, -qy, -qz ] @ omega_x, ...
    # F = np.eye(num_states)
    # q_prev = x_hat[0:4]
    # omega = gyro_obs_raw - x_hat[4:7]
    # Q_omega = np.array([
    #     [-x_hat[1], -x_hat[2], -x_hat[3]],
    #     [ x_hat[0], -x_hat[3],  x_hat[2]],
    #     [ x_hat[3],  x_hat[0], -x_hat[1]],
    #     [-x_hat[2],  x_hat[1],  x_hat[0]]
    # ]) # partial q_dot / partial omega_vec
    # F_q_q = (np.eye(4) + 0.5 * np.array([
    #     [0, -omega[0], -omega[1], -omega[2]],
    #     [omega[0], 0, omega[2], -omega[1]],
    #     [omega[1], -omega[2], 0, omega[0]],
    #     [omega[2], omega[1], -omega[0], 0]
    # ]) * dt)
    # F[0:4, 0:4] = F_q_q
    # F[0:4, 4:7] = -0.5 * Q_omega * dt
    return F

# H_k: 観測関数 h のヤコビアン ( dh/dx )
def compute_H_k(x_pred):
    # これも非常に複雑な行列になるため、実装は省略。
    # 予測姿勢クォータニオンに対する加速度（重力）の偏微分など
    # 状態に対する加速度計とジャイロ計の予測観測値の偏微分
    
    # 簡易的な例として、単位行列に近いものを返す
    H = np.zeros((num_observations, num_states))
    
    # 加速度計部分 (H_accel)
    # 予測された姿勢 q_pred が変化したときに、予測される重力ベクトルがどう変化するか
    # これは q_pred に対する偏微分になる
    # H_accel_q = ... (df_accel/dq)
    # H[0:3, 0:4] = H_accel_q # q_predに対する加速度の偏微分
    
    # ジャイロ計部分 (H_gyro)
    # 予測されるジャイロ観測値は、状態のジャイロバイアスに依存する（予測角速度が0でバイアスがそのまま観測される場合）
    # H_gyro_bias = np.eye(3) * -1 # dh_gyro/dbias
    # H[3:6, 4:7] = -np.eye(3) # バイアス成分に対する偏微分
    
    return H

# --- 4. 拡張カルマンフィルタの実行関数 ---

def extended_kalman_filter_6axis(
    observations_accel, observations_gyro, initial_state, initial_covariance,
    process_noise_covariance, observation_noise_covariance, dt, gravity_vector_ref
):
    x_hat = initial_state
    P = initial_covariance

    estimated_states = []
    estimated_covariances = []

    num_data_points = len(observations_accel)

    for i in range(num_data_points):
        # 観測データ
        current_accel_obs = observations_accel[i]
        current_gyro_obs = observations_gyro[i]

        # --- 予測ステップ ---
        # 1. 状態の予測: x_k^- = f(x_{k-1}, u_{k-1})
        x_pred = state_transition_function(x_hat, dt, current_gyro_obs)
        # クォータニオンの正規化を保証
        x_pred[0:4] = x_pred[0:4] / np.linalg.norm(x_pred[0:4])
        
        # 2. ヤコビアン行列 F_k の計算
        # 実際にはここで x_hat と gyro_obs_raw から F_k を計算する
        # compute_F_k の具体的な実装は省略されているが、概念としてはここに挟む
        F_k = compute_F_k(x_hat, dt, current_gyro_obs) # 簡略化のため、実際には計算されるべき
        
        # 3. 共分散の予測: P_k^- = F_k P_{k-1} F_k^T + Q
        P_pred = F_k @ P @ F_k.T + process_noise_covariance

        # --- 更新ステップ ---
        # 1. 観測残差の計算: y_k = z_k - h(x_k^-)
        z_k = np.concatenate((current_accel_obs, current_gyro_obs)) # 実際の観測 (加速度とジャイロ)
        h_x_pred = observation_function(x_pred) # 予測された状態から計算される観測
        innovation = z_k - h_x_pred

        # 2. ヤコビアン行列 H_k の計算
        # 実際にはここで x_pred から H_k を計算する
        # compute_H_k の具体的な実装は省略されているが、概念としてはここに挟む
        H_k = compute_H_k(x_pred) # 簡略化のため、実際には計算されるべき

        # 3. 観測残差共分散 S_k の計算: S_k = H_k P_k^- H_k^T + R
        S_k = H_k @ P_pred @ H_k.T + observation_noise_covariance

        # 4. カルマンゲイン K_k の計算: K_k = P_k^- H_k^T S_k^-1
        K_k = P_pred @ H_k.T @ np.linalg.inv(S_k)

        # 5. 状態の更新: x_k = x_k^- + K_k y_k
        x_hat = x_pred + K_k @ innovation
        # 更新後もクォータニオンを正規化
        x_hat[0:4] = x_hat[0:4] / np.linalg.norm(x_hat[0:4])

        # 6. 共分散の更新: P_k = (I - K_k H_k) P_k^-
        P = (np.eye(num_states) - K_k @ H_k) @ P_pred

        # 結果を保存
        estimated_states.append(x_hat.copy())
        estimated_covariances.append(P.copy())

    return np.array(estimated_states), np.array(estimated_covariances)


# --- 5. シミュレーションデータの生成（簡単な例） ---
if __name__ == "__main__":
    # シミュレーション時間
    total_time = 10 # 秒
    num_steps = int(total_time / dt)

    # 真の姿勢とジャイロバイアス (シミュレーション用)
    true_q = [R_quat.from_euler('xyz', [0.0, 0.0, np.radians(i * 10)]).as_quat()[[3,0,1,2]] # [w,x,y,z]順
              for i in range(num_steps)] # Z軸周りにゆっくり回転

    true_bias_gyro = np.array([0.01, -0.005, 0.02]) # 固定バイアス (rad/s)
    
    # 観測データの生成 (ノイズを追加)
    # 角速度 = 真の角速度 + ジャイロバイアス + ノイズ
    # 加速度 = 真の加速度（重力） + ノイズ
    observations_accel_sim = []
    observations_gyro_sim = []

    # 地球座標系での重力ベクトル
    g_earth = np.array([0.0, 0.0, -9.81]) # 地球座標系でZ軸が上向きなら -9.81

    for i in range(num_steps):
        # 真の姿勢クォータニオン
        current_true_q_scipy = R_quat.from_quat(true_q[i][1:4].tolist() + [true_q[i][0]]) # scipyは [x,y,z,w] 順

        # 加速度センサー観測 (IMU座標系での重力成分 + ノイズ)
        # 静止状態を仮定すると、加速度計は重力ベクトルを観測する
        # IMU座標系での重力 = (逆回転) * 地球座標系での重力
        accel_in_imu_frame = current_true_q_scipy.inv().apply(-g_earth) # 重力は上向きに対して下向きなのでマイナスg_earth
        obs_accel = accel_in_imu_frame + np.random.normal(0, accel_noise_val, 3) # ノイズを加える
        observations_accel_sim.append(obs_accel)

        # ジャイロセンサー観測 (真の角速度 + バイアス + ノイズ)
        # 真の角速度 (Z軸周りに一定で回転)
        true_angular_velocity = np.array([0.0, 0.0, np.radians(10)]) # rad/s
        
        # 観測ジャイロ = 真の角速度 + バイアス + ノイズ
        obs_gyro = true_angular_velocity + true_bias_gyro + np.random.normal(0, gyro_noise_val, 3) # ノイズを加える
        observations_gyro_sim.append(obs_gyro)
    
    observations_accel_sim = np.array(observations_accel_sim)
    observations_gyro_sim = np.array(observations_gyro_sim)

    # カルマンフィルタを実行
    estimated_states, estimated_covariances = extended_kalman_filter_6axis(
        observations_accel_sim, observations_gyro_sim,
        initial_state, initial_covariance,
        process_noise_covariance, observation_noise_covariance,
        dt, gravity_vector_ref
    )

    print("--- 6軸IMU用拡張カルマンフィルタの結果（一部） ---")
    print(f"最終推定状態: {estimated_states[-1]}")
    print(f"最終共分散行列の対角成分: {np.diag(estimated_covariances[-1])}")

    # --- 結果の可視化 ---
    try:
        import matplotlib.pyplot as plt

        # オイラー角に変換してプロット（可視化しやすいように）
        estimated_euler = []
        true_euler = []
        for i in range(num_steps):
            # estimated_statesのクォータニオンは [w,x,y,z] 順
            # scipy.spatial.transform.Rotation.from_quat は [x,y,z,w] 順なので変換が必要
            estimated_q_scipy_format = estimated_states[i][1:4].tolist() + [estimated_states[i][0]]
            estimated_euler.append(R_quat.from_quat(estimated_q_scipy_format).as_euler('xyz', degrees=True))
            
            true_q_scipy_format = true_q[i][1:4].tolist() + [true_q[i][0]]
            true_euler.append(R_quat.from_quat(true_q_scipy_format).as_euler('xyz', degrees=True))
        
        estimated_euler = np.array(estimated_euler)
        true_euler = np.array(true_euler)

        plt.figure(figsize=(15, 8))

        plt.subplot(2, 1, 1)
        plt.plot(true_euler[:, 0], label='True Roll', linestyle='--', color='green')
        plt.plot(true_euler[:, 1], label='True Pitch', linestyle='--', color='blue')
        plt.plot(true_euler[:, 2], label='True Yaw', linestyle='--', color='red')
        plt.plot(estimated_euler[:, 0], label='Estimated Roll', color='darkgreen')
        plt.plot(estimated_euler[:, 1], label='Estimated Pitch', color='darkblue')
        plt.plot(estimated_euler[:, 2], label='Estimated Yaw', color='darkred')
        plt.title('Attitude Estimation (Euler Angles)')
        plt.xlabel('Time Step')
        plt.ylabel('Angle (degrees)')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(np.array(true_bias_gyro[0]), label='True Gyro Bias X', linestyle='--', color='cyan')
        plt.plot(np.array(true_bias_gyro[1]), label='True Gyro Bias Y', linestyle='--', color='magenta')
        plt.plot(np.array(true_bias_gyro[2]), label='True Gyro Bias Z', linestyle='--', color='orange')
        plt.plot(estimated_states[:, 4], label='Estimated Gyro Bias X', color='darkcyan')
        plt.plot(estimated_states[:, 5], label='Estimated Gyro Bias Y', color='darkmagenta')
        plt.plot(estimated_states[:, 6], label='Estimated Gyro Bias Z', color='darkorange')
        plt.title('Gyro Bias Estimation')
        plt.xlabel('Time Step')
        plt.ylabel('Bias (rad/s)')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    except ImportError:
        print("MatplotlibまたはSciPyがインストールされていないため、グラフの表示はスキップされます。")
    except Exception as e:
        print(f"グラフ描画中にエラーが発生しました: {e}")