import serial
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
from collections import deque

# --- グローバル変数として扱うデータ ---
# (アニメーション関数内で値を更新し続けるため)
timestamps = []
estimated_states = []
r_history = []
start_time = 0.0
last_time = 0.0

# EKF関連の変数
state = np.zeros(4)
P = np.eye(4) * 0.1
Q = np.diag([0.001, 0.001, 0.0001, 0.0001])
R = np.diag([0.03, 0.03])
H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])

# 適応化のための変数
WINDOW_SIZE = 50
ADAPTATION_FACTOR = 0.05
innovation_history = deque(maxlen=WINDOW_SIZE)


# --- センサーデータ取得関数 ---
def get_imu_data(ser):
    """シリアルポートから1行読み込み、IMUデータを返す"""
    try:
        line = ser.readline().decode('utf-8').strip()
        if line:
            ax, ay, az, gx, gy, gz = map(float, line.split(","))
            return np.array([ax, ay, az]), np.array([gx, gy, gz])
    except Exception:
        # リアルタイムプロット中はエラー表示を抑制
        pass
    return None, None

# --- 加速度から角度を計算 ---
def euler_from_accel(accel):
    """加速度計の値からロール、ピッチ角を計算"""
    roll = np.arctan2(accel[1], accel[2])
    pitch = np.arctan2(-accel[0], np.sqrt(accel[1]**2 + accel[2]**2))
    return np.array([roll, pitch])


# --- アニメーションのフレーム毎に呼び出される更新関数 ---
def update(frame, ser, lines):
    global state, P, R, last_time

    # --- 1. データの取得と時間の計算 ---
    accel, gyro = get_imu_data(ser)
    if accel is None or gyro is None:
        return lines.values() # データがなければ何もしない

    current_time = time.time()
    dt = current_time - last_time
    if dt <= 0.001: # 更新間隔が短すぎる場合はスキップ
        return lines.values()
    last_time = current_time
    
    # --- 2. EKF計算 ---
    phi, theta = state[0], state[1]
    bias_x, bias_y = state[2], state[3]
    p, q = gyro[:2] - np.array([bias_x, bias_y])

    # 予測
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

    # 更新
    z = euler_from_accel(accel)
    y = z - H @ state[:4]
    
    # Rの適応更新
    innovation_history.append(y)
    if len(innovation_history) == WINDOW_SIZE:
        innov_array = np.array(innovation_history)
        empirical_cov = np.mean([np.outer(inn, inn) for inn in innov_array], axis=0)
        R = (1 - ADAPTATION_FACTOR) * R + ADAPTATION_FACTOR * empirical_cov
    
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    state = state + K @ y
    P = (np.eye(4) - K @ H) @ P
    
    # --- 3. データをリストに追加 ---
    timestamps.append(current_time - start_time)
    estimated_states.append(state.copy())
    r_history.append(np.diag(R).copy())

    # グラフ表示範囲を最新の500サンプルに制限
    display_range = 500
    ts = timestamps[-display_range:]
    est_deg = np.rad2deg(np.array(estimated_states)[-display_range:, :2])
    r_hist = np.array(r_history)[-display_range:]

    # --- 4. グラフの描画データを更新 ---
    lines['roll'].set_data(ts, est_deg[:, 0])
    lines['pitch'].set_data(ts, est_deg[:, 1])
    lines['r_roll'].set_data(ts, r_hist[:, 0])
    lines['r_pitch'].set_data(ts, r_hist[:, 1])
    
    # 軸の範囲を自動調整
    for key in lines:
        ax = lines[key].axes
        ax.relim()
        ax.autoscale_view()

    return lines.values()


def main():
    global start_time, last_time

    # --- シリアルポートの設定 ---
    try:
        ser = serial.Serial('/dev/ttyS3', 115200, timeout=1)
        print(f"{ser.name} に接続しました。グラフウィンドウを閉じるとプログラムが終了します。")
    except serial.SerialException as e:
        print(f"エラー: シリアルポートに接続できません。 {e}")
        return

    # --- 時間の初期化 ---
    start_time = time.time()
    last_time = start_time
    
    # --- グラフの初期設定 ---
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle('Real-time Adaptive EKF', fontsize=16)

    # 上のグラフ（角度）
    roll_line, = axs[0].plot([], [], 'b-', label='EKF Roll')
    pitch_line, = axs[0].plot([], [], 'g-', label='EKF Pitch')
    axs[0].set_ylabel('Angle [deg]')
    axs[0].legend(loc='upper left')
    axs[0].grid(True)
    
    # 下のグラフ（Rの値）
    r_roll_line, = axs[1].plot([], [], 'r-', label='Adaptive R for Roll')
    r_pitch_line, = axs[1].plot([], [], 'm-', label='Adaptive R for Pitch')
    axs[1].set_ylabel('Measurement Noise Covariance (R)')
    axs[1].set_xlabel('Time [s]')
    axs[1].set_yscale('log')
    axs[1].legend(loc='upper left')
    axs[1].grid(True, which="both")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 描画する線を辞書にまとめる
    lines = {
        'roll': roll_line, 'pitch': pitch_line, 
        'r_roll': r_roll_line, 'r_pitch': r_pitch_line
    }

    # --- アニメーションの開始 ---
    ani = animation.FuncAnimation(
        fig, 
        update, 
        fargs=(ser, lines), 
        interval=50, 
        blit=False,
        cache_frame_data=False) # 警告抑制のための引数

    plt.show()

    # ウィンドウが閉じられたらシリアルポートも閉じる
    ser.close()
    print("シリアルポートを閉じました。")


if __name__ == "__main__":
    main()