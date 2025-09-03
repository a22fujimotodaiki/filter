import serial
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial.transform import Rotation as R
import time

# --- グローバル変数 ---
# 状態を保持するための変数をまとめて定義
class ImuState:
    def __init__(self):
        self.ser = None
        self.accel = np.zeros(3)
        self.gyro = np.zeros(3)
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.position = np.zeros(2)
        self.velocity = np.zeros(2)
        self.path_history = []
        self.last_time = None
        self.gyro_offset = np.zeros(3)
        self.alpha = 0.98

# グローバルなstateインスタンスを作成
state = ImuState()

# --- グラフの初期設定 ---
fig, ax = plt.subplots(figsize=(10, 10))
line, = ax.plot([], [], 'm-', lw=2)
start_point, = ax.plot([], [], 'go', markersize=10, label='Start')
end_point, = ax.plot([], [], 'rs', markersize=10, label='End')
ax.set_title('Real-time Estimated 2D Path')
ax.set_xlabel('X Position [m]')
ax.set_ylabel('Y Position [m]')
ax.grid(True)
ax.set_aspect('equal', adjustable='box')
ax.legend()


def setup():
    """シリアルポートの接続とキャリブレーションを行う"""
    SERIAL_PORT = 'COM4' # ご自身の環境に合わせて変更
    BAUD_RATE = 115200

    try:
        state.ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    except serial.SerialException as e:
        print(f"エラー: シリアルポートに接続できません。 {e}")
        return False
    
    print("キャリブレーション開始... 3秒間センサーを静止させてください。")
    accel_samples, gyro_samples = [], []
    start_time = time.time()
    while time.time() - start_time < 3.0:
        line = state.ser.readline().decode('utf-8').strip()
        if line:
            try:
                ax, ay, az, gx, gy, gz = map(float, line.split(','))
                accel_samples.append([ax, ay, az])
                gyro_samples.append([gx, gy, gz])
            except ValueError:
                continue # データ形式が不正な行はスキップ
    
    if not accel_samples:
        print("エラー: キャリブレーションデータを取得できませんでした。")
        return False

    state.gyro_offset = np.mean(gyro_samples, axis=0)
    accel_avg = np.mean(accel_samples, axis=0)
    state.roll = np.arctan2(accel_avg[1], accel_avg[2])
    state.pitch = np.arctan2(-accel_avg[0], np.sqrt(accel_avg[1]**2 + accel_avg[2]**2))
    
    print("キャリブレーション完了。計測を開始します。")
    state.path_history.append(state.position.copy()) # 初期位置を追加
    state.last_time = time.time()
    return True

def update(frame):
    """FuncAnimationによって定期的に呼ばれ、データ処理と描画更新を行う"""
    # 1. データの読み込み
    serial_line = state.ser.readline().decode('utf-8').strip() # ✅ 変数名を変更
    if not serial_line:
        # 戻り値はグラフオブジェクトなので、元の 'line' のまま
        return line, start_point, end_point
    
    try:
        # ✅ 変数名を変更
        ax, ay, az, gx, gy, gz = map(float, serial_line.split(','))
        state.accel = np.array([ax, ay, az])
        state.gyro = np.array([gx, gy, gz])
    except ValueError:
        # 戻り値はグラフオブジェクトなので、元の 'line' のまま
        return line, start_point, end_point

    # 2. 時間計算
    current_time = time.time()
    dt = current_time - state.last_time
    state.last_time = current_time
    if dt <= 0:
        return line, start_point, end_point
        
    # 3. 姿勢計算 (相補フィルター)
    state.gyro -= state.gyro_offset
    accel_roll = np.arctan2(state.accel[1], state.accel[2])
    accel_pitch = np.arctan2(-state.accel[0], np.sqrt(state.accel[1]**2 + state.accel[2]**2))
    
    state.roll = state.alpha * (state.roll + state.gyro[0] * dt) + (1 - state.alpha) * accel_roll
    state.pitch = state.alpha * (state.pitch + state.gyro[1] * dt) + (1 - state.alpha) * accel_pitch
    state.yaw += state.gyro[2] * dt
    
    # 4. 重力除去と経路計算
    r = R.from_euler('xyz', [state.roll, state.pitch, state.yaw])
    accel_world = r.apply(state.accel)
    gravity_world = np.array([0, 0, -1.0]) # 正規化された重力と仮定
    linear_accel_world = accel_world - gravity_world
    
    state.velocity += linear_accel_world[:2] * dt
    state.position += state.velocity * dt
    state.path_history.append(state.position.copy())
    
    # 5. 描画データの更新
    path = np.array(state.path_history)
    line.set_data(path[:, 0], path[:, 1]) # ここでは正しくグラフオブジェクトの`line`が使われる
    
    if len(path) > 0:
        start_point.set_data(path[0, 0], path[0, 1])
        end_point.set_data(path[-1, 0], path[-1, 1])

    # 軸の範囲を自動更新
    ax.relim()
    ax.autoscale_view()
    
    return line, start_point, end_point

# --- メイン処理 ---
if __name__ == "__main__":
    if setup():
        # setupが成功したらアニメーションを開始
        ani = animation.FuncAnimation(fig, update, interval=10, blit=False)
        plt.show()
        # アニメーションウィンドウを閉じたらシリアルポートも閉じる
        if state.ser and state.ser.is_open:
            state.ser.close()
            print("シリアルポートを閉じました。")