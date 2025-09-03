import serial
import numpy as np
import time
import math
from collections import deque

# --- ユーザー設定項目 ---
# 使用するシリアルポートに合わせて変更してください (例: 'COM3' on Windows, '/dev/ttyUSB0' on Linux)
SERIAL_PORT = 'COM4' 
BAUD_RATE = 115200

# --- 定数 (Cコードから移植) ---
# 物理定数
GRAVITY_AMOUNT = 9.80665
EARTH_ROTATION_SPEED_AMOUNT = 7.2921159e-5

# センサー・フィルタ設定
MESUREMENT_FREQUENCY = 30 # センサーのサンプリング周波数 (Hz)
LIST_SIZE = 8              # ガウシアンフィルタのウィンドウサイズ
SIGMA_K = LIST_SIZE / 8.0  # ガウシアンフィルタのシグマ

# Madgwick Filterの重み (ゲイン)
ACC_MADGWICK_FILTER_WEIGHT = 0.11
GYRO_MADGWICK_FILTER_WEIGHT = 0.00000001

# ノイズ関連の定数 (Cコードの値を参照)
GYRO_NOISE_DENSITY = 1.0e-3 * np.pi / 180.0
ACCEL_NOISE_DENSITY = 14.0e-6 * GRAVITY_AMOUNT
GYRO_NOISE_AMOUNT = GYRO_NOISE_DENSITY * math.sqrt(MESUREMENT_FREQUENCY)
ACCEL_NOISE_AMOUNT = ACCEL_NOISE_DENSITY * math.sqrt(MESUREMENT_FREQUENCY)
ACCEL_BIAS_DRIFT = 4.43e-6 * GRAVITY_AMOUNT * 3.0

# --- センサーデータ取得関数 (ユーザー提供) ---
def get_imu_data(ser):
    """シリアルポートから1行読み込み、IMUデータを返す"""
    try:
        line = ser.readline().decode('utf-8').strip()
        if line:
            # カンマ区切りのデータをfloatに変換
            ax, ay, az, gx, gy, gz = map(float, line.split(","))
            # Cコードはrad/sを前提としているため、単位を合わせる
            return np.array([ax, ay, az]), np.array([gx, gy, gz])
    except Exception as e:
        print(f"データ読み取りエラー: {e}")
    return None, None

# --- 数学関数 (Cコードから移植) ---
def diff_quaternion(q, omega):
    """クォータニオンの微分を計算"""
    w, x, y, z = q
    omega_x, omega_y, omega_z = omega
    dqdt = np.zeros(4)
    dqdt[0] = 0.5 * (-x * omega_x - y * omega_y - z * omega_z)
    dqdt[1] = 0.5 * (w * omega_x + y * omega_z - z * omega_y)
    dqdt[2] = 0.5 * (w * omega_y - x * omega_z + z * omega_x)
    dqdt[3] = 0.5 * (w * omega_z + x * omega_y - y * omega_x)
    return dqdt

def runge_kutta_update(q, omega, h):
    """RK4法によるクォータニオン更新"""
    k1 = diff_quaternion(q, omega)
    k2 = diff_quaternion(q + h / 2.0 * k1, omega)
    k3 = diff_quaternion(q + h / 2.0 * k2, omega)
    k4 = diff_quaternion(q + h * k3, omega)
    q_next = q + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    # 正規化
    return q_next / np.linalg.norm(q_next)

def update_vector_rk4(v, omega, h):
    """RK4法による3次元ベクトルの回転更新"""
    k1 = np.cross(omega, v)
    k2 = np.cross(omega, v + h / 2.0 * k1)
    k3 = np.cross(omega, v + h / 2.0 * k2)
    k4 = np.cross(omega, v + h * k3)
    v_next = v + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return v_next

def apply_rotation(q, v):
    """クォータニオンを使ってベクトルを回転"""
    q_w, q_vec = q[0], q[1:]
    v_rotated = v + 2 * np.cross(q_vec, q_w * v + np.cross(q_vec, v))
    return v_rotated

# --- フィルタ&アルゴリズム ---
class IMUProcessor:
    def __init__(self):
        # 状態変数
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        self.velocity = np.zeros(3)
        self.position = np.zeros(3)
        self.current_gravity = np.zeros(3)
        self.old_acceleration = np.zeros(3)
        self.old_velocity = np.zeros(3)
        
        # ゼロ速度補正用
        self.biased_velocity = 0.0
        self.zero_velocity_counter = 0

        # キャリブレーション用
        self.calibrate_counter = 0

        # データ保存用バッファ (deque)
        self.accel_hist = [deque(maxlen=LIST_SIZE) for _ in range(3)]
        self.gyro_hist = [deque(maxlen=LIST_SIZE) for _ in range(3)]

        # ガウシアンフィルタのカーネル
        self.gaussian_kernel = self._create_gaussian_kernel()

        # 時間管理
        self.last_time = None

    def _create_gaussian_kernel(self):
        """ガウシアンフィルタのカーネルを事前計算"""
        kernel = np.zeros(LIST_SIZE)
        for i in range(LIST_SIZE):
            kernel[i] = (1.0 / (math.sqrt(2.0 * math.pi) * SIGMA_K)) * \
                        math.exp(-(i * i) / (2.0 * SIGMA_K * SIGMA_K))
        return kernel / np.sum(kernel) # 正規化

    def apply_causal_gaussian_filter(self, history):
        """因果的なガウシアンフィルタを適用"""
        # dequeは自動的に古いデータを破棄するので、常に最新のN個が保たれる
        if len(history) < LIST_SIZE:
            return history[-1] if history else 0.0
        
        # historyは時系列順（古い->新しい）なので、カーネルを逆順に適用
        filtered_value = np.dot(np.array(history), self.gaussian_kernel[::-1])
        return filtered_value

    def zero_velocity_correction(self, target_velocity, dt):
        """ゼロ速度補正 (ZUPT)"""
        self.biased_velocity += ACCEL_BIAS_DRIFT * dt
        
        if np.all(np.abs(target_velocity) < self.biased_velocity):
            self.zero_velocity_counter += 1
        else:
            self.zero_velocity_counter = 0
            
        if self.zero_velocity_counter > MESUREMENT_FREQUENCY / 10: # 0.1秒静止でリセット
            self.biased_velocity = 0.0
            self.zero_velocity_counter = 0
            return True
        return False

    def initialize(self, ser):
        """静止状態から初期姿勢をキャリブレーション"""
        print("キャリブレーション開始... 1秒間センサーを静止させてください。")
        
        accel_sum = np.zeros(3)
        gyro_sum = np.zeros(3)
        count = 0
        
        start_time = time.time()
        while time.time() - start_time < 1.0:
            accel, gyro = get_imu_data(ser)
            if accel is not None and gyro is not None:
                accel_sum += accel
                gyro_sum += gyro
                count += 1
        
        if count == 0:
            print("エラー: キャリブレーション中にデータを取得できませんでした。")
            return False

        # 平均値を計算
        est_accel = accel_sum / count
        est_gyro = gyro_sum / count

        # 初期重力ベクトルとして保存
        self.current_gravity = est_accel.copy()
        
        # データを履歴バッファに満たす
        for i in range(3):
            for _ in range(LIST_SIZE):
                self.accel_hist[i].append(est_accel[i])
                self.gyro_hist[i].append(est_gyro[i])
        
        # --- 初期クォータニオンの計算 ---
        # (CコードのロジックをNumPyで実装)
        accel_norm = np.linalg.norm(est_accel)
        z_axis = est_accel / accel_norm
        
        dot_product = np.dot(est_accel, est_gyro)
        earth_rotation = est_gyro - (dot_product / (accel_norm**2)) * est_accel
        earth_norm = np.linalg.norm(earth_rotation)
        
        if earth_norm < 1e-6: # ゼロ除算を回避
            print("警告: 地球の自転を検出できません。初期姿勢が不正確になる可能性があります。")
            # 仮のY軸を設定
            y_axis_temp = np.array([0, 1, 0] if abs(z_axis[1]) < 0.9 else [1, 0, 0])
            x_axis = np.cross(y_axis_temp, z_axis)
            x_axis /= np.linalg.norm(x_axis)
            y_axis = np.cross(z_axis, x_axis)
        else:
            y_axis = -earth_rotation / earth_norm # 北半球を仮定
            x_axis = np.cross(y_axis, z_axis)

        # 回転行列を作成
        R = np.array([x_axis, y_axis, z_axis]).T
        
        # 回転行列からクォータニオンへ変換
        tr = np.trace(R)
        if tr > 0:
            S = math.sqrt(tr + 1.0) * 2
            qw = 0.25 * S
            qx = (R[2, 1] - R[1, 2]) / S
            qy = (R[0, 2] - R[2, 0]) / S
            qz = (R[1, 0] - R[0, 1]) / S
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S
            
        self.quaternion = np.array([qw, qx, qy, qz])
        self.last_time = time.time()
        print("キャリブレーション完了。計測を開始します。")
        return True


    def update(self, accel, gyro):
        """データを受け取り、状態を更新する"""
        current_time = time.time()
        dt = current_time - self.last_time
        if dt <= 0: return # 時間経過がない場合はスキップ
        self.last_time = current_time

        # データを履歴に追加
        for i in range(3):
            self.accel_hist[i].append(accel[i])
            self.gyro_hist[i].append(gyro[i])

        # ガウスフィルタを適用
        est_accel = np.array([self.apply_causal_gaussian_filter(h) for h in self.accel_hist])
        est_gyro = np.array([self.apply_causal_gaussian_filter(h) for h in self.gyro_hist])

        # Madgwickフィルタによる補正
        accel_norm = np.linalg.norm(est_accel)
        
        # Cコードの条件を移植
        if (abs(accel_norm - GRAVITY_AMOUNT) < ACCEL_NOISE_AMOUNT * 40):
            self.calibrate_counter += 1
        else:
            self.calibrate_counter = 0

        if self.calibrate_counter > MESUREMENT_FREQUENCY / 30: # 約33ms静止で補正
            # Madgwickの補正ステップ
            g_ref = np.array([0, 0, 1.0]) # 正規化された重力ベクトル
            
            # 加速度による補正
            # 現在の姿勢で重力がどう見えるか
            g_est = apply_rotation(np.array([self.quaternion[0], -self.quaternion[1], -self.quaternion[2], -self.quaternion[3]]), g_ref)
            
            # 観測された加速度（正規化）と推定された重力の外積で誤差を計算
            error = np.cross(est_accel / accel_norm, g_est)
            # 誤差を使ってジャイロを補正
            est_gyro -= error * ACC_MADGWICK_FILTER_WEIGHT
            
            # 重力ベクトルとクォータニオンもリセットして安定化
            self.current_gravity = est_accel.copy()
            self.velocity = np.zeros(3)
            self.calibrate_counter = 0

        # 姿勢の予測 (RK4)
        self.quaternion = runge_kutta_update(self.quaternion, est_gyro, dt)
        
        # 重力ベクトルの回転 (RK4)
        self.current_gravity = update_vector_rk4(self.current_gravity, -est_gyro, dt)

        # 重力成分を除去
        linear_accel_sensor_frame = est_accel - self.current_gravity
        
        # ワールド座標系に変換
        linear_accel_world_frame = apply_rotation(self.quaternion, linear_accel_sensor_frame)
        
        # 速度と位置を積分 (台形則)
        self.velocity += (linear_accel_world_frame + self.old_acceleration) / 2.0 * dt
        self.position += (self.velocity + self.old_velocity) / 2.0 * dt
        
        # ゼロ速度補正
        if self.zero_velocity_correction(self.velocity, dt):
            self.velocity = np.zeros(3)

        # 古い値を更新
        self.old_acceleration = linear_accel_world_frame
        self.old_velocity = self.velocity.copy()

# --- メイン実行部 ---
def main():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"{ser.name}に接続しました。")
    except serial.SerialException as e:
        print(f"エラー: シリアルポートに接続できません。 {e}")
        return

    processor = IMUProcessor()
    
    if not processor.initialize(ser):
        ser.close()
        return

    try:
        while True:
            accel, gyro = get_imu_data(ser)
            if accel is not None and gyro is not None:
                processor.update(accel, gyro)
                
                # 30Hzでデータを表示 (Cコードのロジックに合わせる)
                # C言語の `execute_counter` に相当する処理
                if int(time.time() * 30) % 2 == 0:
                    pos = processor.position
                    vel = processor.velocity
                    q = processor.quaternion
                    print(f"Pos: [x:{pos[0]:.2f} y:{pos[1]:.2f} z:{pos[2]:.2f}], "
                          f"Vel: [x:{vel[0]:.2f} y:{vel[1]:.2f} z:{vel[2]:.2f}], "
                          f"Q: [w:{q[0]:.2f} x:{q[1]:.2f} y:{q[2]:.2f} z:{q[3]:.2f}]", end='\r')

    except KeyboardInterrupt:
        print("\nプログラムを終了します。")
    finally:
        ser.close()
        print("シリアルポートを閉じました。")

if __name__ == "__main__":
    main()