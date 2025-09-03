import serial
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
from scipy.spatial.transform import Rotation as R
import collections # dequeを使うために追加

# --- グローバル変数としてデータを保持するリスト ---
# FuncAnimationからアクセスできるようにするため
path_history = collections.deque(maxlen=500) # 最新の500点だけ保持
timestamps = collections.deque(maxlen=500)
current_roll_pitch_yaw = [0.0, 0.0, 0.0]

# --- リアルタイムプロット用の設定 ---
fig, (ax_path, ax_angles) = plt.subplots(1, 2, figsize=(16, 8))
# 経路プロット
line_path, = ax_path.plot([], [], 'm-', label='Movement Path', linewidth=2)
ax_path.plot([], [], 'go', markersize=8, label='Start') # 開始地点は固定
ax_path.plot([], [], 'rs', markersize=8, label='End')   # 終了地点は動的に更新
ax_path.set_title('Real-time Estimated 2D Path')
ax_path.set_xlabel('X Position [m]')
ax_path.set_ylabel('Y Position [m]')
ax_path.legend(loc='upper left')
ax_path.grid(True)
ax_path.set_aspect('equal', adjustable='box') # 軸のスケールを固定

# 角度プロット (Roll, Pitch, Yaw)
line_roll, = ax_angles.plot([], [], 'r-', label='Roll [deg]')
line_pitch, = ax_angles.plot([], [], 'g-', label='Pitch [deg]')
line_yaw, = ax_angles.plot([], [], 'b-', label='Yaw [deg]')
ax_angles.set_title('Real-time Angles')
ax_angles.set_xlabel('Time [s]')
ax_angles.set_ylabel('Angle [deg]')
ax_angles.legend(loc='upper left')
ax_angles.grid(True)


def get_imu_data(ser_obj): # 引数名を ser_obj に変更
    """シリアルポートから1行読み込み、IMUデータを返す"""
    try:
        line = ser_obj.readline().decode('utf-8').strip()
        if line:
            ax, ay, az, gx, gy, gz = map(float, line.split(","))
            return np.array([ax, ay, az]), np.array([gx, gy, gz])
    except Exception as e:
        # print(f"データ読み取りエラー: {e}") # リアルタイム描画中は表示を抑制
        pass 
    return None, None

def update_plot(frame):
    """グラフを更新する関数 (FuncAnimationによって繰り返し呼び出される)"""
    if not path_history:
        return line_path, ax_path.lines[1], ax_path.lines[2], line_roll, line_pitch, line_yaw # 変更: lines[1], lines[2]も返す

    # 経路プロットの更新
    current_path = np.array(path_history)
    line_path.set_data(current_path[:, 0], current_path[:, 1])

    # 軸の自動調整（適宜調整してください）
    # x_min, x_max = np.min(current_path[:, 0]), np.max(current_path[:, 0])
    # y_min, y_max = np.min(current_path[:, 1]), np.max(current_path[:, 1])
    # ax_path.set_xlim(x_min - 1, x_max + 1)
    # ax_path.set_ylim(y_min - 1, y_max + 1)
    # もし軸が自動で動きすぎると感じる場合は、固定することも検討

    # 終了地点マーカーの更新
    ax_path.lines[2].set_data(current_path[-1, 0], current_path[-1, 1]) # lines[2]がEndマーカー
    # 開始地点マーカーは最初の一度だけ設定されるため更新不要

    # 角度プロットの更新
    current_timestamps = np.array(timestamps)
    roll_deg = np.rad2deg(np.array([a[0] for a in path_history])) # path_historyからロール角を取り出す
    pitch_deg = np.rad2deg(np.array([a[1] for a in path_history])) # path_historyからピッチ角を取り出す
    yaw_deg = np.rad2deg(np.array([a[2] for a in path_history])) # path_historyからヨー角を取り出す

    line_roll.set_data(current_timestamps, roll_deg)
    line_pitch.set_data(current_timestamps, pitch_deg)
    line_yaw.set_data(current_timestamps, yaw_deg)

    # 角度軸の調整
    if len(current_timestamps) > 1:
        ax_angles.set_xlim(current_timestamps[0], current_timestamps[-1])
    ax_angles.set_ylim(min(roll_deg.min(), pitch_deg.min(), yaw_deg.min()) - 10,
                       max(roll_deg.max(), pitch_deg.max(), yaw_deg.max()) + 10)


    return line_path, ax_path.lines[1], ax_path.lines[2], line_roll, line_pitch, line_yaw

# --- IMU計測とリアルタイム更新を並行して行うための関数 ---
def imu_measurement_thread(ser_obj):
    global path_history, timestamps, current_roll_pitch_yaw

    # --- シリアルポートの設定 ---
    # ser はすでに main() で開かれている想定

    # --- 2. 重力キャリブレーション (最初の3秒間) ---
    print("\n--- 重力キャリブレーションを開始します ---")
    print("3秒間、センサーを完全に静止させてください...")
    
    accel_samples = []
    gyro_samples = []
    calib_start_time = time.time()
    while time.time() - calib_start_time < 3.0:
        accel, gyro = get_imu_data(ser_obj)
        if accel is not None and gyro is not None:
            accel_samples.append(accel)
            gyro_samples.append(gyro)
            # print(f"キャリブレーション中...残り {3.0 - (time.time() - calib_start_time):.1f} 秒", end='\r')
    
    if not accel_samples or not gyro_samples:
        print("\nエラー: キャリブレーションデータを取得できませんでした。")
        ser_obj.close()
        return

    accel_avg = np.mean(accel_samples, axis=0)
    initial_roll = np.arctan2(accel_avg[1], accel_avg[2])
    initial_pitch = np.arctan2(-accel_avg[0], np.sqrt(accel_avg[1]**2 + accel_avg[2]**2))
    gyro_offset = np.mean(gyro_samples, axis=0)
    
    print("\nキャリブレーション完了。")
    print(f"ジャイロオフセット: {gyro_offset}")
    print(f"初期角度(Roll, Pitch): {np.rad2deg(initial_roll):.2f}, {np.rad2deg(initial_pitch):.2f}")

    # --- 変数の初期化 ---
    current_position = np.zeros(2)    # 2次元位置 [x, y]
    current_velocity = np.zeros(2)    # 2次元速度
    
    # 相補フィルター用の変数
    roll, pitch, yaw = initial_roll, initial_pitch, 0.0
    alpha = 0.98  # ジャイロを信頼する割合 (大きいほどジャイロ重視)

    last_time = time.time()
    main_start_time = time.time() # 計測開始時刻
    print("\n--- 計測を開始します ---")
    print("センサーを動かしてください... (停止するには Ctrl+C を押してください)")
    
    try:
        while True:
            accel, gyro = get_imu_data(ser_obj)
            if accel is not None and gyro is not None:
                current_time = time.time()
                dt = current_time - last_time
                if dt <= 0:
                    continue
                last_time = current_time
                
                # ジャイロのオフセットを引く
                gyro -= gyro_offset

                # --- 1. 相補フィルターで姿勢を計算 ---
                accel_roll = np.arctan2(accel[1], accel[2])
                accel_pitch = np.arctan2(-accel[0], np.sqrt(accel[1]**2 + accel[2]**2))

                roll = alpha * (roll + gyro[0] * dt) + (1 - alpha) * accel_roll
                pitch = alpha * (pitch + gyro[1] * dt) + (1 - alpha) * accel_pitch
                yaw += gyro[2] * dt 

                # --- 2. 姿勢から重力成分を除去 ---
                r = R.from_euler('xyz', [roll, pitch, yaw], degrees=False)
                accel_body_rotated_by_imu = r.apply(accel) # センサー座標系の加速度をワールド座標系へ変換
                
                # ワールド座標系の重力ベクトル（ここではY軸を基準）
                gravity_world = np.array([0, 0, -1.0]) # Z軸下向きを重力方向として正規化
                
                # 純粋な移動加速度
                linear_accel_world = accel_body_rotated_by_imu - gravity_world
                
                # --- 3. 経路を計算 ---
                current_velocity += linear_accel_world[:2] * dt # X-Y平面のみ
                current_position += current_velocity * dt
                
                # グローバル変数にデータを追加
                path_history.append(current_position.copy())
                timestamps.append(current_time - main_start_time)
                current_roll_pitch_yaw = [roll, pitch, yaw] # 角度もグローバル変数で共有

                # コンソール表示
                print(f"Roll:{np.rad2deg(roll):6.1f}, Pitch:{np.rad2deg(pitch):6.1f}, Yaw:{np.rad2deg(yaw):6.1f} | Pos:(x={current_position[0]:.2f}, y={current_position[1]:.2f})", end='\r')
            
            plt.pause(0.001) # 描画更新のための短いポーズ

    except KeyboardInterrupt:
        print("\n\n計測を停止しました。")
    finally:
        ser_obj.close()
        print("シリアルポートを閉じました。")


def main():
    SERIAL_PORT = 'COM4' 
    BAUD_RATE = 115200

    ser = None
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        
        # 測定スレッドを実行
        # FuncAnimationの描画ループとIMU計測ループを並行して実行
        # 簡単な方法として、IMU計測ループ内でplt.pause()を使用
        imu_measurement_thread(ser)

    except serial.SerialException as e:
        print(f"エラー: シリアルポートに接続できません。 {e}")
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")
    finally:
        if ser and ser.is_open:
            ser.close()
            
    # 測定終了後、最終的なグラフを保存 (任意)
    if len(path_history) > 1:
        final_path = np.array(path_history)
        plt.figure(figsize=(8, 8))
        plt.plot(final_path[:, 0], final_path[:, 1], 'm-', label='Movement Path')
        plt.plot(final_path[0, 0], final_path[0, 1], 'go', markersize=10, label='Start')
        plt.plot(final_path[-1, 0], final_path[-1, 1], 'rs', markersize=10, label='End')
        plt.title('Final Estimated 2D Path (Complementary Filter)')
        plt.xlabel('X Position [m]')
        plt.ylabel('Y Position [m]')
        plt.legend()
        plt.grid(True)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.savefig('final_path_realtime.png')
        # plt.show() # リアルタイム描画後に再度表示しない

if __name__ == "__main__":
    # FuncAnimation をセットアップ (メインループとは別に描画を更新)
    # intervalはms単位で、100msごとにupdate_plotを呼び出す
    ani = animation.FuncAnimation(fig, update_plot, interval=100, blit=False)    
    # グラフウィンドウを表示 (これがノンブロッキングになる)
    plt.show(block=False) 
    
    # IMU計測処理を開始 (plt.show()とは別の処理フロー)
    main()
    
    # プログラムが完全に終了するまで待機
    input("リアルタイム描画が終了しました。ウィンドウを閉じるにはEnterを押してください。\n")