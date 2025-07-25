import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# kalman_filter関数は変更なし
def kalman_filter(obs_val, u, ini_sta, ini_cov, tra_mat, inp_mat, obs_mat, pron_cov, obsn_cov):
    """
    カルマンフィルタ（入力項あり）
    """
    # 予測ステップ
    pred_state = tra_mat @ ini_sta + inp_mat @ u # 入力uを考慮
    pred_cov = tra_mat @ ini_cov @ tra_mat.T + pron_cov

    # 更新ステップ
    innovation = obs_val - obs_mat @ pred_state
    innovation_cov = obs_mat @ pred_cov @ obs_mat.T + obsn_cov
    
    kalman_gain = pred_cov @ obs_mat.T @ np.linalg.inv(innovation_cov)
    
    updated_state = pred_state + kalman_gain @ innovation
    updated_cov = (np.eye(len(kalman_gain)) - kalman_gain @ obs_mat) @ pred_cov
    
    return updated_state, updated_cov


def main():
    # --- データの準備 ---
    # ▼▼▼▼▼ 修正点 ▼▼▼▼▼
    # サンプルデータの機能を削除し、常に'imu_data.csv'を読み込むように変更。
    # このため、imu_data.csvファイルがないとエラーで停止します。
    df = pd.read_csv('imu_data.csv')
    # ▲▲▲▲▲ 修正点 ▲▲▲▲▲

    # --- カルマンフィルタの初期設定 ---
    # 状態 [angle, bias]
    state = np.zeros((2, 1)) 
    # 誤差共分散行列
    P = np.eye(2) * 0.1

    # 観測行列 H
    H = np.array([[1.0, 0.0]])
    
    # プロセスノイズの共分散 Q (ジャイロの信頼度)
    Q = np.array([[0.001, 0], [0, 0.003]])

    # 観測ノイズの共分散 R (加速度計の信頼度)
    R = np.array([[0.03]])
    
    # 結果を保存するリスト
    estimated_angles = []
    timestamps = df['timestamp'].values
    
    # --- ループ処理でフィルタを実行 ---
    for i in range(len(df)):
        # サンプリング時間を計算
        dt = timestamps[i] - timestamps[i-1] if i > 0 else timestamps[1] - timestamps[0]
        
        # 状態遷移行列 F と 入力行列 B を更新
        F = np.array([[1.0, -dt], [0.0, 1.0]])
        B = np.array([[dt], [0.0]])
        
        # 観測値 z (加速度計から角度を計算)
        acc_y = df['acc_y'].iloc[i]
        acc_z = df['acc_z'].iloc[i]
        angle_acc = np.arctan2(acc_y, acc_z)
        z = np.array([[angle_acc]])
        
        # 入力 u (ジャイロの測定値)
        u = np.array([[df['gyro_x'].iloc[i]]])

        # カルマンフィルタを実行
        state, P = kalman_filter(z, u, state, P, F, B, H, Q, R)
        
        estimated_angles.append(state[0, 0])

    # --- 結果のプロット ---
    plt.figure(figsize=(12, 6))
    # 加速度計のみから計算した角度（ノイズが多い）
    plt.plot(df['timestamp'], np.arctan2(df['acc_y'], df['acc_z']), 'r-', label='Angle from Accelerometer', alpha=0.5)
    # カルマンフィルタで推定した角度（滑らか）
    plt.plot(df['timestamp'], estimated_angles, 'b-', label='Kalman Filter Estimate', linewidth=2)
    plt.title('IMU Attitude Estimation (Roll Angle)')
    plt.xlabel('Time [s]')
    plt.ylabel('Angle [rad]')
    plt.legend()
    plt.grid(True)
    plt.savefig('kalman_filter_result.png')
    print("グラフを 'kalman_filter_result.png' として保存しました。")


if __name__ == "__main__":
    main()