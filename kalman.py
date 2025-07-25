import numpy as np

def kalman_filter(obs_acc, obs_gyro, ini_sta, ini_cov, tra_mat, obs_mat, pron_cov, obsn_cov):
    """
    Kalman filter implementation for sensor data fusion.
    
    Parameters:
    obs_acc (np.ndarray): Observed acceleration data.
    obs_gyro (np.ndarray): Observed gyroscope data.
    ini_ata (np.ndarray): Initial state estimate.
    ini_cov (np.ndarray): Initial covariance estimate.
    tra_mat (np.ndarray): State transition matrix.
    obs_mat (np.ndarray): Observation matrix.
    pron_cov (np.ndarray): Process noise covariance.
    obsn_cov (np.ndarray): Observation noise covariance.

    Returns:
    np.ndarray: Updated state estimate after applying the Kalman filter.
    """
    
    # Prediction step
    pred_state = tra_mat @ ini_sta
    pred_cov = tra_mat @ ini_cov @ tra_mat.T + pron_cov
    
    # Update step
    obs_pred = obs_mat @ pred_state
    innovation = np.concatenate((obs_acc, obs_gyro)) - obs_pred
    innovation_cov = obs_mat @ pred_cov @ obs_mat.T + obsn_cov
    
    kalman_gain = pred_cov @ obs_mat.T @ np.linalg.inv(innovation_cov)
    
    updated_state = pred_state + kalman_gain @ innovation
    updated_cov = (np.eye(len(kalman_gain)) - kalman_gain @ obs_mat) @ pred_cov
    
    return updated_state, updated_cov

def main():
    # Example usage of the Kalman filter
    obs_acc = np.array([0.1, 0.2, 0.3])
    obs_gyro = np.array([0.01, 0.02, 0.03])
    ini_sta = np.zeros(6)  # Initial state
    ini_cov = np.eye(6) * 0.1  # Initial covariance
    tra_mat = np.eye(6)  # State transition matrix
    obs_mat = np.eye(6)[:3]  # Observation matrix for acceleration
    pron_cov = np.eye(6) * 0.01  # Process noise covariance
    obsn_cov = np.eye(3) * 0.05  # Observation noise covariance

    updated_state, updated_cov = kalman_filter(obs_acc, obs_gyro, ini_sta, ini_cov, tra_mat, obs_mat, pron_cov, obsn_cov)
    
    print("Updated State:", updated_state)
    print("Updated Covariance:", updated_cov)

if __name__ == "__main__":
    main()
# This code implements a basic Kalman filter for sensor data fusion, specifically for accelerometer and gyroscope data.
