# State Estimator

ROS 2 package implementing an Error-State EKF that fuses IMU and Doppler radar.  

---

## Configuration

All parameters are defined in `config/revolt_ekf.yaml`:
- IMU & radar topics  
- Process noise `Q`  
- Doppler noise `radar_sigma_vr`  
- Bias time constants  
- Radar–IMU extrinsics (`l_BR_B`, `q_R_B`)  
- Initial covariance values
