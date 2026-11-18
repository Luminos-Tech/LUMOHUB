---
license: mit
tags:
- tinyml
- esp32-s3
- imu
- fall-detection
- physics-mlp
- mpu6050
- lumohub
metrics:
- accuracy
- f1
---

# LumoHub Physics-Feature Event MLP (`LumoEventMLP`) — Approach 2

Physics-informed Multi-Layer Perceptron running on **LumoHub (`ESP32-S3`)** using **16 biomechanical features** extracted from each 3.0-second 6-axis IMU burst window (`free-fall depth`, `impact pulse width`, `3D gravity tilt angle Δθ`, `upright axis collapse`, `post-impact micro-tremor log-score`, `peak jerk`, `recovery energy ratio`, etc.).

## Architecture
- **Input:** `16` standardized physics features (`(x - mean) / std`)
- **Head:** `Linear(16->64)` + ReLU -> `Linear(64->32)` + ReLU -> `Linear(32->5)` (BatchNorm folded into weights)
- **Footprint:** `3,365` floats (`13,460` bytes Flash), `3,232` MACs/inference

## Benchmark Metrics (Held-out Test Split, N = 4,500)
- **Accuracy:** `99.73%`
- **Macro F1:** `99.73%`
- **Wrist-Shock False Alert Rate:** `0.00%`
- **Device Drop-Off False Alert Rate:** `0.00%`
- **Binary Fall Sensitivity (Classes 3 & 4):** `99.94%`

### BLE Packet Drop & Noise Stress Test
| Scenario | Accuracy | Macro F1 |
| :--- | :---: | :---: |
| Clean 10Hz BLE (0% loss) | 99.73% | 99.73% |
| BLE Packet Drop 10% | 96.56% | 96.58% |
| BLE Packet Drop 25% | 89.49% | 89.73% |
| BLE Packet Drop 40% | 80.29% | 80.99% |
| High Sensor Noise (2x) | 99.73% | 99.73% |

### Top Physics Features (Random Forest Baseline Importance)
| Feature | Importance |
| :--- | :---: |
| `f12_post_micro_tremor_score` | 0.1733 |
| `f0_pre_min_mag_g` | 0.1425 |
| `f6_tilt_angle_norm` | 0.0904 |
| `f3_impact_width_ratio` | 0.0863 |
| `f14_recovery_energy_ratio` | 0.0858 |
| `f11_post_std_gyro_norm` | 0.0775 |
| `f9_post_std_mag_g` | 0.0563 |
| `f10_post_mean_gyro_norm` | 0.0550 |
| `f13_peak_jerk_g` | 0.0520 |
| `f2_impact_mean_g` | 0.0508 |
| `f1_impact_peak_g` | 0.0412 |
| `f15_post_stillness_ratio` | 0.0321 |
| `f4_impact_peak_gyro_norm` | 0.0192 |
| `f5_impact_gyro_rms` | 0.0162 |
| `f7_gravity_vertical_drop` | 0.0117 |
| `f8_post_mean_mag_g` | 0.0096 |

## Embedded Firmware Integration
Folded C99 weights + scaler constants are exported in `lumo_hub_feature_model.h` (`lumo_imu_ai.c`).
