---
license: mit
tags:
- tinyml
- esp32-s3
- imu
- fall-detection
- 1d-cnn
- mpu6050
- lumohub
metrics:
- accuracy
- f1
---

# LumoHub IMU Burst 1D-CNN (`LumoBurst1DCNN`) — Approach 1

End-to-end 1D Convolutional Neural Network running directly on the **LumoHub (`ESP32-S3`)** to eliminate wearable wrist false alerts and classify 3.0-second 6-axis IMU burst sequences (`30 samples x 6 channels` at `10 Hz`) received from the wearable over BLE (`0x03` MPU6050 diagnostic stream).

## Architecture
- **Input:** `[30, 6]` normalized `(ax/8g, ay/8g, az/8g, gx/250dps, gy/250dps, gz/250dps)`
- **Backbone:** `Conv1D(6->24, k=5)` + ReLU + `MaxPool(2)` -> `Conv1D(24->32, k=3)` + ReLU + `MaxPool(3)`
- **Dual Pooling:** Concatenated `GlobalAvgPool1D` + `GlobalMaxPool1D` (`64` dims)
- **Head:** `Linear(64->32)` + ReLU + `Linear(32->5)`
- **Footprint:** `5,325` parameters (`21,300` bytes Flash), `58,368` MACs/inference

## 5-Class Taxonomy
0. `NORMAL_ADL` — Walking, sitting, lying down smoothly, everyday movement
1. `WRIST_SHOCK_FALSE_ALERT` — High-G wrist shock (clapping, table slap, knocking) without body collapse
2. `DEVICE_DROP_OFF_WRIST` — Unbuckled/dropped wearable onto a rigid surface (zero physiological tremor)
3. `SOFT_FALL_RECOVERING` — Stumble/fall with posture collapse followed by self-recovery motion
4. `CRITICAL_FALL_IMMOBILE` — Severe fall with posture collapse and prolonged immobility

## Benchmark Metrics (Held-out Test Split, N = 4,500)
- **Accuracy:** `98.38%`
- **Macro F1:** `98.38%`
- **Wrist-Shock False Alert Rate:** `0.11%`
- **Device Drop-Off False Alert Rate:** `0.00%`
- **Binary Fall Sensitivity (Classes 3 & 4):** `99.33%`

### BLE Packet Drop & Noise Stress Test
| Scenario | Accuracy | Macro F1 |
| :--- | :---: | :---: |
| Clean 10Hz BLE (0% loss) | 98.38% | 98.38% |
| BLE Packet Drop 10% | 93.60% | 93.62% |
| BLE Packet Drop 25% | 85.18% | 85.28% |
| BLE Packet Drop 40% | 75.16% | 75.40% |
| High Sensor Noise (2x) | 98.44% | 98.44% |

## Embedded Firmware Integration
Folded C99 weights are exported in `lumo_hub_cnn_model.h` for zero-dependency inference on `ESP32-S3` (`lumo_imu_ai.c`).
