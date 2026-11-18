# Lumo-Band ESP32-C3 BLE Wearable (`v0.4.0-mpu-burst-ai`)

ESP-IDF 5.5.5 firmware for the ESP32-C3 SuperMini wearable + `MPU6050` 6-axis IMU. The band advertises as `LUMO-TEST-C3`, maintains an on-board fall state machine, streams raw IMU diagnostics over BLE to `lumo-hub` (`ESP32-S3`), and runs a self-test / edge copy of the 3-model TinyML false-alert filter.

## Wiring (`MPU6050` via I2C)

- `SDA`: `GPIO 3`
- `SCL`: `GPIO 4`
- `Address`: `0x68` (`WHO_AM_I = 0x68`, `±8g` accel, `±250 dps` gyro @ `10 Hz`)

## Build and flash

Open the `Lumo Band C3 ESP-IDF` terminal:

```powershell
idf.py set-target esp32c3
idf.py build
idf.py -p COMx flash monitor
```

## BLE contract

- Service: `12345678-1234-5678-1234-56789abcdef0`
- Characteristic: `12345678-1234-5678-1234-56789abcdef1`
- Properties: READ and NOTIFY
- Packet **`0x02` (12 bytes @ 1 Hz / immediate on fall)**: `version=0x02`, `state`, `edge_ai_class`, `edge_ai_conf_pct`, `hr`, `steps`, `battery`
- Packet **`0x03` (20 bytes @ 0.5 Hz idle, boosted to `10 Hz` for 5s on `|a| >= 2.5g`)**: `version=0x03`, `state`, `sequence`, `sensor_age_ms`, raw `ax/ay/az`, raw `gx/gy/gz`, `magnitude_x100`

## On-Device AI Self-Test (`RF-25 + Physics-MLP + 1D-CNN`)

On boot, [`lumo_ai_run_self_test()`](src/lumo_imu_ai.c) verifies all 5 scenarios (`NORMAL_ADL`, `WRIST_SHOCK_FALSE_ALERT`, `DEVICE_DROP_OFF_WRIST`, `SOFT_FALL_RECOVERING`, `CRITICAL_FALL_IMMOBILE`), suppressing `IMPACT_WAIT` / `FALL_CONFIRMED` when the fused ensemble classifies wrist shock, device drop-off, or normal ADL.
