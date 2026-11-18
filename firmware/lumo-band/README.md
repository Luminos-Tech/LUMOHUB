# Lumo-Band ESP32-C3 BLE Wearable (`v0.4.1-mpu-ble-stream`)

ESP-IDF 5.5.5 firmware for the ESP32-C3 SuperMini wearable + `MPU6050` 6-axis IMU. The band acts as a pure BLE sensor node (no on-device TinyML weights): it advertises as `LUMO-TEST-C3`, maintains the baseline fall state machine, and streams raw IMU packets over BLE so `lumo-hub` (`ESP32-S3`) performs all TinyML classification.

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
- Packet **`0x02` (12 bytes @ 1 Hz / immediate on fall)**: `version=0x02`, `state`, `reserved0=0xFF`, `reserved1=0`, `hr`, `steps`, `battery`
- Packet **`0x03` (20 bytes @ 0.5 Hz idle, boosted to `10 Hz` for 5s on `|a| >= 2.5g`)**: `version=0x03`, `state`, `sequence`, `sensor_age_ms`, raw `ax/ay/az`, raw `gx/gy/gz`, `magnitude_x100`
