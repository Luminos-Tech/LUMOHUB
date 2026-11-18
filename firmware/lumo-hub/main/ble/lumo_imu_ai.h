/**
 * @file lumo_imu_ai.h
 * @brief Dual-Architecture On-Hub TinyML IMU Fall & False-Alert Classifier.
 *
 * Implements two complementary models trained on SisFall + LumoHub 6-axis IMU data:
 *   - Approach 1: LumoBurst1DCNN  (Raw 30x6 10Hz IMU burst tensor)
 *   - Approach 2: LumoEventMLP    (16 biomechanical physics features)
 *
 * Generates fused predictions, latency metrics, false-alert filtering flags,
 * and Vietnamese diagnostic commentary directly on the ESP32-S3 Hub.
 */
#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LUMO_AI_WINDOW_LEN   30
#define LUMO_AI_CHANNELS     6
#define LUMO_AI_NUM_FEATURES 16

typedef enum {
    LUMO_AI_CLASS_NORMAL_ADL        = 0,
    LUMO_AI_CLASS_WRIST_SHOCK_FALSE = 1,
    LUMO_AI_CLASS_DEVICE_DROPPED    = 2,
    LUMO_AI_CLASS_SOFT_FALL_RECOVER = 3,
    LUMO_AI_CLASS_CRITICAL_FALL     = 4,
    LUMO_AI_NUM_CLASSES             = 5,
} lumo_ai_class_t;

typedef struct {
    lumo_ai_class_t pred_class;
    float probs[LUMO_AI_NUM_CLASSES];
    uint8_t confidence_pct;
    uint32_t latency_us;
} lumo_ai_pred_t;

typedef struct {
    lumo_ai_pred_t cnn;
    lumo_ai_pred_t mlp;
    lumo_ai_pred_t rf;
    lumo_ai_class_t final_class;
    float ensemble_probs[LUMO_AI_NUM_CLASSES];
    uint8_t final_confidence_pct;
    float features[LUMO_AI_NUM_FEATURES];
    float peak_g;
    float tilt_deg;
    float pulse_width_ratio;
    bool models_agree;
    bool should_suppress_alarm;
    bool should_trigger_voice_check;
    char commentary[188];
} lumo_ai_dual_result_t;

void lumo_ai_init(void);
void lumo_ai_reset_buffer(void);
void lumo_ai_push_sample(float ax_g, float ay_g, float az_g,
                         float gx_dps, float gy_dps, float gz_dps,
                         uint64_t timestamp_ms);
void lumo_ai_push_raw_mpu6050(int16_t ax_raw, int16_t ay_raw, int16_t az_raw,
                              int16_t gx_raw, int16_t gy_raw, int16_t gz_raw,
                              uint64_t timestamp_ms);
bool lumo_ai_buffer_ready(void);
uint8_t lumo_ai_buffer_count(void);

void lumo_ai_extract_features(const float window[LUMO_AI_WINDOW_LEN][LUMO_AI_CHANNELS],
                              float out_features[LUMO_AI_NUM_FEATURES]);
void lumo_ai_predict_cnn(const float window[LUMO_AI_WINDOW_LEN][LUMO_AI_CHANNELS],
                         lumo_ai_pred_t *out);
void lumo_ai_predict_feature_mlp(const float features[LUMO_AI_NUM_FEATURES],
                                 lumo_ai_pred_t *out);
void lumo_ai_predict_feature_rf(const float features[LUMO_AI_NUM_FEATURES],
                                lumo_ai_pred_t *out);
void lumo_ai_evaluate_window(const float window[LUMO_AI_WINDOW_LEN][LUMO_AI_CHANNELS],
                             lumo_ai_dual_result_t *out);
bool lumo_ai_evaluate_ring(lumo_ai_dual_result_t *out);
bool lumo_ai_ring_has_recent_peak(float threshold_g);
void lumo_ai_run_self_test(void);
const char *lumo_ai_class_name(lumo_ai_class_t cls);

#ifdef __cplusplus
}
#endif
