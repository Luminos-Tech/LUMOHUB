/**
 * @file lumo_imu_ai.c
 * @brief Dual-Architecture TinyML IMU Inference & Commentary Engine for LumoHub (ESP32-S3).
 *
 * Runs both:
 *   1) LumoBurst1DCNN  (raw 30x6 10Hz IMU sequence)
 *   2) LumoEventMLP    (16 physics-informed features)
 * on the Hub, compares them, and produces Vietnamese diagnostic commentary
 * for false-alert suppression and voice check-in escalation.
 */

#include "lumo_imu_ai.h"
#if LUMO_AI_USE_CNN
#include "lumo_hub_cnn_model.h"
#endif
#if LUMO_AI_USE_MLP
#include "lumo_hub_feature_model.h"
#endif
#if LUMO_AI_USE_RF
#include "lumo_hub_rf_model.h"
#endif

#include <math.h>
#include <stdio.h>
#include <string.h>

#if defined(ESP_PLATFORM) || defined(ARDUINO_ARCH_ESP32)
#include "esp_timer.h"
static uint64_t lumo_ai_now_us(void)
{
    return (uint64_t)esp_timer_get_time();
}
#else
static uint64_t lumo_ai_now_us(void)
{
    return 0ULL;
}
#endif

#define LUMO_MPU_ACCEL_LSB_PER_G  4096.0f
#define LUMO_MPU_GYRO_LSB_PER_DPS 131.0f
#define LUMO_RAD_TO_DEG           57.2957795f

static float s_ring[LUMO_AI_WINDOW_LEN][LUMO_AI_CHANNELS];
static uint8_t s_ring_head = 0;
static uint8_t s_ring_count = 0;

static const char *s_class_names[LUMO_AI_NUM_CLASSES] = {
    "NORMAL_ADL",
    "WRIST_SHOCK_FALSE_ALERT",
    "DEVICE_DROP_OFF_WRIST",
    "SOFT_FALL_RECOVERING",
    "CRITICAL_FALL_IMMOBILE",
};

const char *lumo_ai_active_model_name(void)
{
#if LUMO_AI_ACTIVE_MODEL == LUMO_AI_MODEL_MLP
    return "LumoEventMLP";
#elif LUMO_AI_ACTIVE_MODEL == LUMO_AI_MODEL_RF25
    return "Compact-RF25";
#elif LUMO_AI_ACTIVE_MODEL == LUMO_AI_MODEL_CNN
    return "LumoBurst1DCNN";
#else
    return "RF-25 + Physics-MLP + 1D-CNN";
#endif
}

const char *lumo_ai_class_name(lumo_ai_class_t cls)
{
    if ((int)cls >= 0 && (int)cls < LUMO_AI_NUM_CLASSES) {
        return s_class_names[(int)cls];
    }
    return "UNKNOWN";
}

void lumo_ai_init(void)
{
    lumo_ai_reset_buffer();
}

void lumo_ai_reset_buffer(void)
{
    memset(s_ring, 0, sizeof(s_ring));
    s_ring_head = 0;
    s_ring_count = 0;
}

void lumo_ai_push_sample(float ax_g, float ay_g, float az_g,
                         float gx_dps, float gy_dps, float gz_dps,
                         uint64_t timestamp_ms)
{
    (void)timestamp_ms;
    s_ring[s_ring_head][0] = ax_g;
    s_ring[s_ring_head][1] = ay_g;
    s_ring[s_ring_head][2] = az_g;
    s_ring[s_ring_head][3] = gx_dps;
    s_ring[s_ring_head][4] = gy_dps;
    s_ring[s_ring_head][5] = gz_dps;

    s_ring_head = (uint8_t)((s_ring_head + 1U) % LUMO_AI_WINDOW_LEN);
    if (s_ring_count < LUMO_AI_WINDOW_LEN) {
        s_ring_count++;
    }
}

void lumo_ai_push_raw_mpu6050(int16_t ax_raw, int16_t ay_raw, int16_t az_raw,
                              int16_t gx_raw, int16_t gy_raw, int16_t gz_raw,
                              uint64_t timestamp_ms)
{
    lumo_ai_push_sample(
        (float)ax_raw / LUMO_MPU_ACCEL_LSB_PER_G,
        (float)ay_raw / LUMO_MPU_ACCEL_LSB_PER_G,
        (float)az_raw / LUMO_MPU_ACCEL_LSB_PER_G,
        (float)gx_raw / LUMO_MPU_GYRO_LSB_PER_DPS,
        (float)gy_raw / LUMO_MPU_GYRO_LSB_PER_DPS,
        (float)gz_raw / LUMO_MPU_GYRO_LSB_PER_DPS,
        timestamp_ms
    );
}

bool lumo_ai_buffer_ready(void)
{
    return s_ring_count >= LUMO_AI_WINDOW_LEN;
}

uint8_t lumo_ai_buffer_count(void)
{
    return s_ring_count;
}

#if LUMO_AI_USE_CNN || LUMO_AI_USE_MLP
static void softmax_5(const float logits[LUMO_AI_NUM_CLASSES], float probs[LUMO_AI_NUM_CLASSES])
{
    float max_v = logits[0];
    for (int i = 1; i < LUMO_AI_NUM_CLASSES; i++) {
        if (logits[i] > max_v) {
            max_v = logits[i];
        }
    }
    float sum_e = 0.0f;
    for (int i = 0; i < LUMO_AI_NUM_CLASSES; i++) {
        float e = expf(logits[i] - max_v);
        probs[i] = e;
        sum_e += e;
    }
    if (sum_e > 0.0f) {
        for (int i = 0; i < LUMO_AI_NUM_CLASSES; i++) {
            probs[i] /= sum_e;
        }
    }
}
#endif

static lumo_ai_class_t argmax_5(const float probs[LUMO_AI_NUM_CLASSES], uint8_t *conf_pct)
{
    int best = 0;
    float best_p = probs[0];
    for (int i = 1; i < LUMO_AI_NUM_CLASSES; i++) {
        if (probs[i] > best_p) {
            best_p = probs[i];
            best = i;
        }
    }
    if (conf_pct != NULL) {
        int pct = (int)(best_p * 100.0f + 0.5f);
        if (pct < 0) pct = 0;
        if (pct > 100) pct = 100;
        *conf_pct = (uint8_t)pct;
    }
    return (lumo_ai_class_t)best;
}

void lumo_ai_extract_features(const float window[LUMO_AI_WINDOW_LEN][LUMO_AI_CHANNELS],
                              float out_features[LUMO_AI_NUM_FEATURES])
{
    float mag_a[LUMO_AI_WINDOW_LEN];
    float mag_g[LUMO_AI_WINDOW_LEN];
    float gyro_norm[LUMO_AI_WINDOW_LEN];

    for (int t = 0; t < LUMO_AI_WINDOW_LEN; t++) {
        float ax = window[t][0], ay = window[t][1], az = window[t][2];
        float gx = window[t][3], gy = window[t][4], gz = window[t][5];
        mag_a[t] = sqrtf(ax * ax + ay * ay + az * az);
        mag_g[t] = sqrtf(gx * gx + gy * gy + gz * gz);
        gyro_norm[t] = mag_g[t] / 250.0f;
    }

    /* f0: pre_min_mag_g [0..11] */
    float pre_min = mag_a[0];
    for (int t = 1; t < 12; t++) {
        if (mag_a[t] < pre_min) pre_min = mag_a[t];
    }
    out_features[0] = pre_min;

    /* f1: impact_peak_g [0..29] */
    float peak_g = mag_a[0];
    for (int t = 1; t < LUMO_AI_WINDOW_LEN; t++) {
        if (mag_a[t] > peak_g) peak_g = mag_a[t];
    }
    out_features[1] = peak_g;

    /* f2: impact_mean_g [9..14] */
    float sum_imp_a = 0.0f;
    for (int t = 9; t <= 14; t++) {
        sum_imp_a += mag_a[t];
    }
    out_features[2] = sum_imp_a / 6.0f;

    /* f3: impact_width_ratio >= 2.2g in [8..16] (9 samples) */
    int width_cnt = 0;
    float peak_gyro = gyro_norm[8];
    for (int t = 8; t <= 16; t++) {
        if (mag_a[t] >= 2.2f) width_cnt++;
        if (gyro_norm[t] > peak_gyro) peak_gyro = gyro_norm[t];
    }
    out_features[3] = (float)width_cnt / 9.0f;

    /* f4: impact_peak_gyro_norm */
    out_features[4] = peak_gyro;

    /* f5: impact_gyro_rms [9..14] */
    float sum_g_sq = 0.0f;
    for (int t = 9; t <= 14; t++) {
        sum_g_sq += gyro_norm[t] * gyro_norm[t];
    }
    out_features[5] = sqrtf(sum_g_sq / 6.0f);

    /* f6: tilt_angle_norm between mean(0..7) and mean(18..29) */
    float a_pre[3] = {0.0f, 0.0f, 0.0f};
    float a_post[3] = {0.0f, 0.0f, 0.0f};
    for (int t = 0; t < 8; t++) {
        a_pre[0] += window[t][0];
        a_pre[1] += window[t][1];
        a_pre[2] += window[t][2];
    }
    for (int c = 0; c < 3; c++) a_pre[c] /= 8.0f;

    for (int t = 18; t < 30; t++) {
        a_post[0] += window[t][0];
        a_post[1] += window[t][1];
        a_post[2] += window[t][2];
    }
    for (int c = 0; c < 3; c++) a_post[c] /= 12.0f;

    float n_pre = sqrtf(a_pre[0] * a_pre[0] + a_pre[1] * a_pre[1] + a_pre[2] * a_pre[2]);
    float n_post = sqrtf(a_post[0] * a_post[0] + a_post[1] * a_post[1] + a_post[2] * a_post[2]);
    float dot = a_pre[0] * a_post[0] + a_pre[1] * a_post[1] + a_pre[2] * a_post[2];
    float cos_t = dot / (n_pre * n_post + 1e-6f);
    if (cos_t < -1.0f) cos_t = -1.0f;
    if (cos_t > 1.0f) cos_t = 1.0f;
    float tilt_deg = acosf(cos_t) * LUMO_RAD_TO_DEG;
    out_features[6] = tilt_deg / 180.0f;

    /* f7: gravity_vertical_drop */
    int primary_axis = 0;
    float max_abs_pre = fabsf(a_pre[0]);
    for (int c = 1; c < 3; c++) {
        float v = fabsf(a_pre[c]);
        if (v > max_abs_pre) {
            max_abs_pre = v;
            primary_axis = c;
        }
    }
    out_features[7] = max_abs_pre - fabsf(a_post[primary_axis]);

    /* f8..f11: post mean/std [16..29] (14 samples, ddof=0) */
    float sum_pa = 0.0f, sum_pg = 0.0f;
    for (int t = 16; t < 30; t++) {
        sum_pa += mag_a[t];
        sum_pg += gyro_norm[t];
    }
    float mean_pa = sum_pa / 14.0f;
    float mean_pg = sum_pg / 14.0f;
    float var_pa = 0.0f, var_pg = 0.0f;
    for (int t = 16; t < 30; t++) {
        float da = mag_a[t] - mean_pa;
        float dg = gyro_norm[t] - mean_pg;
        var_pa += da * da;
        var_pg += dg * dg;
    }
    out_features[8] = mean_pa;
    out_features[9] = sqrtf(var_pa / 14.0f);
    out_features[10] = mean_pg;
    out_features[11] = sqrtf(var_pg / 14.0f);

    /* f12: post_micro_tremor_score over [18..29] (12 samples) */
    float sum_mg = 0.0f;
    for (int t = 18; t < 30; t++) sum_mg += mag_g[t];
    float mean_mg = sum_mg / 12.0f;
    float var_mg = 0.0f;
    for (int t = 18; t < 30; t++) {
        float d = mag_g[t] - mean_mg;
        var_mg += d * d;
    }
    float micro_std = sqrtf(var_mg / 12.0f);
    out_features[12] = log10f(micro_std + 1e-4f) / 4.0f + 1.0f;

    /* f13: peak_jerk_g [8..15] */
    float max_jerk = 0.0f;
    for (int t = 8; t <= 15; t++) {
        float j = fabsf(mag_a[t] - mag_a[t - 1]);
        if (j > max_jerk) max_jerk = j;
    }
    out_features[13] = max_jerk;

    /* f14: recovery_energy_ratio */
    float post_e = 0.0f, imp_e = 0.0f;
    for (int t = 16; t < 30; t++) {
        post_e += fabsf(mag_a[t] - 1.0f) + mag_g[t] / 100.0f;
    }
    for (int t = 9; t <= 14; t++) {
        imp_e += fabsf(mag_a[t] - 1.0f) + mag_g[t] / 100.0f;
    }
    out_features[14] = post_e / (imp_e + 0.1f);

    /* f15: post_stillness_ratio [16..29] */
    int still_cnt = 0;
    for (int t = 16; t < 30; t++) {
        if (mag_a[t] >= 0.85f && mag_a[t] <= 1.15f && mag_g[t] < 12.0f) {
            still_cnt++;
        }
    }
    out_features[15] = (float)still_cnt / 14.0f;
}

void lumo_ai_predict_cnn(const float window[LUMO_AI_WINDOW_LEN][LUMO_AI_CHANNELS],
                         lumo_ai_pred_t *out)
{
#if LUMO_AI_USE_CNN
    uint64_t t0 = lumo_ai_now_us();

    static float x_norm[6][30];
    for (int t = 0; t < 30; t++) {
        x_norm[0][t] = window[t][0] / 8.0f;
        x_norm[1][t] = window[t][1] / 8.0f;
        x_norm[2][t] = window[t][2] / 8.0f;
        x_norm[3][t] = window[t][3] / 250.0f;
        x_norm[4][t] = window[t][4] / 250.0f;
        x_norm[5][t] = window[t][5] / 250.0f;
    }

    /* Layer 1: Conv1D(6 -> 24, k=5, pad=2) + ReLU + MaxPool1D(2) -> [24][15] */
    static float h1_pool[24][15];
    for (int oc = 0; oc < 24; oc++) {
        float conv_row[30];
        for (int t = 0; t < 30; t++) {
            float acc = LUMO_CNN_CONV1_B[oc];
            for (int ic = 0; ic < 6; ic++) {
                for (int k = 0; k < 5; k++) {
                    int ti = t + k - 2;
                    if (ti >= 0 && ti < 30) {
                        acc += x_norm[ic][ti] * LUMO_CNN_CONV1_W[oc][ic][k];
                    }
                }
            }
            conv_row[t] = acc > 0.0f ? acc : 0.0f;
        }
        for (int p = 0; p < 15; p++) {
            float a = conv_row[p * 2];
            float b = conv_row[p * 2 + 1];
            h1_pool[oc][p] = (a > b) ? a : b;
        }
    }

    /* Layer 2: Conv1D(24 -> 32, k=3, pad=1) + ReLU + MaxPool1D(3) -> [32][5] + GAP/GMP */
    float feat[64];
    for (int oc = 0; oc < 32; oc++) {
        float conv_row[15];
        for (int t = 0; t < 15; t++) {
            float acc = LUMO_CNN_CONV2_B[oc];
            for (int ic = 0; ic < 24; ic++) {
                for (int k = 0; k < 3; k++) {
                    int ti = t + k - 1;
                    if (ti >= 0 && ti < 15) {
                        acc += h1_pool[ic][ti] * LUMO_CNN_CONV2_W[oc][ic][k];
                    }
                }
            }
            conv_row[t] = acc > 0.0f ? acc : 0.0f;
        }
        float sum_p = 0.0f;
        float max_p = -1e9f;
        for (int p = 0; p < 5; p++) {
            float m = conv_row[p * 3];
            if (conv_row[p * 3 + 1] > m) m = conv_row[p * 3 + 1];
            if (conv_row[p * 3 + 2] > m) m = conv_row[p * 3 + 2];
            sum_p += m;
            if (m > max_p) max_p = m;
        }
        feat[oc] = sum_p / 5.0f;
        feat[32 + oc] = max_p;
    }

    /* FC1(64 -> 32) + ReLU */
    float fc1[32];
    for (int o = 0; o < 32; o++) {
        float acc = LUMO_CNN_FC1_B[o];
        for (int i = 0; i < 64; i++) {
            acc += feat[i] * LUMO_CNN_FC1_W[o][i];
        }
        fc1[o] = acc > 0.0f ? acc : 0.0f;
    }

    /* FC2(32 -> 5) */
    float logits[LUMO_AI_NUM_CLASSES];
    for (int o = 0; o < LUMO_AI_NUM_CLASSES; o++) {
        float acc = LUMO_CNN_FC2_B[o];
        for (int i = 0; i < 32; i++) {
            acc += fc1[i] * LUMO_CNN_FC2_W[o][i];
        }
        logits[o] = acc;
    }

    uint64_t t1 = lumo_ai_now_us();
    softmax_5(logits, out->probs);
    out->pred_class = argmax_5(out->probs, &out->confidence_pct);
    out->latency_us = (uint32_t)(t1 >= t0 ? (t1 - t0) : 0ULL);
#else
    (void)window;
    memset(out, 0, sizeof(*out));
#endif
}

void lumo_ai_predict_feature_mlp(const float features[LUMO_AI_NUM_FEATURES],
                                 lumo_ai_pred_t *out)
{
#if LUMO_AI_USE_MLP
    uint64_t t0 = lumo_ai_now_us();

    float x_in[LUMO_MLP_IN_FEATURES];
    for (int i = 0; i < LUMO_MLP_IN_FEATURES; i++) {
        x_in[i] = (features[i] - LUMO_MLP_SCALER_MEAN[i]) / (LUMO_MLP_SCALER_STD[i] + 1e-6f);
    }

    float h1[LUMO_MLP_HIDDEN1];
    for (int o = 0; o < LUMO_MLP_HIDDEN1; o++) {
        float acc = LUMO_MLP_FC1_B[o];
        for (int i = 0; i < LUMO_MLP_IN_FEATURES; i++) {
            acc += x_in[i] * LUMO_MLP_FC1_W[o][i];
        }
        h1[o] = acc > 0.0f ? acc : 0.0f;
    }

    float h2[LUMO_MLP_HIDDEN2];
    for (int o = 0; o < LUMO_MLP_HIDDEN2; o++) {
        float acc = LUMO_MLP_FC2_B[o];
        for (int i = 0; i < LUMO_MLP_HIDDEN1; i++) {
            acc += h1[i] * LUMO_MLP_FC2_W[o][i];
        }
        h2[o] = acc > 0.0f ? acc : 0.0f;
    }

    float logits[LUMO_AI_NUM_CLASSES];
    for (int o = 0; o < LUMO_AI_NUM_CLASSES; o++) {
        float acc = LUMO_MLP_FC3_B[o];
        for (int i = 0; i < LUMO_MLP_HIDDEN2; i++) {
            acc += h2[i] * LUMO_MLP_FC3_W[o][i];
        }
        logits[o] = acc;
    }

    uint64_t t1 = lumo_ai_now_us();
    softmax_5(logits, out->probs);
    out->pred_class = argmax_5(out->probs, &out->confidence_pct);
    out->latency_us = (uint32_t)(t1 >= t0 ? (t1 - t0) : 0ULL);
#else
    (void)features;
    memset(out, 0, sizeof(*out));
#endif
}

void lumo_ai_predict_feature_rf(const float features[LUMO_AI_NUM_FEATURES],
                                lumo_ai_pred_t *out)
{
#if LUMO_AI_USE_RF
    uint64_t t0 = lumo_ai_now_us();

    float acc[LUMO_AI_NUM_CLASSES] = {0.0f};
    for (int t = 0; t < LUMO_RF_NUM_TREES; t++) {
        int node = (int)LUMO_RF_TREE_OFFSET[t];
        while (LUMO_RF_FEATURE[node] >= 0) {
            int f_idx = (int)LUMO_RF_FEATURE[node];
            if (features[f_idx] <= LUMO_RF_THRESHOLD[node]) {
                node = (int)LUMO_RF_LEFT[node];
            } else {
                node = (int)LUMO_RF_RIGHT[node];
            }
        }
        int leaf_idx = (int)LUMO_RF_LEFT[node];
        for (int c = 0; c < LUMO_AI_NUM_CLASSES; c++) {
            acc[c] += (float)LUMO_RF_LEAF_PROBS[leaf_idx][c] / 255.0f;
        }
    }

    const float inv_t = 1.0f / (float)LUMO_RF_NUM_TREES;
    for (int c = 0; c < LUMO_AI_NUM_CLASSES; c++) {
        out->probs[c] = acc[c] * inv_t;
    }

    uint64_t t1 = lumo_ai_now_us();
    out->pred_class = argmax_5(out->probs, &out->confidence_pct);
    out->latency_us = (uint32_t)(t1 >= t0 ? (t1 - t0) : 0ULL);
#else
    (void)features;
    memset(out, 0, sizeof(*out));
#endif
}

static void format_active_tag(const lumo_ai_dual_result_t *res, char *tag, size_t tag_len)
{
#if LUMO_AI_ACTIVE_MODEL == LUMO_AI_MODEL_MLP
    snprintf(tag, tag_len, "MLP:%u%%", res->mlp.confidence_pct);
#elif LUMO_AI_ACTIVE_MODEL == LUMO_AI_MODEL_RF25
    snprintf(tag, tag_len, "RF:%u%%", res->rf.confidence_pct);
#elif LUMO_AI_ACTIVE_MODEL == LUMO_AI_MODEL_CNN
    snprintf(tag, tag_len, "CNN:%u%%", res->cnn.confidence_pct);
#else
    snprintf(tag, tag_len, "RF:%u%% MLP:%u%% CNN:%u%%",
             res->rf.confidence_pct, res->mlp.confidence_pct, res->cnn.confidence_pct);
#endif
}

void lumo_ai_format_metrics(const lumo_ai_dual_result_t *res, char *buf, size_t buf_len)
{
    if (res == NULL || buf == NULL || buf_len == 0) {
        return;
    }
#if LUMO_AI_ACTIVE_MODEL == LUMO_AI_MODEL_MLP
    snprintf(buf, buf_len, "MLP=%s(%u%%,%luus) | FINAL=%s(%u%%)",
             lumo_ai_class_name(res->mlp.pred_class), res->mlp.confidence_pct, (unsigned long)res->mlp.latency_us,
             lumo_ai_class_name(res->final_class), res->final_confidence_pct);
#elif LUMO_AI_ACTIVE_MODEL == LUMO_AI_MODEL_RF25
    snprintf(buf, buf_len, "RF=%s(%u%%,%luus) | FINAL=%s(%u%%)",
             lumo_ai_class_name(res->rf.pred_class), res->rf.confidence_pct, (unsigned long)res->rf.latency_us,
             lumo_ai_class_name(res->final_class), res->final_confidence_pct);
#elif LUMO_AI_ACTIVE_MODEL == LUMO_AI_MODEL_CNN
    snprintf(buf, buf_len, "CNN=%s(%u%%,%luus) | FINAL=%s(%u%%)",
             lumo_ai_class_name(res->cnn.pred_class), res->cnn.confidence_pct, (unsigned long)res->cnn.latency_us,
             lumo_ai_class_name(res->final_class), res->final_confidence_pct);
#else
    snprintf(buf, buf_len,
             "RF=%s(%u%%,%luus) | MLP=%s(%u%%,%luus) | CNN=%s(%u%%,%luus) | FUSED=%s(%u%%)",
             lumo_ai_class_name(res->rf.pred_class), res->rf.confidence_pct, (unsigned long)res->rf.latency_us,
             lumo_ai_class_name(res->mlp.pred_class), res->mlp.confidence_pct, (unsigned long)res->mlp.latency_us,
             lumo_ai_class_name(res->cnn.pred_class), res->cnn.confidence_pct, (unsigned long)res->cnn.latency_us,
             lumo_ai_class_name(res->final_class), res->final_confidence_pct);
#endif
}

void lumo_ai_evaluate_window(const float window[LUMO_AI_WINDOW_LEN][LUMO_AI_CHANNELS],
                             lumo_ai_dual_result_t *out)
{
    memset(out, 0, sizeof(*out));
    lumo_ai_extract_features(window, out->features);
    lumo_ai_predict_cnn(window, &out->cnn);
    lumo_ai_predict_feature_mlp(out->features, &out->mlp);
    lumo_ai_predict_feature_rf(out->features, &out->rf);

#if LUMO_AI_ACTIVE_MODEL == LUMO_AI_MODEL_MLP
    memcpy(out->ensemble_probs, out->mlp.probs, sizeof(out->ensemble_probs));
    out->models_agree = true;
#elif LUMO_AI_ACTIVE_MODEL == LUMO_AI_MODEL_RF25
    memcpy(out->ensemble_probs, out->rf.probs, sizeof(out->ensemble_probs));
    out->models_agree = true;
#elif LUMO_AI_ACTIVE_MODEL == LUMO_AI_MODEL_CNN
    memcpy(out->ensemble_probs, out->cnn.probs, sizeof(out->ensemble_probs));
    out->models_agree = true;
#else
    for (int c = 0; c < LUMO_AI_NUM_CLASSES; c++) {
        out->ensemble_probs[c] =
            0.20f * out->cnn.probs[c] +
            0.40f * out->mlp.probs[c] +
            0.40f * out->rf.probs[c];
    }
    out->models_agree =
        (out->cnn.pred_class == out->mlp.pred_class) &&
        (out->mlp.pred_class == out->rf.pred_class);
#endif
    out->final_class = argmax_5(out->ensemble_probs, &out->final_confidence_pct);
    out->peak_g = out->features[1];
    out->tilt_deg = out->features[6] * 180.0f;
    out->pulse_width_ratio = out->features[3];

    out->should_suppress_alarm =
        (out->final_class == LUMO_AI_CLASS_NORMAL_ADL ||
         out->final_class == LUMO_AI_CLASS_WRIST_SHOCK_FALSE ||
         out->final_class == LUMO_AI_CLASS_DEVICE_DROPPED);
    out->should_trigger_voice_check =
        (out->final_class == LUMO_AI_CLASS_SOFT_FALL_RECOVER ||
         out->final_class == LUMO_AI_CLASS_CRITICAL_FALL);

    char model_tag[48];
    format_active_tag(out, model_tag, sizeof(model_tag));

    switch (out->final_class) {
    case LUMO_AI_CLASS_NORMAL_ADL:
        snprintf(out->commentary, sizeof(out->commentary),
                 "[AI] Binh thuong: %.2fg goc %.0f deg (%s)",
                 (double)out->peak_g, (double)out->tilt_deg, model_tag);
        break;
    case LUMO_AI_CLASS_WRIST_SHOCK_FALSE:
        snprintf(out->commentary, sizeof(out->commentary),
                 "[AI] Chan bao gia: Chan dong co tay %.2fg goc %.0f deg -> Go ban/vo tay (%s)",
                 (double)out->peak_g, (double)out->tilt_deg, model_tag);
        break;
    case LUMO_AI_CLASS_DEVICE_DROPPED:
        snprintf(out->commentary, sizeof(out->commentary),
                 "[AI] Chan bao gia: Vong tay roi/thao ra %.2fg, 0 rung dong sinh hoc (%s)",
                 (double)out->peak_g, model_tag);
        break;
    case LUMO_AI_CLASS_SOFT_FALL_RECOVER:
        snprintf(out->commentary, sizeof(out->commentary),
                 "[AI] Canh bao vang: Truot nga %.2fg (nghieng %.0f deg) dang hoi phuc -> Hoi tham (%s)",
                 (double)out->peak_g, (double)out->tilt_deg, model_tag);
        break;
    case LUMO_AI_CLASS_CRITICAL_FALL:
        snprintf(out->commentary, sizeof(out->commentary),
                 "[AI] KHAN CAP (%u%%): Te nga nang %.2fg, do %.0f deg & bat dong (%s)",
                 out->final_confidence_pct, (double)out->peak_g, (double)out->tilt_deg, model_tag);
        break;
    default:
        snprintf(out->commentary, sizeof(out->commentary), "[AI] Unknown class");
        break;
    }
}

static void fill_standing_baseline(float win[LUMO_AI_WINDOW_LEN][LUMO_AI_CHANNELS],
                                   float noise_a, float noise_g)
{
    for (int t = 0; t < LUMO_AI_WINDOW_LEN; t++) {
        float jitter = (float)((t % 3) - 1) * noise_a;
        float gj = ((t % 2 == 0) ? 1.0f : -1.0f) * noise_g;
        win[t][0] = jitter * 0.5f;
        win[t][1] = jitter * 0.5f;
        win[t][2] = 1.0f + jitter;
        win[t][3] = 1.2f + gj;
        win[t][4] = -0.8f + gj * 0.6f;
        win[t][5] = 0.4f - gj * 0.5f;
    }
}

void lumo_ai_run_self_test(void)
{
    static const char *labels[5] = {
        "1) Normal Walking 1.4Hz",
        "2) Wrist Slap/Clap 5.2g",
        "3) Dropped Band Desk 5.8g",
        "4) Soft Fall Recover 4.4g",
        "5) Critical Hard Fall 5.6g",
    };
    static float win[LUMO_AI_WINDOW_LEN][LUMO_AI_CHANNELS];
    static lumo_ai_dual_result_t res;
    char ai_metrics[160];

    printf("\n=== LUMOHUB ON-DEVICE AI BENCHMARK (%s) ===\n", lumo_ai_active_model_name());
    for (int sc = 0; sc < 5; sc++) {
        if (sc == 0) {
            fill_standing_baseline(win, 0.18f, 18.0f);
            for (int t = 0; t < LUMO_AI_WINDOW_LEN; t++) {
                win[t][2] = 1.0f + 0.28f * sinf((float)t * 0.85f);
                win[t][0] = 0.15f * cosf((float)t * 0.85f);
            }
        } else if (sc == 1) {
            fill_standing_baseline(win, 0.02f, 1.5f);
            win[11][0] = 4.9f; win[11][2] = 1.8f;
            win[11][3] = 145.0f; win[11][4] = -95.0f;
            for (int t = 12; t < 30; t++) {
                win[t][0] = 0.21f; win[t][2] = 0.98f;
                win[t][3] = 1.2f + ((t % 2 == 0) ? 1.8f : -1.4f);
                win[t][4] = -0.8f + ((t % 3 == 0) ? 1.5f : -1.2f);
            }
        } else if (sc == 2) {
            fill_standing_baseline(win, 0.02f, 15.0f);
            for (int t = 8; t <= 10; t++) {
                win[t][0] = 0.08f; win[t][1] = -0.10f; win[t][2] = 0.12f;
                win[t][3] = 180.0f; win[t][4] = -140.0f; win[t][5] = 95.0f;
            }
            win[11][0] = 4.2f; win[11][1] = 3.6f; win[11][2] = 1.6f;
            win[11][3] = 210.0f; win[11][4] = -110.0f;
            win[12][0] = 1.45f; win[12][1] = 0.45f; win[12][2] = 0.35f;
            win[12][3] = 45.0f; win[12][4] = -30.0f; win[12][5] = 15.0f;
            for (int t = 13; t < 30; t++) {
                win[t][0] = 0.98f; win[t][1] = 0.12f; win[t][2] = 0.15f;
                win[t][3] = 1.2f + ((t % 2 == 0) ? 0.06f : -0.06f);
                win[t][4] = -0.8f + ((t % 2 == 0) ? -0.05f : 0.05f);
                win[t][5] = 0.4f;
            }
        } else if (sc == 3) {
            fill_standing_baseline(win, 0.03f, 5.0f);
            for (int t = 9; t <= 10; t++) {
                win[t][0] = 0.25f; win[t][2] = 0.40f;
                win[t][3] = 110.0f; win[t][4] = 75.0f;
            }
            win[11][0] = 3.6f; win[11][2] = 2.4f; win[11][3] = 155.0f;
            win[12][0] = 2.4f; win[12][2] = 1.3f; win[12][3] = 90.0f;
            for (int t = 13; t < 30; t++) {
                float rec = (t >= 17) ? 1.0f : 0.0f;
                win[t][0] = 0.85f - 0.35f * rec * sinf((float)(t - 17) * 0.5f);
                win[t][2] = 0.50f + 0.45f * rec * cosf((float)(t - 17) * 0.5f);
                win[t][3] = 1.2f + rec * 65.0f * sinf((float)(t - 17) * 0.8f);
                win[t][4] = -0.8f + rec * 45.0f * cosf((float)(t - 17) * 0.8f);
            }
        } else {
            fill_standing_baseline(win, 0.03f, 4.0f);
            for (int t = 8; t <= 10; t++) {
                win[t][0] = 0.18f; win[t][2] = 0.28f;
                win[t][3] = 140.0f; win[t][4] = 95.0f;
            }
            win[11][0] = 4.5f; win[11][1] = 1.8f; win[11][2] = 2.7f; win[11][3] = 175.0f;
            win[12][0] = 2.9f; win[12][1] = 1.2f; win[12][2] = 1.6f; win[12][3] = 105.0f;
            for (int t = 14; t < 30; t++) {
                win[t][0] = 0.99f + 0.015f * sinf((float)t * 0.4f);
                win[t][1] = 0.08f;
                win[t][2] = 0.09f;
                win[t][3] = 1.2f + ((t % 2 == 0) ? 2.1f : -1.9f);
                win[t][4] = -0.8f + ((t % 3 == 0) ? 1.7f : -1.5f);
                win[t][5] = 0.4f + ((t % 2 == 0) ? 0.9f : -0.8f);
            }
        }

        lumo_ai_evaluate_window(win, &res);
        lumo_ai_format_metrics(&res, ai_metrics, sizeof(ai_metrics));
        printf("  [%s]\n     %s\n     %s\n", labels[sc], ai_metrics, res.commentary);
    }
    printf("=======================================================================\n\n");
}

bool lumo_ai_evaluate_ring(lumo_ai_dual_result_t *out)
{
    if (s_ring_count < LUMO_AI_WINDOW_LEN || out == NULL) {
        return false;
    }

    static float chronological[LUMO_AI_WINDOW_LEN][LUMO_AI_CHANNELS];
    for (int t = 0; t < LUMO_AI_WINDOW_LEN; t++) {
        int idx = (int)((s_ring_head + t) % LUMO_AI_WINDOW_LEN);
        for (int c = 0; c < LUMO_AI_CHANNELS; c++) {
            chronological[t][c] = s_ring[idx][c];
        }
    }

    lumo_ai_evaluate_window(chronological, out);
    return true;
}

bool lumo_ai_ring_has_recent_peak(float threshold_g)
{
    if (s_ring_count < LUMO_AI_WINDOW_LEN) {
        return false;
    }
    /* Check if impact peak sits inside the trained impact window [8..15] */
    for (int t = 8; t <= 15; t++) {
        int idx = (int)((s_ring_head + t) % LUMO_AI_WINDOW_LEN);
        float ax = s_ring[idx][0], ay = s_ring[idx][1], az = s_ring[idx][2];
        if (sqrtf(ax * ax + ay * ay + az * az) >= threshold_g) {
            return true;
        }
    }
    return false;
}
