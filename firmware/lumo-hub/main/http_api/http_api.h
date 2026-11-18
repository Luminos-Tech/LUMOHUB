#ifndef HTTP_API_H
#define HTTP_API_H

#include <stdbool.h>
#include "esp_err.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#ifdef __cplusplus
extern "C"
{
#endif

    typedef struct
    {
        const char *base_url;       // Ví dụ: https://lumo.vanha2301.online/ota
        const char *assistant_name; // Ví dụ: LUMO
        int id_lumo;                // Ví dụ: 1
        int timeout_ms;             // Ví dụ: 10000
    } http_api_config_t;

    typedef struct
    {
        char id[65];
        char lease[65];
        int volume_percent;
        int audio_bytes;
        int sample_rate;
    } http_tts_job_t;

    /**
     * @brief Gọi server với text truyền vào, parse JSON và trả về field "textRes" (deprecated)
     * @deprecated Không còn được gọi từ firmware. Thay bằng http_api_upload_audio_get_audio.
     */
    esp_err_t http_api_get_text_response(const http_api_config_t *config,
                                         const char *input_text,
                                         char **out_text)
        __attribute__((deprecated));

    /**
     * @brief Upload file WAV lên server, nhận lại text (deprecated — dùng http_api_upload_audio_get_audio)
     * @deprecated Chỉ dùng trong quá trình phát triển. Không còn được gọi từ firmware.
     */
    esp_err_t http_api_upload_audio_get_text(const char *server_url,
                                             const char *file_path,
                                             char **out_text)
        __attribute__((deprecated));

    /**
     * @brief Upload file WAV lên server và stream trực tiếp PCM/WAV phản hồi vào RAM StreamBuffer.
     *        Không ghi file phản hồi xuống SPIFFS (loại bỏ bottleneck ghi Flash).
     *
     * @param server_url     URL endpoint upload (vd: https://api.luminostech.tech/audio/)
     * @param upload_path    Đường dẫn file WAV ghi âm trên SPIFFS (vd: /spiffs/record.wav)
     * @param out_audio_path Tham số giữ tương thích (có thể truyền NULL, không ghi ra file)
     *
     * @return ESP_OK nếu tải và phát I2S hoàn tất
     */
    esp_err_t http_api_upload_audio_get_audio(const char *server_url,
                                              const char *upload_path,
                                              const char *out_audio_path);

    /**
     * @brief Tải stream raw PCM16 mono 24kHz (hoặc WAV) từ URL vào RAM StreamBuffer và phát I2S song song.
     */
    esp_err_t http_api_stream_pcm_url(const char *url,
                                      unsigned volume_percent,
                                      int expected_bytes);

    /**
     * @brief Hỏi backend có job TTS đang xếp hàng cho device_code hay không (HTTP 200 -> true, 204 -> false).
     */
    bool http_api_poll_next_tts_job(const char *base_url,
                                    const char *device_code,
                                    http_tts_job_t *job);

    /**
     * @brief Gửi kết quả phát TTS job (played/failed) về backend.
     */
    void http_api_ack_tts_job(const char *base_url,
                              const char *device_code,
                              const http_tts_job_t *job,
                              bool played);

    /** Send one push notification. Returns an error for transport or HTTP failures. */
    esp_err_t http_api_send_push_notification(int user_id, const char *title,
                                              const char *body, const char *tag,
                                              const char *notification_type);

    /**
     * @brief Gửi push notification lên server.
     *        Gọi 3 lần liên tục để đảm bảo client nhận được.
     *
     * @param user_id ID của user nhận thông báo
     * @param title   Tiêu đề thông báo
     * @param body    Nội dung thông báo
     * @param tag     Tag để phân loại notification (vd: "lumohub-admin")
     * @param notification_type Loại thông báo: "normal" hoặc "alert"
     */
    esp_err_t http_api_send_push_alert(int user_id, const char *title,
                                        const char *body, const char *tag,
                                        const char *notification_type);

#ifdef __cplusplus
}
#endif

#endif // HTTP_API_H
