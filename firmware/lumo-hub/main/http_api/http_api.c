#include "http_api.h"
#include "audio/audio_player.h"
#include "device_control/device_control.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdbool.h>
#include <unistd.h>

#include "esp_log.h"
#include "esp_http_client.h"
#include "esp_timer.h"
#include "cJSON.h"
#include "esp_crt_bundle.h"

static const char *TAG = "HTTP_API";
static const size_t MAX_HTTP_RESPONSE_LEN = 2880000; // 60s PCM16 mono 24 kHz
static const int TTS_DOWNLOAD_CHUNK_BYTES = 4096;
static const int TTS_PLAYBACK_DEADLINE_MS = 120000;

/* Kết nối TLS tái sử dụng theo từng task (connection_task giữ s_poll_client, sound_task giữ s_stream_client) */
static esp_http_client_handle_t s_poll_client = NULL;
static esp_http_client_handle_t s_stream_client = NULL;

typedef struct
{
    char *buffer;
    size_t length;
    size_t capacity;
} http_response_buffer_t;

/* Phát hiện header WAV ('RIFF....WAVE') ở đầu stream và bỏ qua để lấy raw PCM đẩy vào RAM StreamBuffer */
typedef struct
{
    bool decided;
    bool is_wav;
    bool data_found;
    uint32_t sample_rate;
    uint8_t probe[256];
    size_t probe_len;
    uint32_t skip_remaining;
    bool stream_started;
    unsigned volume_percent;
} pcm_stream_adapter_t;

static uint32_t read_u32_le(const uint8_t *p)
{
    return (uint32_t)p[0] |
           ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

static esp_err_t pcm_adapter_push(pcm_stream_adapter_t *ad, const uint8_t *data, size_t len)
{
    if (len == 0)
    {
        return ESP_OK;
    }

    if (!ad->decided)
    {
        size_t space = sizeof(ad->probe) - ad->probe_len;
        size_t copy = (len < space) ? len : space;
        memcpy(ad->probe + ad->probe_len, data, copy);
        ad->probe_len += copy;
        data += copy;
        len -= copy;

        if (ad->probe_len < 12)
        {
            return ESP_OK;
        }

        if (memcmp(ad->probe, "RIFF", 4) == 0 && memcmp(ad->probe + 8, "WAVE", 4) == 0)
        {
            ad->is_wav = true;
            size_t pos = 12;
            uint32_t rate = 24000;
            while (pos + 8 <= ad->probe_len)
            {
                uint32_t chunk_sz = read_u32_le(ad->probe + pos + 4);
                if (memcmp(ad->probe + pos, "fmt ", 4) == 0 && pos + 8 + 8 <= ad->probe_len && chunk_sz >= 16)
                {
                    rate = read_u32_le(ad->probe + pos + 8 + 4);
                    if (rate == 0)
                    {
                        rate = 24000;
                    }
                }
                else if (memcmp(ad->probe + pos, "data", 4) == 0)
                {
                    ad->data_found = true;
                    ad->decided = true;
                    ad->sample_rate = rate;
                    size_t pcm_offset = pos + 8;
                    esp_err_t err = audio_stream_begin(ad->sample_rate, ad->volume_percent);
                    if (err != ESP_OK)
                    {
                        return err;
                    }
                    ad->stream_started = true;
                    if (pcm_offset < ad->probe_len)
                    {
                        err = audio_stream_push(ad->probe + pcm_offset, ad->probe_len - pcm_offset);
                        if (err != ESP_OK)
                        {
                            return err;
                        }
                    }
                    break;
                }
                size_t next_pos = pos + 8 + chunk_sz + (chunk_sz & 1U);
                if (next_pos > ad->probe_len)
                {
                    break;
                }
                pos = next_pos;
            }
            if (!ad->decided && ad->probe_len < sizeof(ad->probe))
            {
                return ESP_OK;
            }
            if (!ad->decided)
            {
                /* Fallback nếu header WAV quá dài */
                ad->decided = true;
                ad->sample_rate = 24000;
                esp_err_t err = audio_stream_begin(ad->sample_rate, ad->volume_percent);
                if (err != ESP_OK)
                {
                    return err;
                }
                ad->stream_started = true;
                if (ad->probe_len > 44)
                {
                    err = audio_stream_push(ad->probe + 44, ad->probe_len - 44);
                    if (err != ESP_OK)
                    {
                        return err;
                    }
                }
            }
        }
        else
        {
            /* Raw PCM16 mono 24 kHz trực tiếp */
            ad->decided = true;
            ad->is_wav = false;
            ad->sample_rate = 24000;
            esp_err_t err = audio_stream_begin(ad->sample_rate, ad->volume_percent);
            if (err != ESP_OK)
            {
                return err;
            }
            ad->stream_started = true;
            err = audio_stream_push(ad->probe, ad->probe_len);
            if (err != ESP_OK)
            {
                return err;
            }
        }
    }

    if (len > 0 && ad->stream_started)
    {
        return audio_stream_push(data, len);
    }
    return ESP_OK;
}

static esp_err_t pcm_adapter_flush_short(pcm_stream_adapter_t *ad)
{
    if (!ad->decided && ad->probe_len > 0)
    {
        ad->decided = true;
        ad->sample_rate = 24000;
        esp_err_t err = audio_stream_begin(ad->sample_rate, ad->volume_percent);
        if (err != ESP_OK)
        {
            return err;
        }
        ad->stream_started = true;
        return audio_stream_push(ad->probe, ad->probe_len);
    }
    return ESP_OK;
}

static void discard_client(esp_http_client_handle_t *holder)
{
    if (holder && *holder)
    {
        esp_http_client_cleanup(*holder);
        *holder = NULL;
    }
}

static void drain_and_close_client(esp_http_client_handle_t *holder)
{
    if (!holder || !*holder)
        return;
    esp_http_client_handle_t client = *holder;
    char remaining[256];
    int drained = 0;
    while (!esp_http_client_is_complete_data_received(client) && drained < 1536)
    {
        int received = esp_http_client_read(client, remaining, sizeof(remaining));
        if (received <= 0)
            break;
        drained += received;
    }
    if (!esp_http_client_is_complete_data_received(client) ||
        !esp_http_client_is_persistent_connection(client))
    {
        discard_client(holder);
    }
    else
    {
        esp_http_client_clear_response_buffer(client);
    }
}

static esp_http_client_handle_t open_persistent_get(
    esp_http_client_handle_t *holder,
    const char *url,
    int timeout_ms,
    const char *accept_header)
{
    esp_http_client_config_t cfg = {
        .url = url,
        .method = HTTP_METHOD_GET,
        .timeout_ms = timeout_ms,
        .crt_bundle_attach = esp_crt_bundle_attach,
        .disable_auto_redirect = true,
        .buffer_size = TTS_DOWNLOAD_CHUNK_BYTES,
        .buffer_size_tx = 1024,
        .keep_alive_enable = true,
        .user_agent = "ESP32/LUMO-Hub",
    };

    if (!*holder)
    {
        *holder = esp_http_client_init(&cfg);
    }
    else if (esp_http_client_set_url(*holder, url) != ESP_OK)
    {
        discard_client(holder);
        return NULL;
    }

    esp_http_client_handle_t client = *holder;
    if (!client)
    {
        return NULL;
    }

    esp_http_client_set_method(client, HTTP_METHOD_GET);
    if (accept_header)
    {
        esp_http_client_set_header(client, "Accept", accept_header);
    }

    esp_err_t opened = esp_http_client_open(client, 0);
    if (opened != ESP_OK)
    {
        ESP_LOGW(TAG, "HTTP GET open failed: %s", esp_err_to_name(opened));
        discard_client(holder);
        return NULL;
    }
    if (esp_http_client_fetch_headers(client) < 0)
    {
        ESP_LOGW(TAG, "HTTP GET fetch_headers failed");
        discard_client(holder);
        return NULL;
    }
    return client;
}

static esp_err_t http_event_handler(esp_http_client_event_t *evt)
{
    http_response_buffer_t *resp = (http_response_buffer_t *)evt->user_data;

    if (evt->event_id == HTTP_EVENT_ON_DATA && evt->data_len > 0 && resp != NULL)
    {
        if (resp->length + evt->data_len > MAX_HTTP_RESPONSE_LEN)
        {
            ESP_LOGW(TAG, "HTTP response body exceeds %u bytes, truncating",
                     (unsigned)MAX_HTTP_RESPONSE_LEN);
            return ESP_OK;
        }

        if (resp->buffer == NULL)
        {
            resp->capacity = 4096;
            resp->buffer = malloc(resp->capacity);
            if (resp->buffer == NULL)
            {
                ESP_LOGE(TAG, "malloc failed for HTTP response buffer");
                return ESP_ERR_NO_MEM;
            }
        }

        size_t needed = resp->length + evt->data_len + 1;
        if (needed > resp->capacity)
        {
            while (resp->capacity < needed)
            {
                resp->capacity *= 2;
            }
            char *new_buf = realloc(resp->buffer, resp->capacity);
            if (new_buf == NULL)
            {
                ESP_LOGE(TAG, "realloc failed for HTTP response buffer");
                return ESP_OK;
            }
            resp->buffer = new_buf;
        }

        memcpy(resp->buffer + resp->length, evt->data, evt->data_len);
        resp->length += evt->data_len;
        resp->buffer[resp->length] = '\0';
    }

    return ESP_OK;
}

static int is_unreserved_char(char c)
{
    return (isalnum((unsigned char)c) || c == '-' || c == '_' || c == '.' || c == '~');
}

static char *url_encode(const char *src)
{
    if (src == NULL)
    {
        return NULL;
    }

    size_t len = strlen(src);
    char *encoded = malloc(len * 3 + 1);
    if (encoded == NULL)
    {
        return NULL;
    }

    char *p = encoded;
    for (size_t i = 0; i < len; i++)
    {
        unsigned char ch = (unsigned char)src[i];
        if (is_unreserved_char((char)ch))
        {
            *p++ = (char)ch;
        }
        else
        {
            sprintf(p, "%%%02X", ch);
            p += 3;
        }
    }
    *p = '\0';
    return encoded;
}

// ============================================================
// GET text response (giữ nguyên để tương thích)
// ============================================================
esp_err_t http_api_get_text_response(const http_api_config_t *config,
                                     const char *input_text,
                                     char **out_text)
{
    esp_err_t err = ESP_FAIL;
    esp_http_client_handle_t client = NULL;
    char *encoded_text = NULL;
    char *encoded_assistant = NULL;
    char *url = NULL;
    cJSON *root = NULL;

    http_response_buffer_t resp = {.buffer = NULL, .length = 0};

    if (config == NULL || config->base_url == NULL || input_text == NULL || out_text == NULL)
    {
        return ESP_ERR_INVALID_ARG;
    }

    *out_text = NULL;

    encoded_text = url_encode(input_text);
    encoded_assistant = url_encode(config->assistant_name ? config->assistant_name : "");
    if (encoded_text == NULL || encoded_assistant == NULL)
    {
        err = ESP_ERR_NO_MEM;
        goto cleanup;
    }

    size_t url_len = strlen(config->base_url) + strlen(encoded_text) + strlen(encoded_assistant) + 128;
    url = malloc(url_len);
    if (url == NULL)
    {
        err = ESP_ERR_NO_MEM;
        goto cleanup;
    }

    snprintf(url, url_len,
             "%s?idLumo=%d&textLumoCallServer=%s&assistant_name=%s",
             config->base_url,
             config->id_lumo,
             encoded_text,
             encoded_assistant);

    ESP_LOGI(TAG, "Request URL: %s", url);

    esp_http_client_config_t client_config = {
        .url = url,
        .method = HTTP_METHOD_GET,
        .event_handler = http_event_handler,
        .user_data = &resp,
        .timeout_ms = (config->timeout_ms > 0) ? config->timeout_ms : 10000,
        .crt_bundle_attach = esp_crt_bundle_attach,
    };

    client = esp_http_client_init(&client_config);
    if (client == NULL)
    {
        ESP_LOGE(TAG, "esp_http_client_init failed");
        err = ESP_FAIL;
        goto cleanup;
    }

    esp_http_client_set_header(client, "Accept", "application/json");

    err = esp_http_client_perform(client);
    if (err != ESP_OK)
    {
        ESP_LOGE(TAG, "HTTP request failed: %s", esp_err_to_name(err));
        goto cleanup;
    }

    int status_code = esp_http_client_get_status_code(client);
    ESP_LOGI(TAG, "HTTP status=%d", status_code);

    if (status_code != 200 || resp.buffer == NULL)
    {
        ESP_LOGE(TAG, "Invalid HTTP response");
        err = ESP_FAIL;
        goto cleanup;
    }

    root = cJSON_Parse(resp.buffer);
    if (root == NULL)
    {
        ESP_LOGE(TAG, "JSON parse failed");
        err = ESP_FAIL;
        goto cleanup;
    }

    cJSON *text_res = cJSON_GetObjectItemCaseSensitive(root, "textRes");
    if (!cJSON_IsString(text_res) || text_res->valuestring == NULL)
    {
        ESP_LOGE(TAG, "JSON field 'textRes' not found");
        err = ESP_FAIL;
        goto cleanup;
    }

    *out_text = strdup(text_res->valuestring);
    if (*out_text == NULL)
    {
        err = ESP_ERR_NO_MEM;
        goto cleanup;
    }

    err = ESP_OK;

cleanup:
    if (root != NULL)
        cJSON_Delete(root);
    if (client != NULL)
        esp_http_client_cleanup(client);
    free(resp.buffer);
    free(encoded_text);
    free(encoded_assistant);
    free(url);

    return err;
}

// ============================================================
// Upload audio → nhận TEXT (giữ lại để tương thích)
// ============================================================
esp_err_t http_api_upload_audio_get_text(const char *server_url,
                                         const char *file_path,
                                         char **out_text)
{
    esp_err_t err = ESP_FAIL;
    esp_http_client_handle_t client = NULL;
    FILE *f = NULL;
    uint8_t *file_buf = NULL;
    uint8_t *body_buf = NULL;
    cJSON *root = NULL;

    http_response_buffer_t resp = {.buffer = NULL, .length = 0};

    f = fopen(file_path, "rb");
    if (!f)
    {
        ESP_LOGE(TAG, "Cannot open file: %s", file_path);
        return ESP_FAIL;
    }

    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    file_buf = malloc(file_size);
    if (!file_buf)
    {
        err = ESP_ERR_NO_MEM;
        goto cleanup;
    }
    fread(file_buf, 1, file_size, f);
    fclose(f);
    f = NULL;

    const char *part_header =
        "------ESP32Boundary\r\n"
        "Content-Disposition: form-data; name=\"audio\"; filename=\"record.wav\"\r\n"
        "Content-Type: audio/wav\r\n\r\n";
    const char *part_footer = "\r\n------ESP32Boundary--\r\n";

    size_t body_len = strlen(part_header) + file_size + strlen(part_footer);
    body_buf = malloc(body_len);
    if (!body_buf)
    {
        err = ESP_ERR_NO_MEM;
        goto cleanup;
    }

    memcpy(body_buf, part_header, strlen(part_header));
    memcpy(body_buf + strlen(part_header), file_buf, file_size);
    memcpy(body_buf + strlen(part_header) + file_size, part_footer, strlen(part_footer));

    esp_http_client_config_t config = {
        .url = server_url,
        .method = HTTP_METHOD_POST,
        .event_handler = http_event_handler,
        .user_data = &resp,
        .timeout_ms = 30000,
        .crt_bundle_attach = esp_crt_bundle_attach,
    };

    client = esp_http_client_init(&config);
    if (!client)
    {
        err = ESP_FAIL;
        goto cleanup;
    }

    esp_http_client_set_header(client, "Content-Type",
                               "multipart/form-data; boundary=----ESP32Boundary");
    esp_http_client_set_post_field(client, (const char *)body_buf, body_len);

    err = esp_http_client_perform(client);
    if (err != ESP_OK)
        goto cleanup;

    int status = esp_http_client_get_status_code(client);
    if (status != 200 || !resp.buffer)
    {
        err = ESP_FAIL;
        goto cleanup;
    }

    root = cJSON_Parse(resp.buffer);
    if (!root)
    {
        err = ESP_FAIL;
        goto cleanup;
    }

    cJSON *text_res = cJSON_GetObjectItemCaseSensitive(root, "textRes");
    if (!cJSON_IsString(text_res))
    {
        err = ESP_FAIL;
        goto cleanup;
    }

    *out_text = strdup(text_res->valuestring);
    err = ESP_OK;

cleanup:
    if (root)
        cJSON_Delete(root);
    if (client)
        esp_http_client_cleanup(client);
    if (f)
        fclose(f);
    free(file_buf);
    free(body_buf);
    free(resp.buffer);
    return err;
}

// ============================================================
// Upload audio WAV → Server STT→TTT→TTS → stream PCM vào RAM StreamBuffer
// ============================================================
esp_err_t http_api_upload_audio_get_audio(const char *server_url,
                                          const char *upload_path,
                                          const char *out_audio_path)
{
    (void)out_audio_path; /* Không ghi xuống SPIFFS nữa */

    esp_err_t err = ESP_FAIL;
    esp_http_client_handle_t client = NULL;
    FILE *in_f = NULL;
    uint8_t *chunk_buf = NULL;
    pcm_stream_adapter_t adapter = {0};
    adapter.volume_percent = 85;

    in_f = fopen(upload_path, "rb");
    if (!in_f)
    {
        ESP_LOGE(TAG, "Cannot open upload file: %s", upload_path);
        return ESP_FAIL;
    }

    fseek(in_f, 0, SEEK_END);
    long file_size = ftell(in_f);
    fseek(in_f, 0, SEEK_SET);

    if (file_size <= 0)
    {
        ESP_LOGE(TAG, "Empty or invalid file: %s", upload_path);
        fclose(in_f);
        return ESP_FAIL;
    }

    ESP_LOGI(TAG, "Uploading %ld bytes from %s (RAM stream mode)", file_size, upload_path);

    esp_http_client_config_t config = {
        .url = server_url,
        .method = HTTP_METHOD_POST,
        .timeout_ms = 60000,
        .crt_bundle_attach = esp_crt_bundle_attach,
        .buffer_size_tx = 4096,
        .buffer_size = TTS_DOWNLOAD_CHUNK_BYTES,
    };

    client = esp_http_client_init(&config);
    if (!client)
    {
        ESP_LOGE(TAG, "esp_http_client_init failed");
        fclose(in_f);
        return ESP_FAIL;
    }

    const char *part_header =
        "------ESP32Boundary\r\n"
        "Content-Disposition: form-data; name=\"audio\"; filename=\"record.wav\"\r\n"
        "Content-Type: audio/wav\r\n\r\n";
    const char *part_footer = "\r\n------ESP32Boundary--\r\n";

    size_t header_len = strlen(part_header);
    size_t footer_len = strlen(part_footer);
    int body_len = (int)(header_len + (size_t)file_size + footer_len);

    esp_http_client_set_header(client, "Content-Type",
                               "multipart/form-data; boundary=----ESP32Boundary");
    esp_http_client_set_header(client, "Accept", "audio/pcm, audio/wav");

    char content_length_str[16];
    snprintf(content_length_str, sizeof(content_length_str), "%d", body_len);
    esp_http_client_set_header(client, "Content-Length", content_length_str);

    err = esp_http_client_open(client, body_len);
    if (err != ESP_OK)
    {
        ESP_LOGE(TAG, "esp_http_client_open failed: %s", esp_err_to_name(err));
        goto cleanup;
    }

    if (esp_http_client_write(client, part_header, (int)header_len) != (int)header_len)
    {
        ESP_LOGE(TAG, "Write multipart header failed");
        err = ESP_FAIL;
        goto cleanup;
    }

    chunk_buf = malloc(TTS_DOWNLOAD_CHUNK_BYTES);
    if (!chunk_buf)
    {
        ESP_LOGE(TAG, "malloc chunk_buf failed");
        err = ESP_ERR_NO_MEM;
        goto cleanup;
    }

    long remaining = file_size;
    while (remaining > 0)
    {
        size_t to_read = (remaining > TTS_DOWNLOAD_CHUNK_BYTES) ? TTS_DOWNLOAD_CHUNK_BYTES : (size_t)remaining;
        if (fread(chunk_buf, 1, to_read, in_f) != to_read)
        {
            ESP_LOGE(TAG, "fread failed at %ld remaining", remaining);
            err = ESP_FAIL;
            goto cleanup;
        }
        int written = esp_http_client_write(client, (const char *)chunk_buf, (int)to_read);
        if (written != (int)to_read)
        {
            ESP_LOGE(TAG, "esp_http_client_write failed: %d/%u", written, (unsigned)to_read);
            err = ESP_FAIL;
            goto cleanup;
        }
        remaining -= (long)to_read;
    }

    if (esp_http_client_write(client, part_footer, (int)footer_len) != (int)footer_len)
    {
        ESP_LOGE(TAG, "Write multipart footer failed");
        err = ESP_FAIL;
        goto cleanup;
    }

    int64_t content_length = esp_http_client_fetch_headers(client);
    if (content_length < 0)
    {
        ESP_LOGE(TAG, "esp_http_client_fetch_headers failed");
        err = ESP_FAIL;
        goto cleanup;
    }

    int status = esp_http_client_get_status_code(client);
    if (status != 200)
    {
        ESP_LOGE(TAG, "Invalid audio response HTTP %d", status);
        err = ESP_FAIL;
        goto cleanup;
    }

    size_t total_received = 0;
    bool download_ok = true;
    int64_t deadline = esp_timer_get_time() + (int64_t)TTS_PLAYBACK_DEADLINE_MS * 1000LL;

    while (true)
    {
        if (esp_timer_get_time() > deadline)
        {
            ESP_LOGE(TAG, "Audio stream deadline exceeded");
            download_ok = false;
            break;
        }
        int received = esp_http_client_read(client, (char *)chunk_buf, TTS_DOWNLOAD_CHUNK_BYTES);
        if (received < 0)
        {
            ESP_LOGE(TAG, "Audio stream read error");
            download_ok = false;
            break;
        }
        if (received == 0)
        {
            break;
        }
        total_received += (size_t)received;
        if (total_received > MAX_HTTP_RESPONSE_LEN)
        {
            ESP_LOGW(TAG, "Audio response exceeded max limit");
            break;
        }
        if (pcm_adapter_push(&adapter, chunk_buf, (size_t)received) != ESP_OK)
        {
            download_ok = false;
            break;
        }
    }

    if (download_ok && !adapter.stream_started)
    {
        if (pcm_adapter_flush_short(&adapter) != ESP_OK)
        {
            download_ok = false;
        }
    }

    if (adapter.stream_started)
    {
        esp_err_t end_err = audio_stream_end(!download_ok || total_received == 0);
        err = (download_ok && total_received > 0 && end_err == ESP_OK) ? ESP_OK : ESP_FAIL;
    }
    else
    {
        err = ESP_FAIL;
    }

cleanup:
    free(chunk_buf);
    if (client)
        esp_http_client_cleanup(client);
    if (in_f)
        fclose(in_f);
    return err;
}

// ============================================================
// Stream PCM/WAV từ URL trực tiếp vào RAM StreamBuffer (DSOFT-AudioTTS pattern)
// ============================================================
esp_err_t http_api_stream_pcm_url(const char *url, unsigned volume_percent, int expected_bytes)
{
    if (url == NULL || !*url)
    {
        return ESP_ERR_INVALID_ARG;
    }

    const int64_t started = esp_timer_get_time();
    ESP_LOGI(TAG, "Streaming TTS audio: expected=%d bytes, url=%s", expected_bytes, url);

    esp_http_client_handle_t client = open_persistent_get(
        &s_stream_client, url, 15000, "application/octet-stream");
    if (!client)
    {
        return ESP_FAIL;
    }

    int status = esp_http_client_get_status_code(client);
    int64_t content_length = esp_http_client_get_content_length(client);
    if (status != 200 ||
        (!esp_http_client_is_chunked_response(client) &&
         content_length >= 0 && expected_bytes > 0 && content_length != expected_bytes))
    {
        ESP_LOGE(TAG, "Stream response rejected: HTTP %d, length=%lld, expected=%d",
                 status, (long long)content_length, expected_bytes);
        drain_and_close_client(&s_stream_client);
        return ESP_FAIL;
    }

    uint8_t *chunk = malloc(TTS_DOWNLOAD_CHUNK_BYTES);
    if (!chunk)
    {
        drain_and_close_client(&s_stream_client);
        return ESP_ERR_NO_MEM;
    }

    pcm_stream_adapter_t adapter = {0};
    adapter.volume_percent = volume_percent;

    size_t downloaded = 0;
    bool download_ok = true;
    const int64_t deadline = started + (int64_t)TTS_PLAYBACK_DEADLINE_MS * 1000LL;

    while (expected_bytes <= 0 || downloaded < (size_t)expected_bytes)
    {
        if (esp_timer_get_time() > deadline)
        {
            ESP_LOGE(TAG, "TTS download deadline exceeded: %u/%d",
                     (unsigned)downloaded, expected_bytes);
            download_ok = false;
            break;
        }
        int received = esp_http_client_read(client, (char *)chunk, TTS_DOWNLOAD_CHUNK_BYTES);
        if (received < 0)
        {
            ESP_LOGE(TAG, "TTS read failed at %u bytes", (unsigned)downloaded);
            download_ok = false;
            break;
        }
        if (received == 0)
        {
            if (expected_bytes > 0 && downloaded < (size_t)expected_bytes)
            {
                ESP_LOGE(TAG, "TTS read stopped early: %u/%d bytes",
                         (unsigned)downloaded, expected_bytes);
                download_ok = false;
            }
            break;
        }
        if (expected_bytes > 0 && downloaded + (size_t)received > (size_t)expected_bytes)
        {
            ESP_LOGE(TAG, "TTS response larger than expected (%u > %d)",
                     (unsigned)(downloaded + (size_t)received), expected_bytes);
            download_ok = false;
            break;
        }
        if (pcm_adapter_push(&adapter, chunk, (size_t)received) != ESP_OK)
        {
            download_ok = false;
            break;
        }
        downloaded += (size_t)received;
    }

    if (download_ok && !esp_http_client_is_complete_data_received(client))
    {
        char extra;
        download_ok = esp_http_client_read(client, &extra, 1) == 0;
    }
    download_ok = download_ok && downloaded > 0 &&
                  (expected_bytes <= 0 || downloaded == (size_t)expected_bytes) &&
                  esp_http_client_is_complete_data_received(client);

    free(chunk);
    drain_and_close_client(&s_stream_client);

    if (download_ok && !adapter.stream_started)
    {
        if (pcm_adapter_flush_short(&adapter) != ESP_OK)
        {
            download_ok = false;
        }
    }

    esp_err_t playback_err = ESP_FAIL;
    if (adapter.stream_started)
    {
        playback_err = audio_stream_end(!download_ok);
    }

    ESP_LOGI(TAG, "TTS stream finished: downloaded=%u/%d, ok=%d, playback=%s, elapsed=%lld ms",
             (unsigned)downloaded, expected_bytes, download_ok,
             esp_err_to_name(playback_err),
             (long long)((esp_timer_get_time() - started) / 1000));

    if (playback_err == ESP_ERR_INVALID_STATE)
    {
        return ESP_ERR_INVALID_STATE;
    }
    return (download_ok && playback_err == ESP_OK) ? ESP_OK : ESP_FAIL;
}

bool http_api_poll_next_tts_job(const char *base_url, const char *device_code, http_tts_job_t *job)
{
    if (!base_url || !*base_url || !device_code || !*device_code || !job)
    {
        return false;
    }
    memset(job, 0, sizeof(*job));

    char url[256];
    snprintf(url, sizeof(url), "%s/api/v1/devices/code/%s/next", base_url, device_code);

    esp_http_client_handle_t client = open_persistent_get(
        &s_poll_client, url, 8000, "application/json");
    if (!client)
    {
        return false;
    }

    int status = esp_http_client_get_status_code(client);
    bool has_job = false;
    if (status == 200)
    {
        char response[1024];
        size_t used = 0;
        while (used < sizeof(response) - 1)
        {
            int received = esp_http_client_read(client, response + used, sizeof(response) - 1 - used);
            if (received <= 0)
                break;
            used += (size_t)received;
        }
        response[used] = '\0';

        if (esp_http_client_is_complete_data_received(client))
        {
            cJSON *json = cJSON_Parse(response);
            if (json)
            {
                const cJSON *id = cJSON_GetObjectItemCaseSensitive(json, "id");
                const cJSON *lease = cJSON_GetObjectItemCaseSensitive(json, "lease");
                const cJSON *volume = cJSON_GetObjectItemCaseSensitive(json, "volume");
                const cJSON *bytes = cJSON_GetObjectItemCaseSensitive(json, "audio_bytes");
                const cJSON *rate = cJSON_GetObjectItemCaseSensitive(json, "sample_rate");
                if (cJSON_IsString(id) && cJSON_IsString(lease) &&
                    cJSON_IsNumber(volume) && cJSON_IsNumber(bytes) && cJSON_IsNumber(rate) &&
                    bytes->valueint > 0 && rate->valueint == 24000)
                {
                    snprintf(job->id, sizeof(job->id), "%s", id->valuestring);
                    snprintf(job->lease, sizeof(job->lease), "%s", lease->valuestring);
                    job->volume_percent = volume->valueint;
                    job->audio_bytes = bytes->valueint;
                    job->sample_rate = rate->valueint;
                    has_job = true;
                }
                cJSON_Delete(json);
            }
        }
    }

    drain_and_close_client(&s_poll_client);
    return has_job;
}

void http_api_ack_tts_job(const char *base_url, const char *device_code, const http_tts_job_t *job, bool played)
{
    if (!base_url || !*base_url || !device_code || !job || !job->id[0] || !job->lease[0])
    {
        return;
    }

    char url[256];
    char body[256];
    snprintf(url, sizeof(url), "%s/api/v1/devices/code/%s/ack", base_url, device_code);
    snprintf(body, sizeof(body), "{\"id\":\"%s\",\"lease\":\"%s\",\"status\":\"%s\"}",
             job->id, job->lease, played ? "played" : "failed");

    for (int attempt = 0; attempt < 3; ++attempt)
    {
        esp_http_client_config_t cfg = {
            .url = url,
            .method = HTTP_METHOD_POST,
            .timeout_ms = 10000,
            .crt_bundle_attach = esp_crt_bundle_attach,
        };
        esp_http_client_handle_t client = esp_http_client_init(&cfg);
        if (!client)
        {
            return;
        }
        esp_http_client_set_header(client, "Content-Type", "application/json");
        esp_http_client_set_post_field(client, body, strlen(body));
        esp_err_t err = esp_http_client_perform(client);
        int status = esp_http_client_get_status_code(client);
        esp_http_client_cleanup(client);

        if (err == ESP_OK && (status == 200 || status == 404 || status == 409))
        {
            ESP_LOGI(TAG, "ACK TTS job %s: %s (HTTP %d)", job->id, played ? "played" : "failed", status);
            return;
        }
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}

// ============================================================
// Send one push notification; band alerts use the three-send wrapper below.
// ============================================================
esp_err_t http_api_send_push_notification(int user_id, const char *title, const char *body, const char *tag, const char *notification_type)
{
    const char *server_url = "https://api.luminostech.tech/api/v1/push/send";

    char post_data[512];
    snprintf(post_data, sizeof(post_data),
             "{\"user_id\":%d,\"title\":\"%s\",\"body\":\"%s\",\"tag\":\"%s\",\"notification_type\":\"%s\"}",
             user_id, title, body, tag, notification_type);

    ESP_LOGI(TAG, "Sending push notification: %s", post_data);

    esp_http_client_config_t config = {
        .url = server_url,
        .method = HTTP_METHOD_POST,
        .timeout_ms = 10000,
        .crt_bundle_attach = esp_crt_bundle_attach,
    };

    esp_http_client_handle_t client = esp_http_client_init(&config);
    if (!client)
    {
        ESP_LOGE(TAG, "esp_http_client_init failed");
        return ESP_ERR_NO_MEM;
    }

    esp_http_client_set_header(client, "Content-Type", "application/json");
    esp_http_client_set_post_field(client, post_data, strlen(post_data));

    esp_err_t err = esp_http_client_perform(client);
    int status = esp_http_client_get_status_code(client);

    if (err == ESP_OK && status >= 200 && status < 300)
    {
        ESP_LOGI(TAG, "Push sent successfully");
        device_control_log('I', "HTTP", "Notification API request succeeded");
    }
    else
    {
        ESP_LOGW(TAG, "Push failed: err=%s, status=%d",
                 esp_err_to_name(err), status);
        device_control_log('E', "HTTP", "Notification API request failed");
        if (err == ESP_OK)
        {
            err = ESP_FAIL;
        }
    }

    esp_http_client_cleanup(client);
    return err;
}

esp_err_t http_api_send_push_alert(int user_id, const char *title, const char *body, const char *tag, const char *notification_type)
{
    /* Preserve the existing three-send behavior for band alerts. */
    for (int attempt = 1; attempt <= 3; attempt++)
    {
        http_api_send_push_notification(user_id, title, body, tag, notification_type);
        vTaskDelay(pdMS_TO_TICKS(200));
    }

    return ESP_OK;
}
