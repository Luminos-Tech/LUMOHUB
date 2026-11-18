#include "audio_player.h"

#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "freertos/stream_buffer.h"
#include "freertos/event_groups.h"
#include "driver/i2s_std.h"
#include "esp_log.h"
#include "esp_err.h"
#include "esp_system.h"
#include "esp_timer.h"

static const char *TAG = "AUDIO";
static volatile bool s_stop_requested = false;
static volatile bool s_is_playing = false;
static SemaphoreHandle_t s_playback_mutex = NULL;

static i2s_chan_handle_t s_tx_handle = NULL;
static int s_bclk_gpio = -1;
static int s_lrck_gpio = -1;
static int s_dout_gpio = -1;
static bool s_i2s_ready = false;
static bool s_i2s_enabled = false;
static uint32_t s_current_sample_rate = 0;

/** Minimum I2S clock stability settle time (ms) after reinit */
#define I2S_SETTLE_MS 5

/* Tham số RAM StreamBuffer tham chiếu từ DSOFT-AudioTTS (speaker_config.h) */
#define AUDIO_STREAM_BUFFER_BYTES         (32 * 1024)
#define AUDIO_STREAM_PREBUFFER_BYTES      (12 * 1024)
#define AUDIO_STREAM_CHUNK_BYTES          1024
#define AUDIO_DATA_WAIT_SLICE_MS          100
#define AUDIO_DATA_MAX_STALL_MS           3000
#define AUDIO_BUFFER_WRITE_TIMEOUT_MS     1000
#define AUDIO_PLAYBACK_FINISH_TIMEOUT_MS  5000
#define AUDIO_I2S_WRITE_TIMEOUT_MS        1000
#define AUDIO_SILENCE_TAIL_BLOCKS         8
#define AUDIO_STREAM_TASK_STACK_BYTES     6144
#define AUDIO_STREAM_TASK_PRIORITY        6

#define STREAM_PLAY_REQUESTED    BIT0
#define STREAM_DOWNLOAD_FINISHED BIT1
#define STREAM_PLAYBACK_FINISHED BIT2
#define STREAM_PLAYBACK_FAILED   BIT3

static StreamBufferHandle_t s_pcm_buffer = NULL;
static EventGroupHandle_t s_stream_events = NULL;
static TaskHandle_t s_stream_task_handle = NULL;
static unsigned s_volume_percent = 100;
static bool s_playback_requested = false;
static bool s_stream_mutex_held = false;

typedef struct
{
    uint16_t audio_format; // PCM = 1
    uint16_t num_channels; // 1 mono, 2 stereo
    uint32_t sample_rate;
    uint16_t bits_per_sample;
    uint32_t data_offset;
    uint32_t data_size; // even-rounded
} wav_info_t;

static uint16_t read_le16(const uint8_t *p)
{
    return (uint16_t)(p[0] | (p[1] << 8));
}

static uint32_t read_le32(const uint8_t *p)
{
    return (uint32_t)(p[0] |
                      (p[1] << 8) |
                      (p[2] << 16) |
                      (p[3] << 24));
}

static int16_t scale_pcm_sample(int16_t sample, unsigned volume)
{
    return (int16_t)(((int32_t)sample * (int32_t)volume) / 100);
}

static esp_err_t i2s_ensure_enabled(bool enable)
{
    if (!s_i2s_ready || s_tx_handle == NULL)
    {
        return ESP_ERR_INVALID_STATE;
    }
    if (enable && !s_i2s_enabled)
    {
        esp_err_t ret = i2s_channel_enable(s_tx_handle);
        if (ret != ESP_OK)
        {
            ESP_LOGE(TAG, "i2s_channel_enable failed: %s", esp_err_to_name(ret));
            return ret;
        }
        s_i2s_enabled = true;
        vTaskDelay(pdMS_TO_TICKS(I2S_SETTLE_MS));
    }
    else if (!enable && s_i2s_enabled)
    {
        esp_err_t ret = i2s_channel_disable(s_tx_handle);
        if (ret != ESP_OK)
        {
            ESP_LOGW(TAG, "i2s_channel_disable failed: %s", esp_err_to_name(ret));
            return ret;
        }
        s_i2s_enabled = false;
    }
    return ESP_OK;
}

static esp_err_t audio_prepare_i2s(uint32_t sample_rate, bool enable_now)
{
    esp_err_t ret;

    if (s_i2s_ready && s_tx_handle != NULL && s_current_sample_rate == sample_rate)
    {
        return i2s_ensure_enabled(enable_now);
    }

    if (s_i2s_ready && s_tx_handle != NULL)
    {
        if (s_i2s_enabled)
        {
            ret = i2s_channel_disable(s_tx_handle);
            if (ret != ESP_OK)
            {
                ESP_LOGE(TAG, "i2s_channel_disable failed: %s", esp_err_to_name(ret));
                return ret;
            }
            s_i2s_enabled = false;
        }

        ret = i2s_del_channel(s_tx_handle);
        if (ret != ESP_OK)
        {
            ESP_LOGE(TAG, "i2s_del_channel failed: %s", esp_err_to_name(ret));
            return ret;
        }

        s_tx_handle = NULL;
        s_i2s_ready = false;
        s_current_sample_rate = 0;
    }

    i2s_chan_config_t chan_cfg =
        I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_1, I2S_ROLE_MASTER);
    chan_cfg.dma_desc_num = 6;
    chan_cfg.dma_frame_num = 240;
    chan_cfg.auto_clear = true;

    ret = i2s_new_channel(&chan_cfg, &s_tx_handle, NULL);
    if (ret != ESP_OK)
    {
        ESP_LOGE(TAG, "i2s_new_channel failed: %s", esp_err_to_name(ret));
        return ret;
    }

    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(sample_rate),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(
            I2S_DATA_BIT_WIDTH_16BIT,
            I2S_SLOT_MODE_STEREO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = s_bclk_gpio,
            .ws = s_lrck_gpio,
            .dout = s_dout_gpio,
            .din = I2S_GPIO_UNUSED,
            .invert_flags = {
                .mclk_inv = false,
                .bclk_inv = false,
                .ws_inv = false,
            },
        },
    };

    ret = i2s_channel_init_std_mode(s_tx_handle, &std_cfg);
    if (ret != ESP_OK)
    {
        ESP_LOGE(TAG, "i2s_channel_init_std_mode failed: %s",
                 esp_err_to_name(ret));
        i2s_del_channel(s_tx_handle);
        s_tx_handle = NULL;
        return ret;
    }

    s_i2s_ready = true;
    s_i2s_enabled = false;
    s_current_sample_rate = sample_rate;

    if (enable_now)
    {
        ret = i2s_ensure_enabled(true);
        if (ret != ESP_OK)
        {
            i2s_del_channel(s_tx_handle);
            s_tx_handle = NULL;
            s_i2s_ready = false;
            s_current_sample_rate = 0;
            return ret;
        }
    }

    ESP_LOGI(TAG, "I2S ready, sample_rate=%lu", (unsigned long)sample_rate);
    return ESP_OK;
}

static bool write_to_i2s_bytes(const void *data, size_t size)
{
    const uint8_t *bytes = (const uint8_t *)data;
    size_t sent = 0;
    while (sent < size)
    {
        size_t written = 0;
        esp_err_t result = i2s_channel_write(
            s_tx_handle,
            bytes + sent,
            size - sent,
            &written,
            AUDIO_I2S_WRITE_TIMEOUT_MS);
        if (result != ESP_OK || written == 0)
        {
            ESP_LOGE(TAG, "I2S write failed: %s, written=%u",
                     esp_err_to_name(result), (unsigned)written);
            return false;
        }
        sent += written;
    }
    return true;
}

static bool flush_audio_tail(void)
{
    static const int16_t silence[AUDIO_STREAM_CHUNK_BYTES] = {0};
    for (int block = 0; block < AUDIO_SILENCE_TAIL_BLOCKS; ++block)
    {
        if (s_stop_requested)
        {
            return true;
        }
        if (!write_to_i2s_bytes(silence, sizeof(silence)))
        {
            return false;
        }
    }
    return true;
}

static bool play_buffered_audio(void)
{
    /* s_playback_mutex bảo vệ độc quyền truy cập các buffer tĩnh này */
    static uint8_t input[AUDIO_STREAM_CHUNK_BYTES];
    static int16_t stereo[AUDIO_STREAM_CHUNK_BYTES];

    bool has_first_byte = false;
    uint8_t first_byte = 0;
    size_t consumed = 0;
    const int64_t started = esp_timer_get_time();
    int64_t last_progress = started;
    int stall_ms = 0;

    ESP_LOGI(TAG, "Stream playback start: buffer=%u/%u bytes",
             (unsigned)xStreamBufferBytesAvailable(s_pcm_buffer),
             (unsigned)AUDIO_STREAM_BUFFER_BYTES);

    while (true)
    {
        EventBits_t bits = xEventGroupGetBits(s_stream_events);
        if (s_stop_requested || (bits & STREAM_PLAYBACK_FAILED))
        {
            return false;
        }

        size_t received = xStreamBufferReceive(
            s_pcm_buffer,
            input,
            sizeof(input),
            pdMS_TO_TICKS(AUDIO_DATA_WAIT_SLICE_MS));

        if (received == 0)
        {
            bits = xEventGroupGetBits(s_stream_events);
            if (bits & STREAM_DOWNLOAD_FINISHED)
            {
                ESP_LOGI(TAG, "Stream playback drained: pcm=%u bytes, elapsed=%lld ms",
                         (unsigned)consumed, (long long)((esp_timer_get_time() - started) / 1000));
                return !has_first_byte;
            }
            if (s_stop_requested || (bits & STREAM_PLAYBACK_FAILED))
            {
                return false;
            }
            stall_ms += AUDIO_DATA_WAIT_SLICE_MS;
            if (stall_ms >= AUDIO_DATA_MAX_STALL_MS)
            {
                ESP_LOGE(TAG, "Stream buffer underrun: pcm=%u bytes, stall=%d ms",
                         (unsigned)consumed, stall_ms);
                return false;
            }
            continue;
        }

        stall_ms = 0;
        size_t output_samples = 0;
        for (size_t index = 0; index < received; ++index)
        {
            if (!has_first_byte)
            {
                first_byte = input[index];
                has_first_byte = true;
                continue;
            }

            int16_t sample = (int16_t)(
                (uint16_t)first_byte | ((uint16_t)input[index] << 8));
            sample = scale_pcm_sample(sample, s_volume_percent);

            stereo[output_samples++] = sample;
            stereo[output_samples++] = sample;
            has_first_byte = false;
        }

        if (output_samples > 0 && !write_to_i2s_bytes(stereo, output_samples * sizeof(int16_t)))
        {
            return false;
        }

        consumed += received;
        const int64_t now = esp_timer_get_time();
        if (now - last_progress >= 1000000)
        {
            ESP_LOGI(TAG, "Stream playback progress: pcm=%u bytes, buffer=%u/%u, elapsed=%lld ms",
                     (unsigned)consumed,
                     (unsigned)xStreamBufferBytesAvailable(s_pcm_buffer),
                     (unsigned)AUDIO_STREAM_BUFFER_BYTES,
                     (long long)((now - started) / 1000));
            last_progress = now;
        }
    }
}

static void stream_playback_task(void *argument)
{
    (void)argument;
    while (true)
    {
        xEventGroupWaitBits(s_stream_events, STREAM_PLAY_REQUESTED, pdTRUE, pdFALSE, portMAX_DELAY);

        bool played = false;
        if (i2s_ensure_enabled(true) == ESP_OK)
        {
            played = play_buffered_audio();
            if (played)
            {
                played = flush_audio_tail();
            }
            (void)i2s_ensure_enabled(false);
        }

        xEventGroupSetBits(
            s_stream_events,
            STREAM_PLAYBACK_FINISHED | (played ? 0 : STREAM_PLAYBACK_FAILED));
        ESP_LOGI(TAG, "Stream playback task finished: %s", played ? "played" : "failed");
    }
}

static void release_pcm_buffer(void)
{
    if (s_pcm_buffer != NULL)
    {
        vStreamBufferDelete(s_pcm_buffer);
        s_pcm_buffer = NULL;
        ESP_LOGI(TAG, "Audio RAM released: buffer=0 bytes, heap=%lu",
                 (unsigned long)esp_get_free_heap_size());
    }
}

static esp_err_t parse_wav_file(FILE *fp, wav_info_t *info)
{
    uint8_t header[12];

    if (fp == NULL || info == NULL)
    {
        ESP_LOGE(TAG, "parse_wav_file invalid arg");
        return ESP_ERR_INVALID_ARG;
    }

    if (fseek(fp, 0, SEEK_SET) != 0)
    {
        ESP_LOGE(TAG, "fseek to start failed");
        return ESP_FAIL;
    }

    if (fread(header, 1, sizeof(header), fp) != sizeof(header))
    {
        ESP_LOGE(TAG, "read 12-byte WAV header failed");
        return ESP_FAIL;
    }

    ESP_LOGI(TAG, "WAV header: %.4s .... %.4s", header, header + 8);

    if (memcmp(header, "RIFF", 4) != 0 || memcmp(header + 8, "WAVE", 4) != 0)
    {
        ESP_LOGE(TAG, "Not a valid WAV file");
        return ESP_FAIL;
    }

    memset(info, 0, sizeof(*info));

    while (1)
    {
        uint8_t chunk_hdr[8];
        size_t n = fread(chunk_hdr, 1, sizeof(chunk_hdr), fp);
        if (n != sizeof(chunk_hdr))
        {
            ESP_LOGW(TAG, "Reached end of chunk list");
            break;
        }

        uint32_t chunk_size = read_le32(chunk_hdr + 4);
        long chunk_data_pos = ftell(fp);

        char chunk_name[5] = {0};
        memcpy(chunk_name, chunk_hdr, 4);

        ESP_LOGI(TAG, "chunk='%s' size=%lu pos=%ld",
                 chunk_name, (unsigned long)chunk_size, chunk_data_pos);

        if (memcmp(chunk_hdr, "fmt ", 4) == 0)
        {
            uint8_t fmt_buf[32];
            if (chunk_size < 16 || chunk_size > sizeof(fmt_buf))
            {
                ESP_LOGE(TAG, "Unsupported fmt chunk size=%lu", (unsigned long)chunk_size);
                return ESP_ERR_NOT_SUPPORTED;
            }

            if (fread(fmt_buf, 1, chunk_size, fp) != chunk_size)
            {
                ESP_LOGE(TAG, "read fmt chunk failed");
                return ESP_FAIL;
            }

            info->audio_format = read_le16(fmt_buf + 0);
            info->num_channels = read_le16(fmt_buf + 2);
            info->sample_rate = read_le32(fmt_buf + 4);
            info->bits_per_sample = read_le16(fmt_buf + 14);

            ESP_LOGI(TAG,
                     "fmt: format=%u channels=%u rate=%lu bits=%u",
                     info->audio_format,
                     info->num_channels,
                     (unsigned long)info->sample_rate,
                     info->bits_per_sample);
        }
        else if (memcmp(chunk_hdr, "data", 4) == 0)
        {
            info->data_offset = (uint32_t)chunk_data_pos;
            info->data_size = chunk_size & ~1U;

            if (chunk_size != info->data_size)
            {
                ESP_LOGW(TAG, "Odd-byte WAV data (size=%lu) — rounding to %lu",
                         (unsigned long)chunk_size,
                         (unsigned long)info->data_size);
            }

            ESP_LOGI(TAG, "data: offset=%lu size=%lu",
                     (unsigned long)info->data_offset,
                     (unsigned long)info->data_size);

            break;
        }
        else
        {
            ESP_LOGW(TAG, "skip unknown chunk '%s'", chunk_name);
            if (fseek(fp, chunk_size, SEEK_CUR) != 0)
            {
                ESP_LOGE(TAG, "skip unknown chunk failed");
                return ESP_FAIL;
            }
        }

        if (chunk_size & 1)
        {
            if (fseek(fp, 1, SEEK_CUR) != 0)
            {
                ESP_LOGE(TAG, "skip padding byte failed");
                return ESP_FAIL;
            }
        }
    }

    if (info->audio_format != 1)
    {
        ESP_LOGE(TAG, "Only PCM WAV supported, got format=%u", info->audio_format);
        return ESP_ERR_NOT_SUPPORTED;
    }

    if (info->bits_per_sample != 16)
    {
        ESP_LOGE(TAG, "Only 16-bit WAV supported, got %u-bit", info->bits_per_sample);
        return ESP_ERR_NOT_SUPPORTED;
    }

    if (info->num_channels != 1 && info->num_channels != 2)
    {
        ESP_LOGE(TAG, "Only mono/stereo supported, got channels=%u", info->num_channels);
        return ESP_ERR_NOT_SUPPORTED;
    }

    if (info->data_offset == 0 || info->data_size == 0)
    {
        ESP_LOGE(TAG, "No data chunk found");
        return ESP_FAIL;
    }

    return ESP_OK;
}

esp_err_t audio_init(int bclk_gpio, int lrck_gpio, int dout_gpio)
{
    if (bclk_gpio < 0 || lrck_gpio < 0 || dout_gpio < 0)
    {
        return ESP_ERR_INVALID_ARG;
    }
    if (s_playback_mutex == NULL)
    {
        s_playback_mutex = xSemaphoreCreateMutex();
        if (s_playback_mutex == NULL)
            return ESP_ERR_NO_MEM;
    }
    if (s_stream_events == NULL)
    {
        s_stream_events = xEventGroupCreate();
        if (s_stream_events == NULL)
            return ESP_ERR_NO_MEM;
    }
    if (s_stream_task_handle == NULL)
    {
        BaseType_t created = xTaskCreate(
            stream_playback_task,
            "i2s_stream",
            AUDIO_STREAM_TASK_STACK_BYTES,
            NULL,
            AUDIO_STREAM_TASK_PRIORITY,
            &s_stream_task_handle);
        if (created != pdPASS)
            return ESP_ERR_NO_MEM;
    }

    s_bclk_gpio = bclk_gpio;
    s_lrck_gpio = lrck_gpio;
    s_dout_gpio = dout_gpio;
    s_i2s_ready = false;
    s_i2s_enabled = false;
    s_current_sample_rate = 0;

    ESP_LOGI(TAG, "Audio init done (SPIFFS + RAM StreamBuffer ready)");
    return ESP_OK;
}

static esp_err_t audio_play_file(const char *filepath)
{
    if (filepath == NULL)
    {
        ESP_LOGE(TAG, "filepath is NULL");
        return ESP_ERR_INVALID_ARG;
    }

    s_is_playing = true;
    s_stop_requested = false;

    ESP_LOGI(TAG, "Try open file: %s", filepath);

    FILE *fp = fopen(filepath, "rb");
    if (fp == NULL)
    {
        ESP_LOGE(TAG, "Cannot open file: %s", filepath);
        return ESP_ERR_NOT_FOUND;
    }

    wav_info_t info;
    esp_err_t ret = parse_wav_file(fp, &info);
    if (ret != ESP_OK)
    {
        ESP_LOGE(TAG, "parse_wav_file failed: %s", esp_err_to_name(ret));
        fclose(fp);
        return ret;
    }

    ret = audio_prepare_i2s(info.sample_rate, true);
    if (ret != ESP_OK)
    {
        ESP_LOGE(TAG, "audio_prepare_i2s failed: %s", esp_err_to_name(ret));
        fclose(fp);
        return ret;
    }

    if (fseek(fp, info.data_offset, SEEK_SET) != 0)
    {
        ESP_LOGE(TAG, "fseek to data offset failed");
        fclose(fp);
        (void)i2s_ensure_enabled(false);
        return ESP_FAIL;
    }

    ESP_LOGI(TAG,
             "Playing %s (%lu Hz, %u ch, %u bit, %lu bytes)",
             filepath,
             (unsigned long)info.sample_rate,
             info.num_channels,
             info.bits_per_sample,
             (unsigned long)info.data_size);

    size_t remaining = info.data_size;
    size_t total_written = 0;

    if (info.num_channels == 2)
    {
        static uint8_t buf[1024];

        while (remaining > 0)
        {
            size_t to_read = remaining > sizeof(buf) ? sizeof(buf)
                                                     : (remaining & ~3U);

            if (to_read == 0)
                break;

            size_t got = fread(buf, 1, to_read, fp);
            if (got == 0)
            {
                if (ferror(fp))
                {
                    ESP_LOGE(TAG, "fread stereo error");
                    fclose(fp);
                    (void)i2s_ensure_enabled(false);
                    return ESP_FAIL;
                }
                break;
            }

            size_t written = 0;
            ret = i2s_channel_write(
                s_tx_handle,
                buf,
                got,
                &written,
                portMAX_DELAY);

            if (ret != ESP_OK)
            {
                ESP_LOGE(TAG, "i2s_channel_write stereo failed: %s",
                         esp_err_to_name(ret));
                fclose(fp);
                (void)i2s_ensure_enabled(false);
                return ret;
            }

            total_written += written;
            remaining -= written;

            if (s_stop_requested)
            {
                ESP_LOGW(TAG, "Stop requested — halting playback");
                fclose(fp);
                (void)i2s_ensure_enabled(false);
                s_is_playing = false;
                s_stop_requested = false;
                return ESP_OK;
            }

            if (written < got)
            {
                ESP_LOGW(TAG, "I2S buffer full — wrote %u/%u bytes, remaining=%u",
                         (unsigned)written, (unsigned)got, (unsigned)remaining);
                if (fseek(fp, -(long)(got - written), SEEK_CUR) != 0)
                {
                    ESP_LOGE(TAG, "fseek back failed");
                    fclose(fp);
                    (void)i2s_ensure_enabled(false);
                    return ESP_FAIL;
                }
            }
        }
    }
    else
    {
        static int16_t mono_buf[256];
        static int16_t stereo_buf[512];

        while (remaining > 0)
        {
            size_t mono_bytes = remaining > sizeof(mono_buf) ? sizeof(mono_buf)
                                                             : (remaining & ~1U);

            if (mono_bytes == 0)
                break;

            size_t got = fread(mono_buf, 1, mono_bytes, fp);
            if (got == 0)
            {
                if (ferror(fp))
                {
                    ESP_LOGE(TAG, "fread mono error");
                    fclose(fp);
                    (void)i2s_ensure_enabled(false);
                    return ESP_FAIL;
                }
                break;
            }

            size_t mono_samples = got / sizeof(int16_t);

            for (size_t i = 0; i < mono_samples; i++)
            {
                stereo_buf[i * 2] = mono_buf[i];
                stereo_buf[i * 2 + 1] = mono_buf[i];
            }

            size_t stereo_bytes = mono_samples * 2 * sizeof(int16_t);
            size_t written = 0;
            ret = i2s_channel_write(
                s_tx_handle,
                stereo_buf,
                stereo_bytes,
                &written,
                portMAX_DELAY);

            if (ret != ESP_OK)
            {
                ESP_LOGE(TAG, "i2s_channel_write mono failed: %s",
                         esp_err_to_name(ret));
                fclose(fp);
                (void)i2s_ensure_enabled(false);
                return ret;
            }

            size_t consumed = (written / (2 * sizeof(int16_t))) * sizeof(int16_t);
            total_written += consumed;
            remaining -= consumed;

            if (s_stop_requested)
            {
                ESP_LOGW(TAG, "Stop requested — halting playback");
                fclose(fp);
                (void)i2s_ensure_enabled(false);
                s_is_playing = false;
                s_stop_requested = false;
                return ESP_OK;
            }

            if (written < stereo_bytes)
            {
                ESP_LOGW(TAG,
                         "I2S mono underrun — wrote %u/%u bytes, remaining=%u",
                         (unsigned)written, (unsigned)stereo_bytes,
                         (unsigned)remaining);
            }
        }
    }

    fclose(fp);
    (void)i2s_ensure_enabled(false);
    ESP_LOGI(TAG, "Done: %s (wrote %u bytes)", filepath, (unsigned)total_written);
    s_is_playing = false;
    return ESP_OK;
}

esp_err_t audio_play(const char *filepath)
{
    if (s_playback_mutex == NULL)
        return ESP_ERR_INVALID_STATE;
    if (xSemaphoreTake(s_playback_mutex, 0) != pdTRUE)
        return ESP_ERR_INVALID_STATE;
    esp_err_t err = audio_play_file(filepath);
    s_is_playing = false;
    s_stop_requested = false;
    xSemaphoreGive(s_playback_mutex);
    return err;
}

esp_err_t audio_stream_begin(uint32_t sample_rate, unsigned volume_percent)
{
    if (s_playback_mutex == NULL || s_stream_events == NULL)
        return ESP_ERR_INVALID_STATE;
    if (xSemaphoreTake(s_playback_mutex, 0) != pdTRUE)
        return ESP_ERR_INVALID_STATE;

    if (s_pcm_buffer != NULL)
    {
        if (!(xEventGroupGetBits(s_stream_events) & STREAM_PLAYBACK_FINISHED))
        {
            xSemaphoreGive(s_playback_mutex);
            return ESP_ERR_INVALID_STATE;
        }
        release_pcm_buffer();
    }

    esp_err_t err = audio_prepare_i2s(sample_rate, false);
    if (err != ESP_OK)
    {
        xSemaphoreGive(s_playback_mutex);
        return err;
    }

    s_pcm_buffer = xStreamBufferCreate(AUDIO_STREAM_BUFFER_BYTES, 1);
    if (s_pcm_buffer == NULL)
    {
        ESP_LOGE(TAG, "Cannot allocate audio stream RAM (%u bytes)",
                 (unsigned)AUDIO_STREAM_BUFFER_BYTES);
        xSemaphoreGive(s_playback_mutex);
        return ESP_ERR_NO_MEM;
    }

    xEventGroupClearBits(
        s_stream_events,
        STREAM_PLAY_REQUESTED | STREAM_DOWNLOAD_FINISHED |
            STREAM_PLAYBACK_FINISHED | STREAM_PLAYBACK_FAILED);
    s_volume_percent = (volume_percent > 100U) ? 100U : volume_percent;
    s_playback_requested = false;
    s_stop_requested = false;
    s_is_playing = true;
    s_stream_mutex_held = true;

    ESP_LOGI(TAG, "Audio RAM allocated: %u bytes (rate=%lu Hz, volume=%u%%)",
             (unsigned)AUDIO_STREAM_BUFFER_BYTES,
             (unsigned long)sample_rate,
             s_volume_percent);
    return ESP_OK;
}

esp_err_t audio_stream_push(const uint8_t *data, size_t size)
{
    if (s_pcm_buffer == NULL || s_stream_events == NULL)
        return ESP_ERR_INVALID_STATE;

    while (size > 0)
    {
        if (s_stop_requested ||
            (xEventGroupGetBits(s_stream_events) &
             (STREAM_PLAYBACK_FAILED | STREAM_PLAYBACK_FINISHED)))
        {
            return ESP_FAIL;
        }

        size_t written = xStreamBufferSend(
            s_pcm_buffer,
            data,
            size,
            pdMS_TO_TICKS(AUDIO_BUFFER_WRITE_TIMEOUT_MS));
        if (written == 0)
        {
            ESP_LOGE(TAG, "Stream buffer write timeout: buffer=%u/%u bytes",
                     (unsigned)xStreamBufferBytesAvailable(s_pcm_buffer),
                     (unsigned)AUDIO_STREAM_BUFFER_BYTES);
            return ESP_ERR_TIMEOUT;
        }
        data += written;
        size -= written;

        if (!s_playback_requested &&
            xStreamBufferBytesAvailable(s_pcm_buffer) >= AUDIO_STREAM_PREBUFFER_BYTES)
        {
            s_playback_requested = true;
            xEventGroupSetBits(s_stream_events, STREAM_PLAY_REQUESTED);
        }
    }
    return ESP_OK;
}

size_t audio_stream_buffered_bytes(void)
{
    return (s_pcm_buffer != NULL) ? xStreamBufferBytesAvailable(s_pcm_buffer) : 0;
}

esp_err_t audio_stream_end(bool abort_playback)
{
    if (s_pcm_buffer == NULL || s_stream_events == NULL)
    {
        if (s_stream_mutex_held)
        {
            s_stream_mutex_held = false;
            s_is_playing = false;
            xSemaphoreGive(s_playback_mutex);
        }
        return ESP_ERR_INVALID_STATE;
    }

    EventBits_t flags = STREAM_DOWNLOAD_FINISHED;
    if (!s_playback_requested)
    {
        flags |= STREAM_PLAY_REQUESTED;
    }
    if (abort_playback || s_stop_requested)
    {
        flags |= STREAM_PLAYBACK_FAILED;
    }
    xEventGroupSetBits(s_stream_events, flags);

    EventBits_t result = xEventGroupWaitBits(
        s_stream_events,
        STREAM_PLAYBACK_FINISHED,
        pdFALSE,
        pdFALSE,
        pdMS_TO_TICKS(AUDIO_PLAYBACK_FINISH_TIMEOUT_MS));
    bool timed_out = !(result & STREAM_PLAYBACK_FINISHED);
    if (timed_out)
    {
        ESP_LOGE(TAG, "Waiting for stream playback timed out: buffer=%u bytes",
                 (unsigned)audio_stream_buffered_bytes());
        xEventGroupSetBits(s_stream_events, STREAM_PLAYBACK_FAILED);
        result = xEventGroupWaitBits(
            s_stream_events,
            STREAM_PLAYBACK_FINISHED,
            pdFALSE,
            pdFALSE,
            pdMS_TO_TICKS(AUDIO_PLAYBACK_FINISH_TIMEOUT_MS));
    }

    if (result & STREAM_PLAYBACK_FINISHED)
    {
        release_pcm_buffer();
    }
    else
    {
        ESP_LOGE(TAG, "Audio cleanup pending: stream task has not stopped");
    }

    s_is_playing = false;
    bool was_stopped = s_stop_requested;
    s_stop_requested = false;
    if (s_stream_mutex_held)
    {
        s_stream_mutex_held = false;
        xSemaphoreGive(s_playback_mutex);
    }

    if (!timed_out && (result & STREAM_PLAYBACK_FINISHED) && !(result & STREAM_PLAYBACK_FAILED))
    {
        return ESP_OK;
    }
    return was_stopped ? ESP_ERR_INVALID_STATE : ESP_FAIL;
}

esp_err_t audio_stop(void)
{
    if (s_is_playing)
    {
        s_stop_requested = true;
        if (s_stream_events != NULL)
        {
            xEventGroupSetBits(s_stream_events, STREAM_PLAYBACK_FAILED);
        }
        ESP_LOGW(TAG, "Stop requested");
    }
    return ESP_OK;
}

bool audio_is_playing(void)
{
    return s_is_playing;
}
