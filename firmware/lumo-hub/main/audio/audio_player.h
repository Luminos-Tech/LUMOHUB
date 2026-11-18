#ifndef AUDIO_PLAYER_H
#define AUDIO_PLAYER_H

#include "esp_err.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

esp_err_t audio_init(int bclk_gpio, int lrck_gpio, int dout_gpio); // 5 4 6
esp_err_t audio_play(const char *filepath);
esp_err_t audio_stop(void);
bool audio_is_playing(void);

/* Streaming PCM16 mono từ RAM StreamBuffer (32 KB, prebuffer 12 KB), không ghi SPIFFS */
esp_err_t audio_stream_begin(uint32_t sample_rate, unsigned volume_percent);
esp_err_t audio_stream_push(const uint8_t *data, size_t size);
esp_err_t audio_stream_end(bool abort_playback);
size_t audio_stream_buffered_bytes(void);

#endif
