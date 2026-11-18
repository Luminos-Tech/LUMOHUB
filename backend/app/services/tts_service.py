"""Dịch vụ TTS, chuẩn hóa âm thanh, cache SHA-256 và hàng đợi PCM cho LUMO Hub.

Kế thừa và nâng cấp từ DSOFT-AudioTTS:
- Chuẩn hóa đầu ra về PCM16 signed little-endian mono 24 kHz (48.000 B/s).
- Cache đĩa theo SHA-256(cấu hình không gồm secret + text) tách khóa với hàng đợi.
- Hỗ trợ đa nguồn: http (adapter chuẩn DSOFT), elevenlabs, gemini, google_translate, tone, auto.
- Hàng đợi SQLite WAL với lease & ACK idempotent để ESP32 stream trực tiếp vào RAM.
"""
from __future__ import annotations

import base64
import binascii
from contextlib import contextmanager
import hashlib
import io
import json
import logging
import math
from pathlib import Path
import re
import secrets
import shutil
import sqlite3
import struct
import subprocess
import tempfile
import threading
import time
from typing import Any
from urllib.parse import quote, urlparse
import wave

from fastapi import HTTPException
import httpx

from app.core.config import settings as app_settings

SAMPLE_RATE_HZ = 24000
RATE = SAMPLE_RATE_HZ
PCM_BYTES_PER_SAMPLE = 2
CHANNELS = 1
MAX_AUDIO_SECONDS = 60
MAX_AUDIO_BYTES = SAMPLE_RATE_HZ * PCM_BYTES_PER_SAMPLE * MAX_AUDIO_SECONDS  # 2_880_000 bytes
MAX_TEXT_LENGTH = 1000
DEVICE_LEASE_SECONDS = 240
DEVICE_ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"

GEMINI_VOICES = {"Kore", "Aoede", "Puck", "Fenrir", "Charon", "Zephyr"}
DEFAULT_BODY_TEMPLATE = {
    "text": "{{text}}",
    "voice": "{{voice}}",
    "format": "pcm_s16le",
    "sample_rate": SAMPLE_RATE_HZ,
}

logger = logging.getLogger("uvicorn.error")


def pcm_to_wav_bytes(pcm_data: bytes, sample_rate: int = SAMPLE_RATE_HZ) -> bytes:
    """Đóng gói PCM16 mono thành file WAV hoàn chỉnh ngay trong RAM."""
    data_len = len(pcm_data)
    byte_rate = sample_rate * CHANNELS * PCM_BYTES_PER_SAMPLE
    block_align = CHANNELS * PCM_BYTES_PER_SAMPLE
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_len,
        b"WAVE",
        b"fmt ",
        16,
        1,
        CHANNELS,
        sample_rate,
        byte_rate,
        block_align,
        PCM_BYTES_PER_SAMPLE * 8,
        b"data",
        data_len,
    )
    return header + pcm_data


def extract_inline_audio_bytes(data: bytes | str) -> bytes:
    """Giải mã inline_data.data từ Gemini SDK (có thể là raw bytes hoặc base64)."""
    if isinstance(data, str):
        return base64.b64decode(data)

    raw = data
    try:
        ascii_data = raw.decode("ascii")
    except UnicodeDecodeError:
        return raw

    try:
        return base64.b64decode(ascii_data, validate=True)
    except binascii.Error:
        return raw


def sample_rate_from_mime(mime_type: str, default: int = SAMPLE_RATE_HZ) -> int:
    """Trích xuất sample rate từ MIME như 'audio/L16;codec=pcm;rate=24000'."""
    for token in (mime_type or "").split(";"):
        token = token.strip()
        if token.startswith("rate="):
            value = token.split("=", 1)[1]
            if value.isdigit():
                return int(value)
    return default


def provider_failure(response: httpx.Response) -> HTTPException:
    """Chỉ đọc mã lỗi đã biết; không ghi body có thể chứa khóa hoặc văn bản."""
    body = bytearray()
    for chunk in response.iter_bytes(chunk_size=4096):
        body.extend(chunk)
        if len(body) > 16384:
            break
    try:
        data = json.loads(body)
        errors = [data.get(key) for key in ("error", "detail")] if isinstance(data, dict) else []
        errors = [error for error in errors if isinstance(error, dict)]
    except (ValueError, UnicodeError):
        errors = []
    known_codes = {
        "invalid_api_key", "insufficient_quota", "credit_balance_exhausted",
        "organization_spend_limit_exceeded", "project_spend_limit_exceeded",
        "organization_usage_limit_exceeded", "rate_limit_exceeded", "slow_down",
        "model_not_found", "invalid_request_error", "unsupported_country_region_territory",
        "missing_api_key", "missing_permissions", "insufficient_permissions", "unusual_activity",
        "quota_exceeded", "voice_not_found", "voice_access_denied", "model_access_denied",
        "subscription_required", "authentication_error", "authorization_error",
        "payment_required", "insufficient_credits",
    }
    code = next(
        (
            value
            for error in errors
            for field in ("code", "status", "type")
            for value in (error.get(field),)
            if isinstance(value, str) and value in known_codes
        ),
        "unknown",
    )
    status = response.status_code
    if code in {"missing_permissions", "insufficient_permissions", "authorization_error"}:
        message = "Khóa API TTS thiếu quyền. Kiểm tra quyền Text to Speech của khóa trong tài khoản nhà cung cấp."
    elif code == "unusual_activity":
        message = "ElevenLabs đã hạn chế tài khoản do phát hiện hoạt động bất thường. Kiểm tra thông báo tài khoản hoặc liên hệ hỗ trợ ElevenLabs."
    elif code in {"voice_access_denied", "model_access_denied", "subscription_required"}:
        message = "Tài khoản API TTS chưa có quyền dùng giọng, model hoặc tính năng này. Kiểm tra quyền truy cập và gói dịch vụ."
    elif status == 402 or code in {"payment_required", "insufficient_credits"}:
        message = "API TTS yêu cầu thanh toán hoặc không đủ credits. Kiểm tra số dư, hạn mức API key và gói dịch vụ của tài khoản nhà cung cấp."
    elif code == "quota_exceeded":
        message = "Tài khoản API TTS hết số dư hoặc chạm hạn mức sử dụng. Kiểm tra hạn mức của tài khoản và API key."
    elif code == "voice_not_found":
        message = "Không tìm thấy Voice ID. Kiểm tra mã giọng trong URL và thư viện giọng của tài khoản ElevenLabs."
    elif code == "missing_api_key":
        message = "API TTS chưa nhận được khóa. Kiểm tra khóa đã lưu và header xác thực xi-api-key nếu dùng ElevenLabs."
    elif code == "invalid_api_key":
        message = "Khóa API TTS không hợp lệ hoặc đã bị thu hồi. Nhập lại khóa của nhà cung cấp."
    elif status == 401:
        message = "API TTS từ chối xác thực. Kiểm tra khóa đã lưu, header xác thực và quyền của khóa."
    elif status == 403:
        message = "API TTS từ chối quyền truy cập. Kiểm tra quyền của khóa, model và giới hạn IP/khu vực."
    elif status == 429 and code in {
        "insufficient_quota", "credit_balance_exhausted", "organization_spend_limit_exceeded",
        "project_spend_limit_exceeded", "organization_usage_limit_exceeded",
    }:
        message = "Tài khoản API TTS hết số dư hoặc chạm hạn mức sử dụng. Kiểm tra Billing và Limits của nhà cung cấp."
    elif status == 429:
        message = "API TTS đang giới hạn yêu cầu. Đợi một lúc rồi thử lại; kiểm tra Limits nếu lỗi tiếp tục."
    elif status == 404 or code == "model_not_found":
        message = "Không tìm thấy URL API TTS hoặc model, hoặc khóa chưa có quyền dùng model đó."
    elif status in {400, 422}:
        message = "API TTS không chấp nhận cấu hình. Kiểm tra model, voice và JSON request."
        param = next((error.get("param") for error in errors if isinstance(error.get("param"), str)), None)
        if isinstance(param, str) and param in {"model", "input", "voice", "response_format", "speed", "instructions"}:
            message += f" Trường cần kiểm tra: {param}."
    elif status >= 500:
        message = "Dịch vụ TTS hoặc proxy phía nhà cung cấp đang lỗi. Thử lại sau."
    else:
        message = "API TTS từ chối yêu cầu. Kiểm tra URL, khóa và JSON request."
    logger.warning("TTS provider rejected request: upstream_status=%s code=%s", status, code)
    return HTTPException(502, f"{message} (API HTTP {status}; code={code})")


def _resample_wav_in_python(data: bytes) -> bytes | None:
    """Chuyển WAV PCM bất kỳ về PCM16 mono 24 kHz không cần ffmpeg."""
    try:
        with wave.open(io.BytesIO(data), "rb") as source:
            channels = source.getnchannels()
            sampwidth = source.getsampwidth()
            framerate = source.getframerate()
            frames = source.readframes(source.getnframes())
    except (wave.Error, EOFError):
        return None

    if (channels, sampwidth, framerate) == (1, 2, RATE):
        return frames

    if sampwidth not in (1, 2, 3, 4) or channels not in (1, 2) or framerate <= 0:
        return None

    try:
        import audioop  # type: ignore

        pcm = frames
        if sampwidth != 2:
            pcm = audioop.lin2lin(pcm, sampwidth, 2)
        if channels == 2:
            pcm = audioop.tomono(pcm, 2, 0.5, 0.5)
        if framerate != RATE:
            pcm, _ = audioop.ratecv(pcm, 2, 1, framerate, RATE, None)
        return pcm
    except ImportError:
        pass

    # Pure-Python fallback nếu Python >= 3.13 bỏ audioop
    if sampwidth != 2:
        return None
    total_samples = len(frames) // (2 * channels)
    if total_samples == 0:
        return b""
    unpacked = struct.unpack(f"<{total_samples * channels}h", frames[: total_samples * 2 * channels])
    if channels == 2:
        mono = [(unpacked[i] + unpacked[i + 1]) // 2 for i in range(0, len(unpacked), 2)]
    else:
        mono = list(unpacked)
    if framerate == RATE:
        return struct.pack(f"<{len(mono)}h", *mono)
    out_len = max(1, int(round(len(mono) * RATE / framerate)))
    resampled = []
    ratio = (len(mono) - 1) / max(1, out_len - 1)
    for idx in range(out_len):
        pos = idx * ratio
        lo = int(pos)
        hi = min(lo + 1, len(mono) - 1)
        frac = pos - lo
        val = int(mono[lo] * (1.0 - frac) + mono[hi] * frac)
        resampled.append(max(-32768, min(32767, val)))
    return struct.pack(f"<{len(resampled)}h", *resampled)


def _transcode_with_pyav(data: bytes) -> bytes | None:
    """Fallback chuyển MP3/WAV sang PCM16 mono 24 kHz bằng PyAV nếu thiếu ffmpeg."""
    try:
        import av  # type: ignore
    except ImportError:
        return None

    try:
        container = av.open(io.BytesIO(data))
        resampler = av.AudioResampler(format="s16", layout="mono", rate=RATE)
        chunks: list[bytes] = []
        total_bytes = 0
        for frame in container.decode(audio=0):
            for out_frame in resampler.resample(frame):
                raw = out_frame.to_ndarray().tobytes()
                chunks.append(raw)
                total_bytes += len(raw)
                if total_bytes > MAX_AUDIO_BYTES:
                    break
            if total_bytes > MAX_AUDIO_BYTES:
                break
        for out_frame in resampler.resample(None):
            raw = out_frame.to_ndarray().tobytes()
            chunks.append(raw)
            total_bytes += len(raw)
        container.close()
        return b"".join(chunks)[:MAX_AUDIO_BYTES]
    except Exception:
        return None


def transcode(data: bytes) -> bytes:
    """Chuyển MP3/WAV khác định dạng sang PCM16 mono 24 kHz chuẩn cho ESP32."""
    executable = shutil.which("ffmpeg")
    if executable:
        # Limit output, run without a shell, and disable network protocols in the decoder.
        with tempfile.TemporaryDirectory(prefix="lumo-audio-") as tmp:
            src, dst = Path(tmp) / "source.audio", Path(tmp) / "output.pcm"
            src.write_bytes(data)
            try:
                result = subprocess.run(
                    [
                        executable,
                        "-nostdin",
                        "-v",
                        "error",
                        "-protocol_whitelist",
                        "file,pipe",
                        "-i",
                        str(src),
                        "-t",
                        "61",
                        "-f",
                        "s16le",
                        "-acodec",
                        "pcm_s16le",
                        "-ar",
                        str(RATE),
                        "-ac",
                        "1",
                        str(dst),
                    ],
                    capture_output=True,
                    timeout=30,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                raise HTTPException(502, "Audio conversion timed out")
            if result.returncode == 0 and dst.exists():
                return dst.read_bytes()

    wav_fallback = _resample_wav_in_python(data)
    if wav_fallback is not None:
        return wav_fallback

    pyav_fallback = _transcode_with_pyav(data)
    if pyav_fallback is not None:
        return pyav_fallback

    if not executable:
        raise HTTPException(503, "Install ffmpeg to convert MP3/noncanonical WAV")
    raise HTTPException(502, "Unable to decode provider audio")


def normalize_audio(data: bytes, fmt: str) -> bytes:
    """Chuẩn hóa audio nhà cung cấp trả về thành PCM16 mono 24 kHz cho ESP32."""
    normalized_fmt = (fmt or "pcm").lower().strip()
    if normalized_fmt in {"pcm", "pcm_24000", "pcm_s16le"}:
        pcm = data
    elif normalized_fmt == "wav":
        try:
            with wave.open(io.BytesIO(data), "rb") as source:
                if (source.getnchannels(), source.getsampwidth(), source.getframerate()) == (1, 2, RATE):
                    pcm = source.readframes(source.getnframes())
                else:
                    pcm = transcode(data)
        except (wave.Error, EOFError):
            raise HTTPException(502, "Invalid WAV from TTS")
    elif normalized_fmt == "mp3":
        pcm = transcode(data)
    else:
        raise HTTPException(503, "TTS_RESPONSE_FORMAT must be pcm, wav or mp3")

    if not pcm or len(pcm) % 2 or len(pcm) > MAX_AUDIO_BYTES:
        raise HTTPException(502, "TTS must return 0 < duration <= 60s of PCM16 mono 24kHz")
    return pcm


def _split_text_for_google(text: str, max_len: int = 180) -> list[str]:
    """Chia văn bản dài thành các đoạn <= 180 ký tự cho Google Translate TTS."""
    clean = " ".join(text.strip().split())
    if len(clean) <= max_len:
        return [clean] if clean else []

    chunks: list[str] = []
    sentences = re.split(r"(?<=[.!?;,\n])\s+", clean)
    current = ""
    for sentence in sentences:
        if not sentence:
            continue
        if len(sentence) > max_len:
            if current:
                chunks.append(current)
                current = ""
            words = sentence.split(" ")
            buf = ""
            for word in words:
                candidate = f"{buf} {word}".strip() if buf else word
                if len(candidate) <= max_len:
                    buf = candidate
                else:
                    if buf:
                        chunks.append(buf)
                    buf = word[:max_len]
            if buf:
                current = buf
            continue
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= max_len:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    return chunks


def _detect_lang(text: str, default: str = "vi") -> str:
    if any(ord(ch) > 127 for ch in text):
        return "vi"
    return default


def _synthesize_tone() -> bytes:
    return b"".join(
        struct.pack("<h", int(6000 * math.sin(2 * math.pi * 440 * i / RATE)))
        for i in range(RATE)
    )


def _synthesize_http(text: str, cfg: dict[str, Any]) -> bytes:
    url = str(cfg.get("tts_url", "")).strip()
    if not url:
        raise HTTPException(503, "TTS API is not configured; set TTS_PROVIDER and TTS_URL")
    if urlparse(url).scheme != "https":
        raise HTTPException(503, "TTS_URL requires HTTPS")

    prefix = str(cfg.get("auth_prefix", "Bearer")).strip()
    secret = str(cfg.get("tts_key", "") or "")
    value = f"{prefix} {secret}" if prefix else secret
    headers = {str(cfg.get("auth_header", "Authorization")): value} if secret else {}

    variables = {
        "text": text,
        "voice": str(cfg.get("voice", "default")),
        "model": str(cfg.get("model", "")),
    }

    def render_template(val: Any) -> Any:
        if isinstance(val, str):
            return re.sub(r"\{\{(text|voice|model)\}\}", lambda match: variables[match[1]], val)
        if isinstance(val, list):
            return [render_template(item) for item in val]
        if isinstance(val, dict):
            return {key: render_template(item) for key, item in val.items()}
        return val

    template = cfg.get("body_template") or DEFAULT_BODY_TEMPLATE
    payload = render_template(template)

    try:
        with httpx.Client(timeout=40, follow_redirects=False) as client:
            with client.stream("POST", url, headers=headers, json=payload) as response:
                if response.status_code != 200:
                    raise provider_failure(response)
                if "json" in response.headers.get("content-type", "").lower():
                    logger.warning("TTS provider returned JSON instead of binary audio")
                    raise HTTPException(502, "This adapter expects binary audio, not a JSON audio URL")
                chunks, size = [], 0
                for chunk in response.iter_bytes():
                    size += len(chunk)
                    if size > 12 * 1024 * 1024:
                        raise HTTPException(502, "TTS response too large")
                    chunks.append(chunk)
    except httpx.TimeoutException:
        logger.warning("TTS connection timed out after 40 seconds")
        raise HTTPException(502, "API TTS phản hồi quá chậm. Kiểm tra kết nối mạng của server và thử lại.") from None
    except httpx.HTTPError as error:
        logger.warning("TTS connection failed: error_type=%s", type(error).__name__)
        raise HTTPException(502, "TTS connection failed")

    return normalize_audio(b"".join(chunks), str(cfg.get("format", "pcm")))


def _synthesize_elevenlabs(text: str, cfg: dict[str, Any]) -> bytes:
    api_key = str(cfg.get("elevenlabs_key") or cfg.get("tts_key") or app_settings.ELEVENLABS_API_KEY or "").strip()
    if not api_key:
        raise HTTPException(503, "ELEVENLABS_API_KEY is not configured")

    voice = str(cfg.get("voice") or "").strip()
    if not voice or voice in GEMINI_VOICES or voice == "default":
        voice = str(cfg.get("elevenlabs_voice_id") or app_settings.ELEVENLABS_VOICE_ID or "21m00Tcm4TlvDq8ikWAM").strip()

    model = str(cfg.get("model") or app_settings.TTS_MODEL or "eleven_flash_v2_5").strip() or "eleven_flash_v2_5"
    el_cfg = {
        "tts_url": f"https://api.elevenlabs.io/v1/text-to-speech/{quote(voice, safe='')}?output_format=pcm_24000",
        "tts_key": api_key,
        "auth_header": "xi-api-key",
        "auth_prefix": "",
        "voice": voice,
        "model": model,
        "format": "pcm",
        "body_template": {
            "text": "{{text}}",
            "model_id": "{{model}}",
        },
    }
    return _synthesize_http(text, el_cfg)


def _synthesize_google_translate(text: str, cfg: dict[str, Any]) -> bytes:
    chunks_text = _split_text_for_google(text)
    if not chunks_text:
        raise HTTPException(422, "Text cannot be empty")
    lang = str(cfg.get("lang") or _detect_lang(text, "vi")).strip() or "vi"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        "Referer": "https://translate.google.com/",
    }
    pcm_parts: list[bytes] = []
    try:
        with httpx.Client(timeout=10, follow_redirects=False) as client:
            for part in chunks_text:
                resp = client.get(
                    "https://translate.google.com/translate_tts",
                    params={"ie": "UTF-8", "q": part, "tl": lang, "client": "tw-ob"},
                    headers=headers,
                )
                if resp.status_code != 200 or not resp.content:
                    raise HTTPException(502, f"Google Translate TTS failed (HTTP {resp.status_code})")
                pcm_parts.append(normalize_audio(resp.content, "mp3"))
    except httpx.TimeoutException:
        raise HTTPException(502, "Google Translate TTS timed out") from None
    except httpx.HTTPError as error:
        logger.warning("Google Translate TTS failed: error_type=%s", type(error).__name__)
        raise HTTPException(502, "Google Translate TTS connection failed") from None

    pcm = b"".join(pcm_parts)
    if not pcm or len(pcm) % 2 or len(pcm) > MAX_AUDIO_BYTES:
        raise HTTPException(502, "TTS audio length out of bounds")
    return pcm


def _synthesize_gemini(text: str, cfg: dict[str, Any]) -> bytes:
    api_key = str(cfg.get("gemini_key") or app_settings.GEMINI_API_KEY or "").strip()
    if not api_key:
        raise HTTPException(503, "GEMINI_API_KEY not set")

    from google import genai
    import google.genai.types as gtypes

    voice_name = str(cfg.get("voice") or "Kore").strip()
    if voice_name not in GEMINI_VOICES:
        voice_name = "Kore"

    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(
        model="gemini-2.5-flash-preview-tts",
        contents=text,
        config=gtypes.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=gtypes.SpeechConfig(
                voice_config=gtypes.VoiceConfig(
                    prebuilt_voice_config=gtypes.PrebuiltVoiceConfig(
                        voice_name=voice_name,
                    )
                )
            ),
        ),
    )
    candidates = getattr(resp, "candidates", None) or []
    if not candidates:
        raise HTTPException(502, "Gemini TTS returned no candidates")
    parts = getattr(getattr(candidates[0], "content", None), "parts", None) or []
    if not parts or not getattr(parts[0], "inline_data", None):
        raise HTTPException(502, "Gemini TTS returned no inline audio")

    inline_data = parts[0].inline_data
    audio_bytes = extract_inline_audio_bytes(inline_data.data)
    mime_type = getattr(inline_data, "mime_type", "") or ""
    if mime_type == "audio/wav":
        return normalize_audio(audio_bytes, "wav")

    sample_rate = sample_rate_from_mime(mime_type, default=RATE)
    if sample_rate == RATE:
        return normalize_audio(audio_bytes, "pcm")
    return normalize_audio(pcm_to_wav_bytes(audio_bytes, sample_rate=sample_rate), "wav")


def synthesize_pcm(
    text: str,
    voice: str | None = None,
    override_settings: dict[str, Any] | None = None,
) -> bytes:
    """Tổng hợp giọng nói thành PCM16 mono 24 kHz theo provider được cấu hình."""
    cfg = get_effective_settings(override_settings, voice=voice)
    provider = str(cfg.get("provider") or "auto").lower().strip()

    if provider == "tone":
        return _synthesize_tone()
    if provider == "http":
        return _synthesize_http(text, cfg)
    if provider == "elevenlabs":
        return _synthesize_elevenlabs(text, cfg)
    if provider == "google_translate":
        return _synthesize_google_translate(text, cfg)
    if provider == "gemini":
        return _synthesize_gemini(text, cfg)

    if provider not in {"", "auto"}:
        raise HTTPException(503, f"Unsupported TTS provider: {provider}")

    # auto mode: ưu tiên HTTP adapter -> ElevenLabs Flash -> Gemini -> Google Translate
    if str(cfg.get("tts_url") or "").strip():
        return _synthesize_http(text, cfg)

    if str(cfg.get("elevenlabs_key") or app_settings.ELEVENLABS_API_KEY or "").strip():
        try:
            return _synthesize_elevenlabs(text, cfg)
        except Exception as exc:
            logger.warning("ElevenLabs TTS fallback triggered: %s", type(exc).__name__)

    if str(cfg.get("gemini_key") or app_settings.GEMINI_API_KEY or "").strip():
        try:
            return _synthesize_gemini(text, cfg)
        except Exception as exc:
            logger.warning("Gemini TTS fallback to fast engine triggered: %s", type(exc).__name__)

    return _synthesize_google_translate(text, cfg)


def _default_settings() -> dict[str, Any]:
    return {
        "provider": app_settings.TTS_PROVIDER or "auto",
        "tts_url": app_settings.TTS_URL or "",
        "tts_key": app_settings.TTS_API_KEY or "",
        "voice": app_settings.TTS_VOICE or "Kore",
        "model": app_settings.TTS_MODEL or "eleven_flash_v2_5",
        "format": app_settings.TTS_RESPONSE_FORMAT or "pcm",
        "auth_header": app_settings.TTS_AUTH_HEADER or "Authorization",
        "auth_prefix": app_settings.TTS_AUTH_PREFIX or "Bearer",
        "body_template": dict(DEFAULT_BODY_TEMPLATE),
        "revision": "1",
    }


def _cache_digest(text: str, cfg: dict[str, Any]) -> str:
    """Băm SHA-256 không chứa secret để dùng chung cache PCM an toàn."""
    provider = str(cfg.get("provider") or "auto").lower().strip()
    if provider in {"", "auto"}:
        if str(cfg.get("tts_url") or "").strip():
            effective_mode = "http"
        elif str(cfg.get("elevenlabs_key") or app_settings.ELEVENLABS_API_KEY or "").strip():
            effective_mode = "elevenlabs"
        elif str(cfg.get("gemini_key") or app_settings.GEMINI_API_KEY or "").strip():
            effective_mode = "gemini"
        else:
            effective_mode = "google_translate"
    else:
        effective_mode = provider

    cache_input = {
        "provider": effective_mode,
        "tts_url": cfg.get("tts_url", ""),
        "voice": cfg.get("voice", "Kore"),
        "model": cfg.get("model", ""),
        "format": cfg.get("format", "pcm"),
        "revision": cfg.get("revision", "1"),
        "body_template": cfg.get("body_template", DEFAULT_BODY_TEMPLATE),
        "lang": cfg.get("lang", ""),
        "text": text,
    }
    return hashlib.sha256(
        json.dumps(cache_input, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


class TTSJobQueue:
    """Hàng đợi SQLite WAL + SHA-256 PCM disk cache cho LUMO Hub (ESP32)."""

    def __init__(self, cache_dir: Path | str | None = None):
        self.root = Path(cache_dir or app_settings.TTS_CACHE_DIR)
        self.cache_dir = self.root
        self.database = self.root / "speaker_jobs.sqlite3"
        self._lock = threading.RLock()
        self._enqueue_lock = threading.Lock()
        self._cache_lock = threading.Lock()
        self._initialized = False

    def _ensure_ready(self) -> None:
        if self._initialized and self.database.exists():
            return
        with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            with self._connect() as con:
                con.executescript(
                    """
                    PRAGMA journal_mode=WAL;
                    CREATE TABLE IF NOT EXISTS jobs (
                        seq INTEGER PRIMARY KEY AUTOINCREMENT,
                        id TEXT UNIQUE NOT NULL,
                        device TEXT NOT NULL,
                        request_key TEXT UNIQUE NOT NULL,
                        fingerprint TEXT NOT NULL,
                        audio TEXT NOT NULL,
                        audio_bytes INTEGER NOT NULL,
                        volume INTEGER NOT NULL,
                        priority INTEGER NOT NULL,
                        status TEXT NOT NULL,
                        created REAL NOT NULL,
                        lease TEXT,
                        expires REAL,
                        updated REAL NOT NULL,
                        text TEXT NOT NULL DEFAULT '',
                        audio_kind TEXT NOT NULL DEFAULT 'speech'
                    );
                    CREATE INDEX IF NOT EXISTS idx_tts_queue
                        ON jobs(device, status, priority DESC, seq);
                    CREATE TABLE IF NOT EXISTS app_settings (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    );
                    """
                )
            self._initialized = True

    @contextmanager
    def _connect(self):
        con = sqlite3.connect(self.database, timeout=10)
        con.row_factory = sqlite3.Row
        try:
            with con:
                yield con
        finally:
            con.close()

    def get_tts_settings(self) -> dict[str, Any]:
        self._ensure_ready()
        merged = _default_settings()
        with self._lock, self._connect() as con:
            row = con.execute("SELECT value FROM app_settings WHERE key='tts'").fetchone()
        if row:
            try:
                saved = json.loads(row[0])
                if isinstance(saved, dict):
                    merged.update(saved)
            except ValueError:
                pass
        return merged

    def public_tts_settings(self) -> dict[str, Any]:
        values = self.get_tts_settings()
        return {
            **{k: v for k, v in values.items() if k != "tts_key"},
            "has_api_key": bool(values.get("tts_key")),
        }

    def update_tts_settings(self, updates: dict[str, Any]) -> dict[str, Any]:
        self._ensure_ready()
        with self._lock, self._connect() as con:
            current = self.get_tts_settings()
            clear_key = bool(updates.get("clear_api_key", False))
            new_key = updates.get("api_key")
            cleaned = {k: v for k, v in updates.items() if k not in {"api_key", "clear_api_key"}}
            merged = {**current, **cleaned}
            merged["tts_key"] = "" if clear_key else (new_key or current.get("tts_key", ""))
            merged["revision"] = str(time.time_ns())
            con.execute(
                "INSERT INTO app_settings(key,value) VALUES ('tts',?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (json.dumps(merged, ensure_ascii=False),),
            )
        return self.public_tts_settings()

    def get_effective_settings(
        self,
        override_settings: dict[str, Any] | None = None,
        voice: str | None = None,
    ) -> dict[str, Any]:
        """Gộp cấu hình mặc định, cấu hình đã lưu trong SQLite của queue và tham số gọi."""
        merged = self.get_tts_settings() if override_settings is None else {**_default_settings(), **override_settings}
        if voice:
            merged["voice"] = voice
        return merged

    def cached_pcm(
        self,
        text: str,
        voice: str | None = None,
        override_settings: dict[str, Any] | None = None,
    ) -> tuple[Path, bytes]:
        self._ensure_ready()
        cfg = self.get_effective_settings(override_settings, voice=voice)
        digest = _cache_digest(text, cfg)
        audio_path = self.cache_dir / f"{digest}.pcm"
        with self._cache_lock:
            if audio_path.exists():
                return audio_path, audio_path.read_bytes()
            pcm = synthesize_pcm(text, voice=voice, override_settings=cfg)
            temp_path = self.cache_dir / f"{digest}.tmp"
            temp_path.write_bytes(pcm)
            temp_path.replace(audio_path)
            return audio_path, pcm

    def enqueue(
        self,
        device_id: str,
        text: str,
        *,
        volume: int = 80,
        priority: int = 0,
        voice: str | None = None,
        idempotency_key: str | None = None,
        claim_immediately: bool = False,
    ) -> dict[str, Any]:
        clean_text = text.strip()
        if not clean_text:
            raise HTTPException(422, "Text cannot be empty")
        if len(clean_text) > MAX_TEXT_LENGTH:
            raise HTTPException(422, f"Text exceeds {MAX_TEXT_LENGTH} characters")

        self._ensure_ready()
        idem = idempotency_key or secrets.token_hex(16)
        fingerprint_payload = json.dumps(
            {
                "device_id": device_id,
                "text": clean_text,
                "volume": volume,
                "priority": priority,
                "voice": voice or "",
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        fingerprint = hashlib.sha256(fingerprint_payload.encode("utf-8")).hexdigest()
        req_key = hashlib.sha256(f"{device_id}:{idem}".encode("utf-8")).hexdigest()

        with self._enqueue_lock:
            with self._lock, self._connect() as con:
                existing = con.execute(
                    "SELECT * FROM jobs WHERE request_key=?", (req_key,)
                ).fetchone()
                if existing:
                    if existing["fingerprint"] != fingerprint:
                        raise HTTPException(409, "Idempotency key reused with a different payload")
                    return {
                        "id": existing["id"],
                        "lease": existing["lease"],
                        "volume": existing["volume"],
                        "audio_bytes": existing["audio_bytes"],
                        "sample_rate": RATE,
                        "status": existing["status"],
                        "duplicate": True,
                    }
                now = time.time()
                con.execute(
                    "UPDATE jobs SET status='failed', updated=? WHERE status='playing' AND expires<?",
                    (now, now),
                )
                queued = con.execute(
                    "SELECT COUNT(*) FROM jobs WHERE device=? AND status IN ('queued','playing')",
                    (device_id,),
                ).fetchone()[0]
                if queued >= 20:
                    raise HTTPException(429, "Device queue is full")

            # Tạo / lấy cache ngoài queue lock để không chặn polling hoặc ACK của ESP32
            audio_path, pcm = self.cached_pcm(clean_text, voice=voice)
            cfg = self.get_effective_settings(voice=voice)
            audio_kind = "test_tone" if cfg.get("provider") == "tone" else "speech"
            job_id = secrets.token_hex(16)
            lease = secrets.token_hex(16) if claim_immediately else None
            now = time.time()
            expires = (now + DEVICE_LEASE_SECONDS) if claim_immediately else None
            status = "playing" if claim_immediately else "queued"

            with self._lock, self._connect() as con:
                con.execute(
                    """
                    INSERT INTO jobs(
                        id, device, request_key, fingerprint, audio, audio_bytes,
                        volume, priority, status, created, lease, expires, updated,
                        text, audio_kind
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        job_id,
                        device_id,
                        req_key,
                        fingerprint,
                        audio_path.name,
                        len(pcm),
                        int(volume),
                        int(priority),
                        status,
                        now,
                        lease,
                        expires,
                        now,
                        clean_text,
                        audio_kind,
                    ),
                )
            return {
                "id": job_id,
                "lease": lease,
                "volume": int(volume),
                "audio_bytes": len(pcm),
                "sample_rate": RATE,
                "status": status,
                "duplicate": False,
                "audio_kind": audio_kind,
            }

    def claim_next(self, device_id: str) -> dict[str, Any] | None:
        self._ensure_ready()
        with self._lock, self._connect() as con:
            now = time.time()
            con.execute(
                "UPDATE jobs SET status='failed', updated=? WHERE device=? AND status='playing' AND expires<?",
                (now, device_id, now),
            )
            if con.execute(
                "SELECT 1 FROM jobs WHERE device=? AND status='playing'",
                (device_id,),
            ).fetchone():
                return None
            job = con.execute(
                "SELECT * FROM jobs WHERE device=? AND status='queued' ORDER BY priority DESC, seq LIMIT 1",
                (device_id,),
            ).fetchone()
            if not job:
                return None
            lease = secrets.token_hex(16)
            con.execute(
                "UPDATE jobs SET status='playing', lease=?, expires=?, updated=? WHERE id=?",
                (lease, now + DEVICE_LEASE_SECONDS, now, job["id"]),
            )
            return {
                "id": job["id"],
                "lease": lease,
                "volume": job["volume"],
                "sample_rate": RATE,
                "audio_bytes": job["audio_bytes"],
                "lease_seconds": DEVICE_LEASE_SECONDS,
            }

    def get_job_for_streaming(self, device_id: str, job_id: str, lease: str) -> tuple[Path, int]:
        self._ensure_ready()
        with self._lock, self._connect() as con:
            job = con.execute(
                "SELECT * FROM jobs WHERE id=? AND device=?",
                (job_id, device_id),
            ).fetchone()
            if not job:
                raise HTTPException(404, "Job not found")
            if job["status"] != "playing" or job["lease"] != lease or (job["expires"] or 0) < time.time():
                raise HTTPException(409, "Lease invalid or expired")
            path = self.cache_dir / job["audio"]
            if not path.exists():
                raise HTTPException(410, "Audio cache missing")
            return path, int(job["audio_bytes"])

    def ack(self, device_id: str, job_id: str, lease: str, status: str) -> dict[str, Any]:
        self._ensure_ready()
        with self._lock, self._connect() as con:
            job = con.execute(
                "SELECT * FROM jobs WHERE id=? AND device=?",
                (job_id, device_id),
            ).fetchone()
            if not job:
                raise HTTPException(404, "Job not found")
            if job["lease"] != lease:
                raise HTTPException(409, "Stale lease")
            if job["status"] == status:
                return {"id": job_id, "status": status}
            if job["status"] != "playing" or (job["expires"] or 0) < time.time():
                raise HTTPException(409, "Job already completed or lease expired")
            con.execute(
                "UPDATE jobs SET status=?, updated=? WHERE id=?",
                (status, time.time(), job_id),
            )
            return {"id": job_id, "status": status}

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        self._ensure_ready()
        with self._lock, self._connect() as con:
            now = time.time()
            con.execute(
                "UPDATE jobs SET status='failed', updated=? WHERE id=? AND status='playing' AND expires<?",
                (now, job_id, now),
            )
            row = con.execute(
                "SELECT id, device, text, status, volume, priority, audio_bytes, audio_kind, created, updated FROM jobs WHERE id=?",
                (job_id,),
            ).fetchone()
            return dict(row) if row else None

    def list_jobs(self, device_id: str | None = None, limit: int = 30) -> list[dict[str, Any]]:
        self._ensure_ready()
        with self._lock, self._connect() as con:
            now = time.time()
            con.execute(
                "UPDATE jobs SET status='failed', updated=? WHERE status='playing' AND expires<?",
                (now, now),
            )
            if device_id:
                rows = con.execute(
                    "SELECT id, device, text, status, volume, priority, audio_bytes, audio_kind, created, updated "
                    "FROM jobs WHERE device=? ORDER BY seq DESC LIMIT ?",
                    (device_id, limit),
                ).fetchall()
            else:
                rows = con.execute(
                    "SELECT id, device, text, status, volume, priority, audio_bytes, audio_kind, created, updated "
                    "FROM jobs ORDER BY seq DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [dict(row) for row in rows]


tts_queue = TTSJobQueue()


def get_effective_settings(
    override_settings: dict[str, Any] | None = None,
    voice: str | None = None,
) -> dict[str, Any]:
    """Gộp cấu hình mặc định, cấu hình đã lưu trong SQLite và tham số gọi trực tiếp."""
    return tts_queue.get_effective_settings(override_settings=override_settings, voice=voice)


def cached_pcm(
    text: str,
    voice: str | None = None,
    override_settings: dict[str, Any] | None = None,
) -> tuple[Path, bytes]:
    return tts_queue.cached_pcm(text, voice=voice, override_settings=override_settings)
