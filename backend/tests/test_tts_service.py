import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import wave

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from app.routes import lumo as lumo_routes
from app.services import tts_service
from app.services.tts_service import (
    RATE,
    TTSJobQueue,
    normalize_audio,
    pcm_to_wav_bytes,
    provider_failure,
)


def _make_wav(samples: list[int], channels: int = 1, framerate: int = 24000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(framerate)
        wf.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    return buf.getvalue()


class TTSServiceTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.queue = TTSJobQueue(cache_dir=Path(self.tmpdir.name))

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_sha256_cache_hits_disk_without_repeating_synthesis(self):
        self.queue.update_tts_settings({"provider": "tone", "tts_key": "secret-a"})
        calls = []

        def fake_synth(text, voice=None, override_settings=None):
            calls.append(text)
            return b"\x01\x00" * 2400

        with patch.object(tts_service, "synthesize_pcm", side_effect=fake_synth):
            path1, pcm1 = self.queue.cached_pcm("Xin chào ông bà")
            path2, pcm2 = self.queue.cached_pcm("Xin chào ông bà")
            # Đổi secret key nhưng không đổi revision/config -> vẫn dùng cùng digest (không lộ key vào hash)
            _, pcm3 = self.queue.cached_pcm(
                "Xin chào ông bà",
                override_settings={"provider": "tone", "tts_key": "secret-b", "revision": self.queue.get_tts_settings()["revision"], "voice": "Kore", "model": "eleven_flash_v2_5", "format": "pcm", "tts_url": "", "body_template": tts_service.DEFAULT_BODY_TEMPLATE},
            )

        self.assertEqual(len(calls), 1)
        self.assertTrue(path1.exists())
        self.assertEqual(path1, path2)
        self.assertEqual(pcm1, pcm2)
        self.assertEqual(pcm1, pcm3)

    def test_http_adapter_renders_quotes_and_normalizes_noncanonical_wav(self):
        wav_16k_stereo = _make_wav([1000, -1000] * 1600, channels=2, framerate=16000)
        captured_json = {}

        fake_resp = MagicMock()
        fake_resp.status_code = 200
        fake_resp.headers = {"content-type": "audio/wav"}
        fake_resp.iter_bytes.return_value = [wav_16k_stereo]
        fake_resp.__enter__.return_value = fake_resp
        fake_resp.__exit__.return_value = False

        fake_client = MagicMock()
        fake_client.__enter__.return_value = fake_client
        fake_client.__exit__.return_value = False

        def fake_stream(method, url, headers=None, json=None):
            captured_json["method"] = method
            captured_json["url"] = url
            captured_json["headers"] = headers
            captured_json["json"] = json
            return fake_resp

        fake_client.stream.side_effect = fake_stream

        cfg = {
            "provider": "http",
            "tts_url": "https://tts.example.com/v1/speak",
            "tts_key": "top-secret-token",
            "auth_header": "xi-api-key",
            "auth_prefix": "",
            "voice": "voice-1",
            "model": "eleven_flash_v2_5",
            "format": "wav",
            "body_template": {
                "text": "{{text}}",
                "voice_id": "{{voice}}",
                "model_id": "{{model}}",
            },
        }
        with patch.object(tts_service.httpx, "Client", return_value=fake_client):
            pcm = tts_service.synthesize_pcm('Chào "ông", khỏe không?', override_settings=cfg)

        self.assertEqual(captured_json["headers"], {"xi-api-key": "top-secret-token"})
        self.assertEqual(captured_json["json"]["text"], 'Chào "ông", khỏe không?')
        self.assertEqual(captured_json["json"]["model_id"], "eleven_flash_v2_5")
        self.assertEqual(len(pcm) % 2, 0)
        self.assertGreater(len(pcm), 0)

    def test_provider_failure_hides_secrets_and_explains_error(self):
        resp = MagicMock()
        resp.status_code = 401
        resp.iter_bytes.return_value = [
            json.dumps({"detail": {"status": "invalid_api_key", "message": "sk-secret-do-not-leak"}}).encode()
        ]
        exc = provider_failure(resp)
        self.assertIsInstance(exc, HTTPException)
        self.assertEqual(exc.status_code, 502)
        self.assertNotIn("sk-secret-do-not-leak", exc.detail)
        self.assertIn("invalid_api_key", exc.detail)

    def test_pcm_to_wav_and_normalize_audio(self):
        pcm = b"\x00\x10\x00\xf0" * 100
        wav_data = pcm_to_wav_bytes(pcm, sample_rate=RATE)
        self.assertEqual(normalize_audio(wav_data, "wav"), pcm)

    def test_tts_preview_endpoint_returns_wav_without_auth(self):
        self.queue.update_tts_settings({"provider": "tone"})
        app = FastAPI()
        app.include_router(lumo_routes.router, prefix="/api/v1")
        client = TestClient(app)

        with patch.object(lumo_routes, "cached_pcm", side_effect=self.queue.cached_pcm):
            res = client.get("/api/v1/lumo/tts-preview?text=Xin%20chao&voice_name=Kore")

        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.headers["content-type"], "audio/wav")
        self.assertTrue(res.content.startswith(b"RIFF"))
        self.assertEqual(len(res.content), 44 + 48000)

    def test_needs_web_search_skips_tavily_for_normal_greetings(self):
        with patch.object(lumo_routes.settings, "TAVILY_API_KEY", "test-tavily-key"), \
             patch.object(lumo_routes.settings, "LUMO_ENABLE_WEB_SEARCH", False):
            self.assertFalse(lumo_routes._needs_web_search("Chào Lumo buổi sáng"))
            self.assertTrue(lumo_routes._needs_web_search("Thời tiết Hà Nội hôm nay thế nào"))


if __name__ == "__main__":
    unittest.main()
