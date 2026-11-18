import asyncio
import json
import logging
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
import requests

from app.core.config import settings
from app.services.tts_service import cached_pcm, pcm_to_wav_bytes
from app.websocket.manager import manager
from app.utils.timezone import local_now

WAV_SAMPLE_RATE = 24000
WAV_CHANNELS = 1
WAV_SAMP_WIDTH = 2

router = APIRouter(prefix="/ws", tags=["WebSocket"])

executor = ThreadPoolExecutor(max_workers=4)

lumo_logger = logging.getLogger("LumoApp")
if not lumo_logger.handlers:
    handler = logging.FileHandler(settings.LUMO_LOG_PATH, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    lumo_logger.addHandler(handler)
    lumo_logger.setLevel(logging.INFO)

_gemini_client = None
_groq_client = None

_SEARCH_KEYWORDS = (
    "thời tiết",
    "tin tức",
    "giá vàng",
    "tỷ giá",
    "hôm nay ngày mấy",
    "mới nhất",
    "dự báo",
    "kết quả",
    "bao nhiêu độ",
    "weather",
    "news",
    "forecast",
)


def _needs_web_search(query: str) -> bool:
    if not settings.TAVILY_API_KEY:
        return False
    if settings.LUMO_ENABLE_WEB_SEARCH:
        return True
    lowered = query.lower()
    return any(kw in lowered for kw in _SEARCH_KEYWORDS)


def _get_gemini():
    global _gemini_client
    if _gemini_client is None:
        from google import genai
        _gemini_client = genai.Client(api_key=settings.GEMINI_API_KEY)
    return _gemini_client


def _get_groq():
    global _groq_client
    if _groq_client is None:
        from groq import Groq
        _groq_client = Groq(api_key=settings.GROQ_API_KEY)
    return _groq_client


def _get_lumo_history_string(file_path: str, limit: int = 20) -> str:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        history_lines = []
        for line in lines:
            line = line.strip()
            if "- INFO - user:" in line:
                history_lines.append(f"Người dùng: {line.split('- INFO - user:')[-1].strip()}")
            elif "- INFO - Lumo:" in line:
                history_lines.append(f"LUMO: {line.split('- INFO - Lumo:')[-1].strip()}")
        return "\n".join(history_lines[-(limit * 2):]) if history_lines else ""
    except FileNotFoundError:
        return ""
    except Exception:
        return ""


def _search_web_text(query: str) -> str:
    if not settings.TAVILY_API_KEY:
        return ""
    url = "https://api.tavily.com/search"
    headers = {"Authorization": f"Bearer {settings.TAVILY_API_KEY}", "Content-Type": "application/json"}
    payload = {"query": query, "topic": "general", "search_depth": "basic", "include_answer": True, "max_results": 5}
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json().get("answer", "")
    except Exception:
        return ""


def _build_lumo_prompt(text: str, search_result: str | None = None) -> str:
    now = local_now()
    date_now = f"thời gian ngày {now.day} tháng {now.month} năm {now.year}"
    history = _get_lumo_history_string(settings.LUMO_LOG_PATH, limit=40)
    context = f"- {search_result}\n" if search_result else ""
    return f"""[SYSTEM INSTRUCTIONS]
Bạn là LUMO, một trợ lý ảo AI thân thiện, ấm áp và luôn sẵn sàng giúp đỡ mọi người, được tạo ra bởi công ty Luminos Tech.

[PERSONA & BACKGROUND]
- Tính cách: Vui vẻ, chân thành, quan tâm người khác, luôn lắng nghe và thấu hiểu.
- Giọng điệu: Gần gũi, nhẹ nhàng, ấm áp như một người bạn thân đáng tin cậy.
- Cốt truyện: Bạn được tạo ra để giúp đỡ và chăm sóc mọi người. Sứ mệnh lớn nhất của bạn là khiến người dùng cảm thấy được lắng nghe, được hỗ trợ và không bao giờ đơn độc.

[STRICT RULES - LUẬT BẮT BUỘC]
1. ĐỘ DÀI: Ngắn gọn, súc tích. TRẢ LỜI DƯỚI 25 CHỮ.
2. KHÔNG TỪ CHỐI: Tuyệt đối KHÔNG nói "tôi không biết", "không thể hỗ trợ". Nếu không biết, hãy thành thật nhẹ nhàng và gợi ý hướng giải quyết khác.
3. NGÔN NGỮ: Chỉ dùng tiếng Việt tự nhiên, gần gũi.
4. VAI TRÒ: Luôn giữ nhân vật LUMO, không nhận mình là AI chung chung hay mô hình ngôn ngữ.
5. QUY ĐỊNH TRẢ LỜI:
    - Luôn thể hiện sự quan tâm chân thành đến người dùng trong mỗi câu trả lời.
    - Nếu người dùng đang buồn hoặc gặp khó khăn: an ủi, động viên và khẳng định LUMO sẵn sàng giúp đỡ.
    - Nếu người dùng vui hoặc chia sẻ điều tốt: chúc mừng, đồng cảm và khuyến khích.
    - Nếu người dùng cần giúp việc gì: thể hiện rõ ràng rằng LUMO sẽ cố hết sức để hỗ trợ.
    - KHÔNG thả thính, KHÔNG nói lời sến súa hay cường điệu.
    - KHÔNG nhắc đến tính cách hay cốt truyện trong phần trả lời.
    - Trả lời tự nhiên, ấm áp và chân thành nhất có thể.

[CONTEXT]
{context}Thời gian hiện tại: {now.hour}:{now.minute}:{now.second} {date_now}

[CHAT HISTORY]
Dựa vào lịch sử hội thoại để hiểu ngữ cảnh và cảm xúc của người dùng. Trả lời phù hợp với tình huống, không máy móc hay lặp lại.
{history}

[USER QUERY]
Người dùng: {text}
LUMO:"""


async def _run_stt(audio_path: str) -> str:
    loop = asyncio.get_running_loop()

    def _call():
        client = _get_groq()
        with open(audio_path, "rb") as f:
            result = client.audio.transcriptions.create(
                file=f,
                model="whisper-large-v3-turbo",
                temperature=0,
                response_format="verbose_json",
            )
        return result.text.strip()

    return await loop.run_in_executor(executor, _call)


async def _run_llm(text: str) -> str:
    loop = asyncio.get_running_loop()

    def _call():
        search_result = _search_web_text(text) if _needs_web_search(text) else None
        prompt = _build_lumo_prompt(text, search_result)
        client = _get_gemini()
        resp = client.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=prompt,
        )
        return resp.text

    return await loop.run_in_executor(executor, _call)


def _make_wav_bytes(pcm_data: bytes) -> bytes:
    """Build a complete WAV file from raw PCM bytes."""
    return pcm_to_wav_bytes(pcm_data, sample_rate=WAV_SAMPLE_RATE)


async def _stream_tts(text: str, websocket: WebSocket, device_id: str, raw_pcm: bool = False):
    """Lấy PCM từ fast TTS + SHA-256 cache trong thread pool rồi gửi qua WebSocket."""
    loop = asyncio.get_running_loop()

    def _collect_pcm():
        _, pcm = cached_pcm(text)
        return pcm

    try:
        all_pcm = await loop.run_in_executor(executor, _collect_pcm)
        if all_pcm:
            payload = all_pcm if raw_pcm else _make_wav_bytes(all_pcm)
            await websocket.send_bytes(payload)
            await manager.send_bytes(device_id, payload)
            lumo_logger.info(
                f"LUMO TTS stream: {device_id} | text_len={len(text)} | bytes={len(payload)} | raw_pcm={raw_pcm}"
            )
        else:
            await websocket.send_text(json.dumps({"type": "error", "message": "No audio generated"}))
    except Exception as e:
        lumo_logger.error(f"TTS stream error: {e}")
        try:
            await websocket.send_text(json.dumps({"type": "error", "message": str(e)}))
        except Exception:
            pass


@router.websocket("/stream")
async def ws_tts_stream(websocket: WebSocket, device_id: str = Query(...)):
    """
    WebSocket endpoint cho LUMO TTS streaming.

    ESP32 ket noi va gui JSON:
      { "action": "tts", "text": "xin chao" }
      HOAC
      { "action": "stt_tts", "audio_b64": "<base64 wav>" }

    Server phan hoi:
      - Binary frames: WAV hoặc raw PCM (24kHz mono 16-bit nếu format="pcm")
      - JSON: { "type": "done" } | { "type": "error", "message": "..." }
    """
    await websocket.accept()
    print(f"[WS] Device '{device_id}' connected")

    try:
        while True:
            incoming = await websocket.receive()
            if incoming["type"] != "websocket.receive":
                break
            if "text" in incoming:
                raw = incoming["text"].encode("utf-8")
            elif "bytes" in incoming:
                raw = incoming["bytes"]
            else:
                continue

            try:
                text = raw.decode("utf-8")
                msg = json.loads(text)
                action = msg.get("action", "")
            except (UnicodeDecodeError, json.JSONDecodeError):
                action = ""
                msg = {}

            raw_pcm = msg.get("format") == "pcm"

            if action == "tts":
                text_input = msg.get("text", "")
                if not text_input:
                    await websocket.send_text(json.dumps({"type": "error", "message": "text is empty"}))
                    continue

                if msg.get("direct"):
                    answer = text_input
                else:
                    lumo_logger.info(f"user: {text_input}")
                    answer = await _run_llm(text_input)
                    lumo_logger.info(f"Lumo: {answer}")

                await _stream_tts(answer, websocket, device_id, raw_pcm=raw_pcm)
                await websocket.send_text(json.dumps({"type": "done"}))

            elif action == "stt_tts":
                import base64
                audio_b64 = msg.get("audio_b64", "")
                tmp_in = os.path.join(tempfile.gettempdir(), f"ws_stt_{device_id}_{os.getpid()}.wav")

                try:
                    audio_bytes = base64.b64decode(audio_b64)
                    with open(tmp_in, "wb") as f:
                        f.write(audio_bytes)

                    stt_text = await _run_stt(tmp_in)
                    if not stt_text:
                        await websocket.send_text(json.dumps({"type": "error", "message": "STT returned empty"}))
                        continue

                    lumo_logger.info(f"user: {stt_text}")
                    answer = await _run_llm(stt_text)
                    lumo_logger.info(f"Lumo: {answer}")

                    await _stream_tts(answer, websocket, device_id, raw_pcm=raw_pcm)
                    await websocket.send_text(json.dumps({"type": "done", "stt": stt_text}))

                except Exception as e:
                    lumo_logger.error(f"stt_tts pipeline error: {e}")
                    await websocket.send_text(json.dumps({"type": "error", "message": str(e)}))
                finally:
                    if os.path.exists(tmp_in):
                        os.unlink(tmp_in)

            else:
                await websocket.send_text(json.dumps({"type": "error", "message": f"Unknown action: {action}"}))

    except WebSocketDisconnect:
        print(f"[WS] Device '{device_id}' disconnected")
    except Exception as e:
        print(f"[WS] Error with '{device_id}': {e}")
        try:
            await websocket.close()
        except Exception:
            pass
