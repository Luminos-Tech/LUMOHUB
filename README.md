# LUMO Hub

LUMO Hub is an assistive-technology project by Luminos Tech. The product vision is an AI companion ecosystem for older adults and people with disabilities: simple voice interaction, calendar and reminder support, family connection, and privacy-conscious safety monitoring without cameras.

This repository contains the current software platform and an ESP32 Hub prototype. The project proposal also describes future hardware, wearable fall detection, private AI infrastructure, care-organization pilots, and insurance distribution. Those proposal items are marked as roadmap items below and must not be presented as completed features.

## Current Status

Implemented in this repository:

- FastAPI REST API with PostgreSQL/SQLAlchemy models, Alembic migrations, JWT authentication, and admin operations.
- Events, calendar views, reminders, in-app notifications, Web Push subscriptions, devices, and physical event-button records.
- AI text/audio endpoints and LUMO WebSockets for text, STT, LLM, and TTS flows.
- Next.js dashboard for calendar, events, notifications, devices, settings, and administration.
- ESP-IDF firmware for both prototypes:
  - `firmware/lumo-hub/`: ESP32-S3 Hub (`v0.6.0-dsoft-stream-tts`) with Wi-Fi, I2S mic/audio streaming, physical button check-in, NimBLE Central (`0x02` telemetry + `0x03` 10 Hz IMU stream), and an on-device 3-model IMU TinyML ensemble (`Compact RF-25 + LumoEventMLP + LumoBurst1DCNN`) to suppress wrist false alerts.
  - `firmware/lumo-band/`: ESP32-C3 + MPU6050 BLE wearable (`v0.4.0-mpu-burst-ai`) featuring a fall state machine (`IDLE -> IMPACT_WAIT -> FALL_CONFIRMED`), high-rate `10 Hz` raw IMU burst streaming (`0x03`) upon impact (`|a| >= 2.5g` or `|a| < 0.55g`), and an edge self-test copy of the 3-model TinyML engine.
- `ml_pipeline/`: Dataset builder (combining real SisFall, Kaggle ADL, and 3D biomechanics), training, stress-testing, Hugging Face publishing, and zero-dependency C99 weight export (`Compact RF-25`, `LumoEventMLP`, `LumoBurst1DCNN`).

Roadmap or proposal-level capabilities include the production 4G Hub, nRF52840 + MPU6050 BLE fall-detection wearable, activity trend analytics, abnormal-silence escalation, self-hosted private LLM infrastructure, clinical validation, and insurance integration.

## Product Direction

The intended experience is voice-first and accessible: an older adult should not need to learn a complex application or carry a smartphone for the core interaction. A physical check-in button, scheduled family voice recordings, reminders, and conversational prompts are designed to support independence and reduce isolation.

The proposal's planned wearable uses a six-axis IMU and a three-phase fall-detection algorithm (free fall -> impact -> prolonged immobility). Its targets are BLE 5.0, a CR2450 battery lasting one to three years, IP67 protection, and a lightweight non-medical design. These are targets requiring hardware validation, pilot data, certification, and production testing.

The proposal positions the product for B2C, B2B, and B2B2C distribution through families, nursing homes, disability support organizations, and insurance add-on services. The proposal prices are planning assumptions, not audited commercial commitments.

## Architecture

```text
ESP32 LUMO Hub / future wearable
          | WebSocket, HTTP, BLE (roadmap)
          v
FastAPI backend :8000 ---- PostgreSQL :5432
          |
          +-- REST API (/api/v1)
          +-- LUMO AI and audio pipelines
          +-- WebSockets (/ws/lumo, /ws/stream)
          +-- APScheduler reminder/session jobs
          +-- Web Push and device delivery
          ^
Next.js frontend :3000
  same-origin /api/v1 proxy -> INTERNAL_API_URL
```

Backend entry point: `backend/app/main.py` (`app.main:app`). The application creates database tables and starts APScheduler in its lifespan.

Firmware entry point: `firmware/lumo-hub/main/Lumo-LuminosTech.c` with modules under `main/{wifi,http_api,mic,record,audio,ble,device_control,button}`. Band wearable firmware is under `firmware/lumo-band/`.

## Repository Layout

```text
backend/                         FastAPI application
  app/routes/                    REST routes
  app/models/                    SQLAlchemy models
  app/schemas/                   Pydantic schemas
  app/crud/                      Database operations
  app/tasks/scheduler.py        Reminder and session jobs
  app/websocket/                 Device and TTS WebSockets
  alembic/                       Database migrations
  tests/                         Backend unit tests
frontend-web/                    Next.js application
  src/app/                       App Router pages and proxy routes
  src/features/                  API modules
  src/store/                     Zustand stores
  src/components/                UI components
firmware/
  lumo-hub/                      ESP32-S3 Hub ESP-IDF firmware (+ 3-model IMU TinyML)
  lumo-band/                     ESP32-C3 + MPU6050 BLE wearable ESP-IDF firmware
ml_pipeline/                     5-class IMU TinyML dataset, training, C99 export & HF tools
docs/                            Guides, hardware-flow diagrams & PDF, engineering notes
scripts/                         Diagnostic, audio test, QR, and serial utility scripts
docker-compose.yml               Local development stack (PostgreSQL, FastAPI, Next.js)
AGENTS.md                        Agent and architecture guidance
LICENSE                          Apache License 2.0
```

## On-Device IMU TinyML Fall & False-Alert Engine

Both the `lumo-hub` (ESP32-S3 production inference host) and `lumo-band` (ESP32-C3 diagnostic/edge prototype) run a zero-dependency C99 ensemble ([`lumo_imu_ai.c`](firmware/lumo-hub/main/ble/lumo_imu_ai.c)) trained on a 30,000-window 5-class dataset (`3.0s` windows = `30 samples x 6 IMU channels` @ `10 Hz`) to eliminate everyday false alerts (clapping, table slap, martial-arts hand strikes, or taking off and dropping the band) while preserving fall sensitivity:

1. **`0: NORMAL_ADL`** — Walking, sitting, everyday motion
2. **`1: WRIST_SHOCK_FALSE_ALERT`** — High-G wrist shock without body collapse or prolonged immobility
3. **`2: DEVICE_DROP_OFF_WRIST`** — Free-fall/drop onto a rigid surface with zero post-impact physiological micro-tremor
4. **`3: SOFT_FALL_RECOVERING`** — Fall/stumble followed by self-recovery motion (prompts Hub voice check-in)
5. **`4: CRITICAL_FALL_IMMOBILE`** — High-G fall with torso/wrist orientation tilt (`~85 deg`) and prolonged immobility (confirmed emergency alert)

### Ensemble Architecture & Measured Benchmarks (`0.20 * CNN + 0.40 * MLP + 0.40 * RF`)

| Model | Input & Footprint | Held-out Test (`N=4,500`) | Wrist / Drop FAR | ESP32-S3 Latency (`COM10`) | ESP32-C3 Latency (`COM11`) |
| --- | --- | :---: | :---: | :---: | :---: |
| **Compact RF-25** (`lumo_hub_rf_model.h`) | 16 physics features, ~62.7 KB (`25` trees) | **99.87%** | **0.00% / 0.00%** | **0.43 – 1.03 ms** | **0.56 – 1.18 ms** |
| **LumoEventMLP** (`lumo_hub_feature_model.h`) | 16 physics features, `13.5 KB` (`3,232` MACs) | **99.73%** | **0.00% / 0.00%** | **0.82 – 0.97 ms** | **4.59 – 4.80 ms** |
| **LumoBurst1DCNN** (`lumo_hub_cnn_model.h`) | Raw `[30 x 6]` window, `21.3 KB` (`58,368` MACs) | **98.38%** | 0.11% / 0.00% | **13.6 – 13.8 ms** | **82.4 – 84.3 ms** |

### BLE Dual-Packet Architecture (`lumo-band` -> `lumo-hub`)

- **`0x02` (12 bytes @ 1 Hz):** Compact telemetry packet (`version`, `state`, `hr`, `steps`, `battery`, plus optional Band edge AI class/confidence tags).
- **`0x03` (20 bytes @ 0.5 Hz idle -> 10 Hz burst for 5s on `|a| >= 2.5g`):** Raw calibrated 6-axis `MPU6050` payload streamed over BLE to the ESP32-S3 Hub, which buffers `1.8s` post-impact (`18` samples + `12` pre-impact samples = `3.0s`) and runs the 3-model ensemble.
- **Hugging Face Models & Dataset:**
  - Approach 1 (`1D-CNN`): https://huggingface.co/fragilov/lumohub-imu-burst-1dcnn
  - Approach 2 (`Physics-MLP` + `lumo_hub_rf_model.h`): https://huggingface.co/fragilov/lumohub-imu-event-mlp
  - 5-Class Benchmark Dataset: https://huggingface.co/datasets/fragilov/lumohub-imu-fall-5class-dataset

## Backend API

All REST routes use the `/api/v1` prefix and Bearer JWT authentication unless noted.

| Area | Routes |
| --- | --- |
| Auth | `POST /auth/register`, `/auth/login`, `/auth/refresh`, `/auth/logout`; `GET /auth/me` |
| Users | `GET/PATCH /users/me`; `PATCH /users/me/password` |
| Events | CRUD `/events`; `GET /calendar`; `POST /events/extract` |
| Reminders | `GET /reminders`; CRUD reminder routes under `/events/{event_id}/reminders` and `/reminders/{reminder_id}` |
| Notifications | `GET /notifications`; read-one and read-all patches |
| Devices | CRUD `/devices`; `GET /devices/qr` |
| Event buttons | `POST/GET /event-buttons`; `GET /event-buttons/today` |
| Push | `/push/public-key`, subscribe/unsubscribe, send, and status |
| Admin | User, password, lock, role, logs, events, and device commands under `/admin` |
| LUMO | `/lumo/`, `version1`, `version2`, `version3`, wake-word verification, and `/lumo/audio/` |
| Check notifications | CRUD `/check-noti` |

Health check: `GET /health` returns `{"status":"ok","service":"LumoHub API"}`.

## Reminder Flow

1. A user creates an event and optional reminder.
2. `check_reminders` runs every minute and evaluates `event.start_time - remind_before_minutes`.
3. A triggered reminder creates a notification and is marked sent.
4. `lumo` reminders are forwarded through the WebSocket manager to connected devices.
5. `cleanup_expired_sessions` removes expired refresh sessions hourly.

## LUMO WebSocket Protocol

Basic device connection:

```text
ws://localhost:8000/ws/lumo?device_id=<device-id>
```

Audio/TTS stream:

```text
ws://localhost:8000/ws/stream?device_id=<device-id>
```

The stream accepts JSON text or binary UTF-8 frames:

```json
{"action":"tts","text":"Your reminder text"}
{"action":"stt_tts","audio_b64":"<base64 WAV>"}
```

Responses include binary WAV audio and JSON messages such as `{"type":"done"}` or `{"type":"error","message":"..."}`. The current implementation uses a 24 kHz mono 16-bit WAV output for TTS.

## Technology

Backend: Python, FastAPI 0.111, Uvicorn, SQLAlchemy 2, PostgreSQL 16, Alembic, Pydantic 2, JWT, bcrypt, APScheduler, Google GenAI, Groq, HTTPX, pywebpush, Pillow, and qrcode.

Frontend: Next.js 14, React 18, TypeScript, Tailwind CSS, Zustand, Axios, FullCalendar, Zod, react-hook-form, Sonner, Lucide React, date-fns, and html5-qrcode.

Firmware: ESP-IDF/C, FreeRTOS, Wi-Fi, HTTP client/server, I2S audio, OLED, GPIO buttons, SNTP, and cJSON. The planned wearable is based on nRF52840 + MPU6050, but it is not part of the current firmware tree.

## Configuration

Backend variables are documented in `backend/.env.example`:

```env
DATABASE_URL=postgresql://lumohub:lumohub123@localhost:5432/lumohub_db
SECRET_KEY=replace-with-a-long-random-secret
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7
CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
GEMINI_API_KEY=
GROQ_API_KEY=
PERPLEXITY_API_KEY=
GOOGLE_API_KEY=
TAVILY_API_KEY=
VAPID_PUBLIC_KEY=
VAPID_PRIVATE_KEY=
VAPID_SUBJECT=mailto:admin@example.com
```

The frontend proxy uses `INTERNAL_API_URL`; see `frontend-web/.env.local.example`. WebSocket clients use `NEXT_PUBLIC_WS_URL` when configured.

Never commit `.env` files or real credentials.

## Running with Docker

```bash
docker compose up --build
```

Open `http://localhost:3000`, backend docs at `http://localhost:8000/docs`, and health at `http://localhost:8000/health`.

## Running Without Docker

Backend (PowerShell):

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
uvicorn app.main:app --reload --port 8000
```

Frontend:

```powershell
cd frontend-web
npm ci
Copy-Item .env.local.example .env.local
npm run dev
```

PostgreSQL must be available and `DATABASE_URL` must point to it. The application creates tables at startup; use Alembic explicitly when applying migrations to an existing database.

## Validation

```powershell
python -m compileall -q backend\app
cd frontend-web
npm run lint
npm run build
```

Firmware requires an installed ESP-IDF toolchain and the correct board configuration. API-dependent AI, Web Push, database, and device tests require their respective services and credentials.

## Security and Product Limits

- Passwords are hashed with bcrypt; JWT access and refresh sessions are stored and validated by the backend.
- Admin routes require admin authorization; user routes are scoped to the current user.
- Pydantic validation, SQLAlchemy parameters, and configured CORS provide baseline protections, not a complete security review.
- Current AI routes call external providers. Do not describe the current deployment as fully self-hosted or third-party-free.
- LUMO Hub is a prototype and is not currently a certified medical device, emergency-response service, or diagnostic system.
- Any fall-detection accuracy, battery life, IP67 rating, cognitive analysis, or insurance savings claim requires measured evidence and appropriate regulatory review.

## MCP and Agent Setup

See [docs/MCP_SETUP.md](docs/MCP_SETUP.md) for the project-local MCP configuration and progressive memory files. See [AGENTS.md](AGENTS.md) for source-verified architecture, roadmap boundaries, and coding-agent rules.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
