# 🛡️ VoiceShield — AI Voice Fraud & Deepfake Detection System

[![Python Version](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-61DAFB.svg)](https://react.dev/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Real-time voice fraud and deepfake detection powered by SpeechBrain **AASIST** (Anti-Spoofing using Integrated Spectro-Temporal features) and **ECAPA-TDNN** speaker embeddings. VoiceShield implements a modular 6-stage machine learning pipeline coupled with a high-throughput asynchronous FastAPI backend and a responsive React monitoring dashboard.

---

## ✨ Features

- **Real-Time Streaming Voice Analysis**: Stream raw PCM / WAV chunks over low-latency WebSockets with instantaneous intermediate visual feedback.
- **6-Stage Modular ML Pipeline**: Strict stage boundaries from ingestion, preprocessing, and acoustic feature extraction to deepfake detection, speaker verification, and multi-factor risk assessment.
- **State-of-the-Art Deepfake Detection**: Pre-trained AASIST neural architecture targeting synthetic speech, text-to-speech (TTS), and voice conversion artifacts, with robust acoustic heuristic fallbacks.
- **Biometric Speaker Verification**: ECAPA-TDNN 192-dimensional speaker embeddings with cosine distance calculation against enrolled reference voices.
- **Comprehensive Acoustic Feature Profiling**: Real-time extraction of MFCCs, Mel-spectrograms, fundamental pitch ($F_0$), spectral centroid, spectral rolloff, zero-crossing rate (ZCR), chroma, and RMS energy.
- **Dynamic Composite Risk Engine**: Weighted multi-factor risk scoring engine classifying incoming audio into `Low`, `Medium`, and `High` fraud probabilities.
- **Auditable PostgreSQL Storage**: Complete logging of detection verdicts, confidence metrics, acoustic metadata, and speaker samples with asynchronous database persistence.
- **Modern Monitoring Dashboard**: React 18 interface with interactive waveform visualizers, real-time risk gauges, acoustic spectrogram graphs, and historical audit logs.

---

## 🏗️ Architecture

VoiceShield processes live and uploaded audio through a deterministic 6-stage pipeline:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                VoiceShield Engine                               │
└─────────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────┐
│ Stage 1: Capture │ ──> Ingest PCM stream / WAV upload into AudioData object
└──────────────────┘
        │
        ▼
┌──────────────────────┐
│ Stage 2: Preprocess  │ ──> Resample (16kHz), mono conversion, trim silence, RMS norm
└──────────────────────┘
        │
        ▼
┌───────────────────────────┐
│ Stage 3: Feature Extract  │ ──> MFCC, Mel-spec, Pitch, Spectral Centroid, Rolloff, ZCR
└───────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│ Stage 4: Deepfake Detect│ ──> SpeechBrain AASIST model (synthetic vs real)
└─────────────────────────┘
        │
        ▼
┌────────────────────────────┐
│ Stage 5: Speaker Verify    │ ──> ECAPA-TDNN embeddings & cosine similarity check
└────────────────────────────┘
        │
        ▼
┌────────────────────────────┐
│ Stage 6: Risk & Alerting   │ ──> Composite risk formula -> Low / Medium / High alert
└────────────────────────────┘
        │
        ├────────────────────────────────┬───────────────────────────────┐
        ▼                                ▼                               ▼
┌────────────────┐             ┌────────────────────┐          ┌───────────────────┐
│ WebSocket Push │             │ PostgreSQL Audit   │          │ React UI / Alerts │
└────────────────┘             └────────────────────┘          └───────────────────┘
```

### Stage Summary

| Stage | Name | Input | Output | Primary Technology |
|---|---|---|---|---|
| **1** | Audio Capture | Binary PCM / File Upload | `AudioData` | NumPy, SoundFile |
| **2** | Preprocessing | `AudioData` | Cleaned `AudioData` | Librosa, NumPy |
| **3** | Feature Extraction | Cleaned `AudioData` | `FeatureSet` | Librosa, SciPy |
| **4** | Deepfake Detection | Preprocessed Audio | `DetectionOutput` | SpeechBrain AASIST / PyTorch |
| **5** | Speaker Verification | Preprocessed Audio + Enrolled ID | `VerificationOutput` | SpeechBrain ECAPA-TDNN |
| **6** | Risk Scoring | `DetectionOutput` + `VerificationOutput` | `RiskResult` | Weighted Risk Engine |

---

## 📦 Tech Stack

| Category | Component | Technologies |
|---|---|---|
| **Backend** | API & Processing | Python 3.11, FastAPI, Uvicorn, Pydantic v2 |
| **Database & ORM** | Storage & Migrations | PostgreSQL 16, SQLAlchemy 2.0 (Async), asyncpg, Alembic |
| **Machine Learning** | Audio & Deep Learning | PyTorch, SpeechBrain (AASIST, ECAPA-TDNN), Librosa, SoundFile |
| **Frontend** | User Interface | React 18, Vite, Tailwind CSS, Lucide Icons, Recharts |
| **Infrastructure** | Containerization | Docker, Docker Compose, Alpine Linux |

---

## 🚀 Quick Start

### Prerequisites
- [Python 3.11+](https://www.python.org/downloads/)
- [Node.js 20+](https://nodejs.org/) and npm
- [PostgreSQL 16](https://www.postgresql.org/) (or Docker Desktop)
- [FFmpeg](https://ffmpeg.org/) and `libsndfile` installed on host (if running outside Docker)

---

### Option A: Docker Compose (Recommended)

1. **Clone and enter the repository:**
   ```bash
   git clone https://github.com/your-org/voiceshieldAI.git
   cd voiceshieldAI
   ```

2. **Initialize environment configuration:**
   ```bash
   cp .env.example .env
   ```

3. **Launch all services:**
   ```bash
   docker-compose up --build -d
   ```

4. **Verify deployment:**
   - **Frontend Dashboard**: [http://localhost:5173](http://localhost:5173)
   - **Backend API & Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
   - **Alternative ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

5. **Stop containers:**
   ```bash
   docker-compose down
   ```

---

### Option B: Manual Setup

#### 1. Database Setup
Ensure PostgreSQL is running, then create the database and user:
```bash
# Connect as administrative postgres user
psql -U postgres -c "CREATE DATABASE voiceshield_db;"
psql -U postgres -c "CREATE USER voiceshield WITH PASSWORD 'voiceshield_pass';"
psql -U postgres -c "GRANT ALL PRIVILEGES ON DATABASE voiceshield_db TO voiceshield;"
```

#### 2. Backend Setup
```bash
# Navigate to the workspace root
cd voiceshieldAI

# Set up Python virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env to set your DATABASE_URL and SECRET_KEY

# Run database migrations
alembic upgrade head

# Start the FastAPI ASGI server
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

#### 3. Frontend Setup
In a new terminal window:
```bash
cd voiceshieldAI/frontend

# Install dependencies
npm install

# Start Vite dev server
npm run dev
```

Visit [http://localhost:5173](http://localhost:5173) in your browser.

---

## ⚙️ Configuration

Application settings are loaded with fallback hierarchy: **Environment Variables** $\rightarrow$ `config.yaml` $\rightarrow$ Defaults in `backend/config.py`.

### Configuration Parameters (`config.yaml` / `.env`)

```yaml
# Audio Ingestion & Processing Parameters
audio:
  sample_rate: 16000             # Target audio sampling rate (Hz)
  silence_threshold_db: -40.0    # dB threshold below peak to trim silence
  min_audio_length_sec: 1.0      # Minimum allowable duration for valid analysis
  max_audio_length_sec: 30.0     # Maximum audio window duration
  chunk_size_bytes: 4096         # Chunk buffer size for incoming audio stream

# Risk Thresholds
risk:
  low_threshold: 0.35            # Scores < 0.35 are classified as Low Risk
  medium_threshold: 0.65         # Scores 0.35 - 0.65 are classified as Medium Risk
  high_threshold: 0.85           # Scores > 0.85 trigger immediate High Risk alerts

# Machine Learning Models
models:
  detection_model: "speechbrain/asr-wav2vec2-commonvoice-en" # AASIST / deepfake model
  speaker_model: "speechbrain/spkrec-ecapa-voxceleb"          # ECAPA-TDNN verification model
  device: "cpu"                                              # "cpu" or "cuda"

# Storage & Security
storage:
  audio_storage_path: "audio_storage" # Directory for audio payloads
  models_cache_path: "models_cache"   # Directory for downloaded PyTorch checkpoints
```

---

## 🔌 API Reference

### REST Endpoints

| Method | Endpoint | Description | Request Body | Response |
|---|---|---|---|---|
| `POST` | `/api/v1/detect` | Run 6-stage deepfake detection on audio file | `multipart/form-data` (`file`, optional `user_id`) | JSON: verdict, risk score, confidence, acoustic features |
| `POST` | `/api/v1/enroll` | Enroll reference voice sample for biometric verification | `multipart/form-data` (`user_id`, `file`) | JSON: sample ID, embedding status, duration |
| `POST` | `/api/v1/verify` | Verify speaker identity against enrolled sample | `multipart/form-data` (`user_id`, `file`) | JSON: similarity score, `is_same_speaker`, confidence |
| `GET` | `/api/v1/results` | List historical detection logs with filters | Query params: `user_id`, `risk_level`, `limit`, `offset` | JSON array of detection records |
| `GET` | `/api/v1/results/{id}` | Fetch detailed audit log by UUID | Path param: `id` (UUID) | JSON object with full feature breakdown |
| `POST` | `/api/v1/users` | Register a new user profile | JSON: `{"username": "...", "email": "..."}` | JSON: created user record |
| `GET` | `/api/v1/users` | List all registered users | Query params: `limit`, `offset` | JSON array of users |
| `GET` | `/api/v1/health` | System health check (DB connectivity, ML model status) | None | JSON: `{"status": "healthy", ...}` |

---

### WebSocket Protocol (`/ws/audio/{session_id}`)

VoiceShield supports live bi-directional streaming over WebSockets:

1. **Establish Connection**:
   - Connect to `ws://localhost:8000/ws/audio/{session_id}`
   - Server returns:
     ```json
     {
       "type": "connected",
       "session_id": "session-1234",
       "timestamp": "2026-09-23T20:55:00Z",
       "data": { "status": "ready", "sample_rate": 16000 }
     }
     ```

2. **Send Audio Data**:
   - Stream binary 16-bit PCM (16 kHz, single channel) or WAV bytes in 4096-byte buffers.
   - Alternatively, send JSON control packets:
     ```json
     { "action": "stop", "user_id": "optional-user-uuid" }
     ```

3. **Receive Real-Time Events**:
   - **Processing Update**:
     ```json
     {
       "type": "processing",
       "session_id": "session-1234",
       "timestamp": "2026-09-23T20:55:02Z",
       "data": { "chunks_received": 12, "buffered_duration_sec": 3.0 }
     }
     ```
   - **Detection Result**:
     ```json
     {
       "type": "detection_result",
       "session_id": "session-1234",
       "timestamp": "2026-09-23T20:55:04Z",
       "data": {
         "risk_score": 0.89,
         "risk_level": "High",
         "verdict": "synthetic",
         "confidence": 0.94,
         "similarity": 0.32,
         "is_same_speaker": false,
         "model_used": "speechbrain/aasist"
       }
     }
     ```
   - **Error Handling**:
     ```json
     {
       "type": "error",
       "session_id": "session-1234",
       "timestamp": "2026-09-23T20:55:05Z",
       "data": { "error": "Audio duration is below the 1.0s minimum threshold" }
     }
     ```

---

## 🧠 ML Pipeline Details

### Stage 4: Deepfake Detection (AASIST)
- **Model**: Automated Audio Spoofing Integrated Spectro-Temporal Graph Attention Network (**AASIST**).
- **Functionality**: Processes raw audio waveforms using sinc-convolutional frontends and dual-branch graph neural networks (spectro and temporal branches). It extracts high-frequency anomalies, phase irregularities, and unnatural harmonic distributions characteristic of modern neural speech synthesizers (TTS) and voice conversion (VC) models.
- **Rule-Based Fallback**: If GPU/torch weights are unavailable or if lightweight mode is active, the engine activates heuristic acoustic scoring inspecting:
  - Spectral centroid variation ($< 250$ Hz threshold for robotic flattening)
  - Unnatural pitch variance ($\sigma(F_0) < 10$ Hz or discontinuous jumps)
  - Abnormally high zero-crossing rates in silence regions

### Stage 5: Speaker Verification (ECAPA-TDNN)
- **Model**: Emphasized Channel Attention, Propagation, and Aggregation Time Delay Neural Network (**ECAPA-TDNN**).
- **Functionality**: Extracts fixed-dimensional (192D) utterance embeddings from arbitrary-length preprocessed speech.
- **Verification Criterion**: Calculates the cosine similarity between the probe embedding $\vec{e}_{\text{probe}}$ and the enrolled baseline embedding $\vec{e}_{\text{enrolled}}$:
  $$\text{Cosine Similarity} = \frac{\vec{e}_{\text{probe}} \cdot \vec{e}_{\text{enrolled}}}{\|\vec{e}_{\text{probe}}\| \|\vec{e}_{\text{enrolled}}\|}$$
- Matches are classified as authentic when $\text{Cosine Similarity} \ge 0.70$.

### Risk Scoring Formula
The final fraud risk score combines acoustic anomaly indicators, synthetic detection probability, and speaker verification mismatch:

$$\text{Risk Score} = w_d \cdot P(\text{synthetic}) + w_s \cdot (1 - \text{Similarity}) + w_a \cdot A_{\text{acoustic}}$$

- **Weights**:
  - $w_d = 0.55$ (Deepfake detection weight)
  - $w_s = 0.35$ (Speaker verification mismatch weight)
  - $w_a = 0.10$ (Acoustic artifact score)
- **Classification Output**:
  - **$\text{Risk Score} < 0.35$**: `Low` — Audio is authentic and speaker matches enrolled profile.
  - **$0.35 \le \text{Risk Score} \le 0.85$**: `Medium` — Indeterminate patterns; flagged for secondary verification.
  - **$\text{Risk Score} > 0.85$**: `High` — Synthetic speech detected or critical speaker mismatch; fraud alert raised.

---

## 📊 Database Schema

VoiceShield utilizes PostgreSQL with standard relational integrity and JSONB support for fast feature retrieval:

```
┌─────────────────────────────────┐
│              users              │
├─────────────────────────────────┤
│ id (UUID, PK)                   │
│ username (VARCHAR(100), UNIQUE) │
│ email (VARCHAR(255), UNIQUE)    │
│ created_at (TIMESTAMP)          │
└─────────────────────────────────┘
                 │
                 ├── 1:N ──────────────────────────┐
                 │                                 │
                 ▼                                 ▼
┌─────────────────────────────────┐   ┌─────────────────────────────────────┐
│          voice_samples          │   │          detection_results          │
├─────────────────────────────────┤   ├─────────────────────────────────────┤
│ id (UUID, PK)                   │   │ id (UUID, PK)                       │
│ user_id (UUID, FK -> users.id)  │   │ user_id (UUID, FK -> users.id)      │
│ file_path (TEXT)                │   │ session_id (VARCHAR(255), INDEX)    │
│ embedding (JSONB)               │   │ timestamp (TIMESTAMP, INDEX)        │
│ sample_rate (INTEGER)           │   │ risk_score (FLOAT)                  │
│ duration (FLOAT)                │   │ verdict (VARCHAR(50))               │
│ created_at (TIMESTAMP)          │   │ risk_level (VARCHAR(20), INDEX)     │
└─────────────────────────────────┘   │ features_json (JSONB)               │
                                      │ audio_duration (FLOAT)              │
                                      │ model_used (VARCHAR(100))           │
                                      └─────────────────────────────────────┘
```

---

## 🛠️ Development

### Running Tests
Execute the unit and integration test suite:
```bash
# Run pytest with detailed coverage output
pytest tests/ -v --tb=short
```

### Swapping Models
To evaluate custom models or alternate SpeechBrain checkpoints:
1. Open `config.yaml` or set environment variables:
   ```yaml
   models:
     detection_model: "custom_hf_user/aasist-finetuned"
     speaker_model: "speechbrain/spkrec-xvect-voxceleb"
     device: "cuda"
   ```
2. The model loader dynamically initializes the new architecture on startup and downloads required weights to `models_cache/`.

### Adding Pipeline Stages
The pipeline follows an interceptor architecture where each stage consumes and returns structured typed dataclasses:
1. Create `backend/pipeline/stageX_newstage.py`.
2. Define input and output contracts adhering to shared interfaces.
3. Import and chain the stage within `backend/pipeline/orchestrator.py`:
   ```python
   # Example integration
   stage_data = run_stage_x(previous_stage_output)
   ```

---

## 🔒 Security Notes

- **Secret Keys**: Ensure `SECRET_KEY` in `.env` is set to a cryptographically strong 256-bit random string before deploying to production.
- **Transport Security**: Always enforce HTTPS / WSS when receiving audio streams to prevent eavesdropping and adversary-in-the-middle attacks.
- **CORS Protection**: Configure `CORS_ORIGINS` in `config.yaml` or `.env` to restrict cross-origin access exclusively to authorized dashboard hostnames.
- **Audio Privacy & Storage**: By default, audio files are saved locally in `audio_storage/`. For enterprise production environments, configure encrypted S3/GCS buckets with automatic lifecycle deletion policies.

---

## 📝 License

Distributed under the **MIT License**. See `LICENSE` for further details.
