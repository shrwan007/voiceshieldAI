"""
Stage 1 — Audio Capture & Ingestion
Responsibility: Accept audio from multiple sources (file upload, WebSocket stream),
validate format and duration, and convert to a normalized AudioData object.
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from backend.utils import audio_utils
from backend.config import get_settings

@dataclass
class AudioData:
    audio: np.ndarray          # float32 mono audio samples
    sample_rate: int           # samples per second (always 16000 after capture)
    duration: float            # length in seconds
    filename: str              # original filename or "stream_{session_id}"
    source: str = "upload"     # "upload" | "stream"
    session_id: Optional[str] = None

@dataclass
class StreamSession:
    session_id: str
    buffer: bytearray = field(default_factory=bytearray)  # accumulates incoming PCM bytes
    chunks_received: int = 0
    total_bytes: int = 0
    sample_rate: int = 16000
    created_at: float = field(default_factory=time.time)

_stream_sessions: dict[str, StreamSession] = {}  # module-level session registry

async def ingest_upload(file_bytes: bytes, filename: str, target_sr: int = 16000) -> AudioData:
    """
    Stage 1a: Ingest an uploaded audio file (WAV/MP3/FLAC/OGG).
    Validates format, converts to mono float32, resamples to target_sr.
    Raises ValueError if format invalid or duration out of bounds.
    """
    if not file_bytes:
        raise ValueError("Audio file is empty.")
    
    settings = get_settings()
    try:
        audio_array, target_sr = audio_utils.bytes_to_numpy(file_bytes, target_sr)
    except Exception as e:
        raise ValueError(f"Invalid audio format or processing error: {e}")
        
    duration = len(audio_array) / target_sr
    
    if duration < settings.min_audio_length_sec or duration > settings.max_audio_length_sec:
        raise ValueError(f"Audio duration {duration:.2f}s is out of bounds "
                         f"({settings.min_audio_length_sec}s - {settings.max_audio_length_sec}s).")
                         
    return AudioData(
        audio=audio_array,
        sample_rate=target_sr,
        duration=duration,
        filename=filename,
        source="upload"
    )

async def ingest_stream_chunk(
    chunk: bytes,
    session: StreamSession,
    min_buffer_sec: float = 1.0
) -> Optional[AudioData]:
    """
    Stage 1b: Process a raw PCM chunk from a WebSocket stream.
    Accumulates chunks into a session buffer.
    Returns AudioData when buffer reaches min_buffer_sec of audio, else None.
    The caller is responsible for clearing the buffer after consuming AudioData.
    """
    session.buffer.extend(chunk)
    session.chunks_received += 1
    session.total_bytes += len(chunk)
    
    # 16-bit PCM = 2 bytes per sample
    bytes_per_sec = session.sample_rate * 2 
    min_bytes_required = int(min_buffer_sec * bytes_per_sec)
    
    if len(session.buffer) >= min_bytes_required:
        audio_array = audio_utils.pcm_bytes_to_numpy(bytes(session.buffer), session.sample_rate)
        duration = len(audio_array) / session.sample_rate
        session.buffer.clear()
        
        return AudioData(
            audio=audio_array,
            sample_rate=session.sample_rate,
            duration=duration,
            filename=f"stream_{session.session_id}",
            source="stream",
            session_id=session.session_id
        )
    return None

def create_stream_session(session_id: str, sample_rate: int = 16000) -> StreamSession:
    """Create a new streaming session object."""
    session = StreamSession(session_id=session_id, sample_rate=sample_rate)
    _stream_sessions[session_id] = session
    return session

def get_or_create_session(session_id: str) -> StreamSession:
    """Get existing session or create new one."""
    if session_id not in _stream_sessions:
        return create_stream_session(session_id)
    return _stream_sessions[session_id]

def close_session(session_id: str) -> None:
    """Remove session from registry."""
    if session_id in _stream_sessions:
        del _stream_sessions[session_id]
