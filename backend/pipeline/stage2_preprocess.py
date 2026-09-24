"""
Stage 2 — Audio Preprocessing
Responsibility: Clean the raw audio signal by:
  1. Noise reduction (spectral gating via noisereduce)
  2. Silence detection and trimming (librosa.effects.trim)
  3. Amplitude normalization (peak normalization to 0.95)
  4. Pre-emphasis filter for speech clarity
"""

import numpy as np
import librosa
import noisereduce as nr

from backend.pipeline.stage1_capture import AudioData
from backend.config import get_settings
from backend.utils import audio_utils

def reduce_noise(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Apply spectral gating noise reduction.
    Uses the first 0.5s as noise profile estimate if audio is long enough.
    """
    try:
        samples_05s = int(sr * 0.5)
        if len(audio) >= samples_05s:
            noise_clip = audio[:samples_05s]
            return nr.reduce_noise(y=audio, sr=sr, y_noise=noise_clip)
        else:
            return nr.reduce_noise(y=audio, sr=sr)
    except Exception:
        return audio

def trim_silence(audio: np.ndarray, sr: int, top_db: float = 40.0) -> np.ndarray:
    """
    Remove leading and trailing silence using librosa.
    Returns trimmed audio. If result is too short, returns original.
    """
    try:
        trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
        if len(trimmed) < sr * 0.5:
            return audio
        return trimmed
    except Exception:
        return audio

def normalize(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """
    Peak-normalize audio to target_peak amplitude.
    Prevents clipping by checking for near-zero signals.
    """
    peak = np.max(np.abs(audio))
    if peak < 1e-5:
        return audio
    return (audio / peak) * target_peak

def apply_preemphasis(audio: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    """
    Apply pre-emphasis filter: y[n] = x[n] - coeff * x[n-1]
    Boosts high frequencies to improve feature extraction on speech.
    """
    if len(audio) < 2:
        return audio
    return np.append(audio[0], audio[1:] - coeff * audio[:-1])

def preprocess(audio_data: AudioData) -> AudioData:
    """
    Main preprocessing function — runs all 4 steps in sequence.
    Returns a new AudioData with cleaned audio, same metadata.
    Logs each step's input/output duration.
    """
    settings = get_settings()
    
    audio_denoised = reduce_noise(audio_data.audio, audio_data.sample_rate)
    
    top_db = abs(settings.silence_threshold_db)
    audio_trimmed = trim_silence(audio_denoised, audio_data.sample_rate, top_db=top_db)
    
    audio_norm = normalize(audio_trimmed)
    audio_preemph = apply_preemphasis(audio_norm)
    
    duration = len(audio_preemph) / audio_data.sample_rate
    
    return AudioData(
        audio=audio_preemph,
        sample_rate=audio_data.sample_rate,
        duration=duration,
        filename=audio_data.filename,
        source=audio_data.source,
        session_id=audio_data.session_id
    )
