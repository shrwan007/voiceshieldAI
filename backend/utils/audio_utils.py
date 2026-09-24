import io
import os
import uuid
import numpy as np
import soundfile as sf
import librosa
from typing import Tuple, Optional

import shutil
import subprocess

def validate_audio_format(file_bytes: bytes, filename: str) -> bool:
    """
    Validate that the file is an allowed audio format and not empty.
    """
    if len(file_bytes) == 0:
        return False
    
    allowed_extensions = [".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm", ".aac", ".opus"]
    ext = os.path.splitext(filename)[1].lower()
    return ext in allowed_extensions

def _decode_with_ffmpeg(file_bytes: bytes, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Fallback audio decoder using ffmpeg subprocess pipe.
    Supports WebM, Opus, M4A, AAC, and any container format.
    """
    ffmpeg_cmd = (
        shutil.which("ffmpeg")
        or os.path.expanduser(r"~\scoop\shims\ffmpeg.exe")
        or os.path.expanduser(r"~\scoop\apps\ffmpeg\current\bin\ffmpeg.exe")
    )
    if not ffmpeg_cmd or (not os.path.exists(ffmpeg_cmd) and not shutil.which("ffmpeg")):
        raise RuntimeError("ffmpeg not found for audio decoding.")
    
    cmd = [
        ffmpeg_cmd,
        "-i", "pipe:0",
        "-f", "s16le",
        "-acodec", "pcm_s16le",
        "-ac", "1",
        "-ar", str(target_sr),
        "pipe:1"
    ]
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    out, err = process.communicate(input=file_bytes)
    if process.returncode != 0:
        raise ValueError(f"ffmpeg conversion failed: {err.decode('utf-8', errors='ignore')}")
    
    audio_int = np.frombuffer(out, dtype=np.int16)
    audio_float = audio_int.astype(np.float32) / 32768.0
    return audio_float, target_sr

def bytes_to_numpy(file_bytes: bytes, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Convert raw audio bytes to a numpy array.
    Tries soundfile first; falls back to ffmpeg for webm/opus/m4a/etc.
    """
    try:
        # First attempt: standard soundfile (WAV, FLAC, OGG)
        audio_io = io.BytesIO(file_bytes)
        audio, orig_sr = sf.read(audio_io)
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio.T)
            
        # Resample if needed
        if orig_sr != target_sr:
            audio = librosa.resample(y=audio, orig_sr=orig_sr, target_sr=target_sr)
            
        return audio, target_sr
    except Exception:
        # Second attempt: ffmpeg fallback (WebM, Opus, M4A, etc.)
        try:
            return _decode_with_ffmpeg(file_bytes, target_sr)
        except Exception as ffmpeg_err:
            raise ValueError(f"Failed to process audio bytes: {ffmpeg_err}")

def numpy_to_bytes(audio: np.ndarray, sr: int, fmt: str = "WAV") -> bytes:
    """
    Convert a numpy array to audio bytes.
    """
    try:
        audio_io = io.BytesIO()
        sf.write(audio_io, audio, sr, format=fmt)
        return audio_io.getvalue()
    except Exception as e:
        raise ValueError(f"Failed to convert numpy to bytes: {e}")

def get_audio_duration(audio: np.ndarray, sr: int) -> float:
    """
    Calculate the duration of the audio in seconds.
    """
    if sr <= 0:
        raise ValueError("Sample rate must be positive.")
    return len(audio) / sr

def save_audio_file(audio: np.ndarray, sr: int, directory: str, filename: Optional[str] = None) -> str:
    """
    Save a numpy array as a WAV file.
    """
    try:
        os.makedirs(directory, exist_ok=True)
        if not filename:
            filename = f"{uuid.uuid4()}.wav"
        if not filename.lower().endswith(".wav"):
            filename += ".wav"
            
        file_path = os.path.join(directory, filename)
        sf.write(file_path, audio, sr)
        return file_path
    except Exception as e:
        raise IOError(f"Failed to save audio file: {e}")

def load_audio_file(file_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Load an audio file into a numpy array.
    """
    try:
        audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
        return audio, sr
    except Exception as e:
        raise IOError(f"Failed to load audio file {file_path}: {e}")

def pcm_bytes_to_numpy(pcm_bytes: bytes, sample_rate: int = 16000, sample_width: int = 2) -> np.ndarray:
    """
    Convert raw PCM bytes to a float32 numpy array normalized to [-1, 1].
    """
    try:
        if sample_width == 2:
            dtype = np.int16
        elif sample_width == 4:
            dtype = np.int32
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")
            
        audio_int = np.frombuffer(pcm_bytes, dtype=dtype)
        
        # Normalize to float32
        max_val = float(np.iinfo(dtype).max)
        audio_float = audio_int.astype(np.float32) / max_val
        
        return audio_float
    except Exception as e:
        raise ValueError(f"Failed to convert PCM bytes to numpy: {e}")
