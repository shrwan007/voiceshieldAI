"""
Stage 3 — Feature Extraction
Responsibility: Extract rich acoustic features from preprocessed audio for
deepfake detection. Features include MFCC, Mel-Spectrogram, pitch contour,
prosodic features, and spectral features.
"""

import numpy as np
import librosa
from dataclasses import dataclass

from backend.pipeline.stage1_capture import AudioData
from backend.utils import audio_utils

@dataclass
class FeatureSet:
    # Spectral
    mfcc: list
    mfcc_delta: list
    mfcc_delta2: list
    mel_spectrogram: list
    chroma: list
    
    # Pitch / Prosody
    pitch: list
    pitch_mean: float
    pitch_std: float
    
    # Spectral statistics
    spectral_centroid: float
    spectral_rolloff: float
    spectral_flatness: float
    spectral_bandwidth: float
    zero_crossing_rate: float
    
    # Energy
    rms_energy: float
    rms_energy_std: float
    
    def to_dict(self) -> dict:
        """Serialize all features to JSON-serializable dict."""
        return {
            "mfcc_mean": float(np.mean(self.mfcc)) if self.mfcc else 0.0,
            "mel_spectrogram_mean": float(np.mean(self.mel_spectrogram)) if self.mel_spectrogram else 0.0,
            "chroma_mean": float(np.mean(self.chroma)) if self.chroma else 0.0,
            "pitch_mean": self.pitch_mean,
            "pitch_std": self.pitch_std,
            "spectral_centroid": self.spectral_centroid,
            "spectral_rolloff": self.spectral_rolloff,
            "spectral_flatness": self.spectral_flatness,
            "spectral_bandwidth": self.spectral_bandwidth,
            "zero_crossing_rate": self.zero_crossing_rate,
            "rms_energy": self.rms_energy,
            "rms_energy_std": self.rms_energy_std
        }

def extract_mfcc(audio: np.ndarray, sr: int, n_mfcc: int = 40) -> tuple[list, list, list]:
    """Extract MFCC + delta + delta-delta."""
    try:
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        return mfcc.tolist(), delta.tolist(), delta2.tolist()
    except Exception:
        return [], [], []

def extract_mel_spectrogram(audio: np.ndarray, sr: int, n_mels: int = 128) -> list:
    """Extract log-power Mel spectrogram."""
    try:
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        return mel_db.tolist()
    except Exception:
        return []

def extract_pitch(audio: np.ndarray, sr: int) -> tuple[list, float, float]:
    """Extract fundamental frequency (F0) contour."""
    try:
        f0, voiced_flag, _ = librosa.pyin(y=audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
        f0_voiced = f0[voiced_flag]
        if len(f0_voiced) > 0:
            return f0_voiced.tolist(), float(np.mean(f0_voiced)), float(np.std(f0_voiced))
        return [], 0.0, 0.0
    except Exception:
        return [], 0.0, 0.0

def extract_spectral_features(audio: np.ndarray, sr: int) -> dict:
    """Extract spectral centroid, rolloff, flatness, bandwidth, ZCR."""
    try:
        centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, roll_percent=0.85)
        flatness = librosa.feature.spectral_flatness(y=audio)
        bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y=audio)
        
        return {
            "spectral_centroid": float(np.mean(centroid)),
            "spectral_rolloff": float(np.mean(rolloff)),
            "spectral_flatness": float(np.mean(flatness)),
            "spectral_bandwidth": float(np.mean(bandwidth)),
            "zero_crossing_rate": float(np.mean(zcr))
        }
    except Exception:
        return {
            "spectral_centroid": 0.0,
            "spectral_rolloff": 0.0,
            "spectral_flatness": 0.0,
            "spectral_bandwidth": 0.0,
            "zero_crossing_rate": 0.0
        }

def extract_features(audio_data: AudioData) -> FeatureSet:
    """Main feature extraction function."""
    sr = audio_data.sample_rate
    audio = audio_data.audio
    
    mfcc, delta, delta2 = extract_mfcc(audio, sr)
    mel_spec = extract_mel_spectrogram(audio, sr)
    pitch, pitch_mean, pitch_std = extract_pitch(audio, sr)
    spec_feats = extract_spectral_features(audio, sr)
    
    try:
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr).tolist()
    except Exception:
        chroma = []
        
    try:
        rms = librosa.feature.rms(y=audio)
        rms_mean = float(np.mean(rms))
        rms_std = float(np.std(rms))
    except Exception:
        rms_mean, rms_std = 0.0, 0.0
        
    return FeatureSet(
        mfcc=mfcc,
        mfcc_delta=delta,
        mfcc_delta2=delta2,
        mel_spectrogram=mel_spec,
        chroma=chroma,
        pitch=pitch,
        pitch_mean=pitch_mean,
        pitch_std=pitch_std,
        spectral_centroid=spec_feats["spectral_centroid"],
        spectral_rolloff=spec_feats["spectral_rolloff"],
        spectral_flatness=spec_feats["spectral_flatness"],
        spectral_bandwidth=spec_feats["spectral_bandwidth"],
        zero_crossing_rate=spec_feats["zero_crossing_rate"],
        rms_energy=rms_mean,
        rms_energy_std=rms_std
    )
