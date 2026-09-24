"""
Stage 4 — AI Voice Deepfake Detection
Responsibility: Classify audio as 'real', 'synthetic', or 'cloned'.

Primary model: SpeechBrain AASIST anti-spoofing model (HuggingFace).
Fallback: Rule-based classifier using spectral features when model unavailable.
"""

import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

import torch
import numpy as np

try:
    from speechbrain.inference.classifiers import AudioClassifier
except ImportError:
    AudioClassifier = None

from backend.pipeline.stage1_capture import AudioData
from backend.pipeline.stage3_features import FeatureSet
from backend.config import get_settings
from backend.utils import audio_utils

@dataclass
class DetectionOutput:
    verdict: str
    confidence: float
    raw_score: float
    model_used: str

class RuleBasedDetector:
    """Fallback detector using spectral feature heuristics."""
    def detect(self, features: FeatureSet) -> DetectionOutput:
        score = 0.0
        if features.spectral_flatness > 0.01:
            score += 0.3
        if features.pitch_std < 5.0 and features.pitch_mean > 0:
            score += 0.4
        if features.rms_energy_std < 0.01:
            score += 0.2
            
        score = min(1.0, score)
        
        if score >= 0.65:
            verdict = "synthetic"
        elif score >= 0.45:
            verdict = "cloned"
        else:
            verdict = "real"
            
        return DetectionOutput(
            verdict=verdict,
            confidence=max(score, 1 - score),
            raw_score=score,
            model_used="rule_based"
        )

class SpeechBrainDetector:
    """AASIST-based deepfake detection using SpeechBrain."""
    def __init__(self, model_name: str, device: str, cache_path: str):
        self.model_name = model_name
        self.device = device
        self.cache_path = cache_path
        self._model = None
        self._load_lock = threading.Lock()
    
    def _load_model(self):
        if self._model is not None:
            return
            
        with self._load_lock:
            if self._model is not None:
                return
            try:
                if AudioClassifier:
                    self._model = AudioClassifier.from_hparams(
                        source=self.model_name,
                        savedir=self.cache_path,
                        run_opts={"device": self.device}
                    )
            except Exception as e:
                print(f"Failed to load SpeechBrain model: {e}")
                self._model = None
                
    def detect(self, audio_data: AudioData, features: FeatureSet) -> Optional[DetectionOutput]:
        self._load_model()
        if self._model is None:
            return None
            
        try:
            signal = torch.FloatTensor(audio_data.audio).unsqueeze(0).to(self.device)
            score, index, text_lab = self._model.classify_batch(signal)
            
            spoof_prob = float(score[0].exp().item()) if len(score.shape) > 0 else 0.5
            
            if spoof_prob > 0.8:
                verdict = "synthetic"
            elif spoof_prob > 0.5:
                verdict = "cloned"
            else:
                verdict = "real"
                
            return DetectionOutput(
                verdict=verdict,
                confidence=max(spoof_prob, 1 - spoof_prob),
                raw_score=spoof_prob,
                model_used="aasist"
            )
        except Exception as e:
            print(f"SpeechBrain detection failed: {e}")
            return None

class VoiceDetector:
    """Main detector facade — tries SpeechBrain, falls back to rule-based."""
    def __init__(self, settings):
        self._speechbrain = SpeechBrainDetector(
            model_name=settings.detection_model,
            device=settings.device,
            cache_path=settings.models_cache_path
        )
        self._rule_based = RuleBasedDetector()
        self._executor = ThreadPoolExecutor(max_workers=2)
    
    def detect(self, audio_data: AudioData, features: FeatureSet) -> DetectionOutput:
        sb_result = self._speechbrain.detect(audio_data, features)
        if sb_result:
            return sb_result
        return self._rule_based.detect(features)
    
    async def detect_async(self, audio_data: AudioData, features: FeatureSet) -> DetectionOutput:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            self.detect,
            audio_data,
            features
        )

_detector: Optional[VoiceDetector] = None

def get_detector() -> VoiceDetector:
    global _detector
    if _detector is None:
        _detector = VoiceDetector(get_settings())
    return _detector

def run_detection(audio_data: AudioData, features: FeatureSet) -> DetectionOutput:
    return get_detector().detect(audio_data, features)

async def run_detection_async(audio_data: AudioData, features: FeatureSet) -> DetectionOutput:
    return await get_detector().detect_async(audio_data, features)
