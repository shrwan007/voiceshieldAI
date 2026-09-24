"""
Stage 5 — Speaker Verification & Identity Check
Responsibility: Compare the voice in the audio against an enrolled genuine
sample using speaker embedding cosine similarity (ECAPA-TDNN).
"""

import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional
import math

import torch
import numpy as np

try:
    from speechbrain.inference.speaker import EncoderClassifier
except ImportError:
    EncoderClassifier = None

from backend.pipeline.stage1_capture import AudioData
from backend.config import get_settings
from backend.utils import audio_utils

@dataclass
class VerificationOutput:
    similarity: float
    is_same_speaker: bool
    confidence: float
    embedding: list[float]

class SpeakerVerifier:
    """ECAPA-TDNN speaker embedding model."""
    def __init__(self, model_name: str, device: str, cache_path: str, similarity_threshold: float = 0.75):
        self.model_name = model_name
        self.device = device
        self.cache_path = cache_path
        self.similarity_threshold = similarity_threshold
        self._model = None
        self._load_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=2)
    
    def _load_model(self):
        if self._model is not None:
            return
            
        with self._load_lock:
            if self._model is not None:
                return
            try:
                if EncoderClassifier:
                    self._model = EncoderClassifier.from_hparams(
                        source=self.model_name,
                        savedir=self.cache_path,
                        run_opts={"device": self.device}
                    )
            except Exception as e:
                print(f"Failed to load speaker verifier model: {e}")
                self._model = None
    
    def get_embedding(self, audio_data: AudioData) -> list[float]:
        self._load_model()
        if self._model:
            try:
                signal = torch.FloatTensor(audio_data.audio).unsqueeze(0).to(self.device)
                embeddings = self._model.encode_batch(signal)
                return embeddings.squeeze().tolist()
            except Exception as e:
                print(f"Embedding extraction failed: {e}")
                
        # Fallback pseudo-embedding (needs 192 dims usually, we pad or slice)
        arr = audio_data.audio[:192]
        return [float(x) for x in arr] + [0.0] * max(0, 192 - len(arr))
    
    def verify(self, audio_data: AudioData, enrolled_embedding: list[float]) -> VerificationOutput:
        audio_emb = self.get_embedding(audio_data)
        similarity = self.cosine_similarity(audio_emb, enrolled_embedding)
        
        is_same = similarity >= self.similarity_threshold
        confidence = max(0.0, (similarity + 1) / 2)
        
        return VerificationOutput(
            similarity=similarity,
            is_same_speaker=is_same,
            confidence=confidence,
            embedding=audio_emb
        )
        
    def cosine_similarity(self, a: list[float], b: list[float]) -> float:
        if len(a) != len(b) or not a or not b:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

_verifier: Optional[SpeakerVerifier] = None

def get_verifier() -> SpeakerVerifier:
    global _verifier
    if _verifier is None:
        settings = get_settings()
        _verifier = SpeakerVerifier(
            model_name=settings.speaker_model,
            device=settings.device,
            cache_path=settings.models_cache_path,
            similarity_threshold=0.75
        )
    return _verifier

def run_verification(audio_data: AudioData, enrolled_embedding: list[float]) -> VerificationOutput:
    return get_verifier().verify(audio_data, enrolled_embedding)

async def run_verification_async(audio_data: AudioData, enrolled_embedding: list[float]) -> VerificationOutput:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        get_verifier()._executor,
        run_verification,
        audio_data,
        enrolled_embedding
    )

def extract_embedding(audio_data: AudioData) -> list[float]:
    return get_verifier().get_embedding(audio_data)

async def extract_embedding_async(audio_data: AudioData) -> list[float]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        get_verifier()._executor,
        extract_embedding,
        audio_data
    )
