"""
BijectiveVerifier: Forward determinism + inverse preimage enumeration.

Verifies SemanticSpace projections by:
1. Forward pass: re-project text → same bucket signature?
2. Inverse: enumerate preimages mapping to same signature
3. Confidence: unique preimage = high confidence

This is the "structural checksum" layer of VRE - deterministic, not factual.
A flat lie passes bijective verification; factual truth needs oracle grounding.
"""

import random
import math
from dataclasses import dataclass
from typing import Any

from .semantic_space import SemanticProjection, VerificationResult, distance, compose, project
from .checksum231 import commitment


@dataclass
class BijectiveVerificationResult(VerificationResult):
    forward_pass: bool
    inverse_preimage_count: int
    preimage_samples: list[str]
    semantic_drift: float
    collision_resistance: float


async def bijective_verify(
    projection: SemanticProjection,
    neighborhood_radius: int = 10
) -> BijectiveVerificationResult:
    """Verify a projection bijectively.
    - Re-runs forward projection (deterministic check)
    - Enumerates inverse preimages in neighborhood
    - Measures collision resistance
    """
    from .semantic_space import verify as semantic_verify
    
    # 1. Forward determinism check
    forward_result = await semantic_verify(projection)
    
    # 2. Inverse preimage enumeration
    preimage_samples = [projection.text]
    inverse_preimage_count = 1
    
    # Single-character mutations
    for i in range(min(len(projection.text), neighborhood_radius)):
        for ch in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?":
            if ch == projection.text[i]:
                continue
            mut = projection.text[:i] + ch + projection.text[i + 1:]
            mut_proj = await project(mut)
            if mut_proj.bucket_signature == projection.bucket_signature:
                inverse_preimage_count += 1
                if len(preimage_samples) < 5:
                    preimage_samples.append(mut)
    
    # Transposition mutations (adjacent swaps)
    for i in range(min(len(projection.text) - 1, 5)):
        if projection.text[i] != projection.text[i + 1]:
            mut = projection.text[:i] + projection.text[i + 1] + projection.text[i] + projection.text[i + 2:]
            mut_proj = await project(mut)
            if mut_proj.bucket_signature == projection.bucket_signature:
                inverse_preimage_count += 1
                if len(preimage_samples) < 5:
                    preimage_samples.append(mut)
    
    # 3. Semantic drift to nearest neighbor with different text
    semantic_drift = 1.0
    for _ in range(100):
        mut = random_mutation(projection.text)
        mut_proj = await project(mut)
        if mut_proj.bucket_signature != projection.bucket_signature:
            d = distance(projection, mut_proj)
            if d < semantic_drift:
                semantic_drift = d
    
    # 4. Collision resistance (entropy bits)
    entropy = -sum(p * math.log2(p) for p in projection.buckets if p > 0)
    collision_resistance = entropy * 5
    
    confidence = 100 if forward_result.forward_pass and inverse_preimage_count == 1 else (80 if forward_result.forward_pass else 0)
    
    return BijectiveVerificationResult(
        is_valid=forward_result.forward_pass,
        forward_pass=forward_result.forward_pass,
        inverse_preimage_count=inverse_preimage_count,
        preimage_samples=preimage_samples,
        semantic_drift=semantic_drift,
        collision_resistance=collision_resistance,
        confidence=confidence,
    )


async def verify_chain(projections: list[SemanticProjection]) -> dict:
    """Verify a chain of projections (composed reasoning)."""
    results = []
    all_valid = True
    
    for proj in projections:
        result = await bijective_verify(proj)
        results.append(result)
        if not result.is_valid:
            all_valid = False
    
    # Composition drift: distance between composed and sequential
    composition_drift = 0.0
    if len(projections) >= 2:
        composed = compose(projections[0], projections[1])
        sequential = await project(projections[0].text + " " + projections[1].text)
        composition_drift = distance(composed, sequential)
    
    return {"all_valid": all_valid, "results": results, "composition_drift": composition_drift}


def random_mutation(text: str) -> str:
    chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?"
    idx = random.randint(0, len(text) - 1)
    ch = random.choice(chars)
    return text[:idx] + ch + text[idx + 1:]


class SparseFingerprinter:
    """Top-k Jaccard for compose phase deduplication."""
    
    def __init__(self, k: int = 16):
        self.k = k
        self.projections: dict[str, SemanticProjection] = {}
    
    def add(self, projection: SemanticProjection) -> str:
        fp = self.fingerprint(projection)
        self.projections[fp] = projection
        return fp
    
    def fingerprint(self, projection: SemanticProjection) -> str:
        indexed = [(i, w) for i, w in enumerate(projection.buckets)]
        indexed.sort(key=lambda x: x[1], reverse=True)
        return "|".join(f"{idx}:{w:.3f}" for idx, w in indexed[:self.k])
    
    def jaccard(self, fp1: str, fp2: str) -> float:
        set1 = set(fp1.split("|"))
        set2 = set(fp2.split("|"))
        inter = len(set1 & set2)
        union = len(set1) + len(set2) - inter
        return inter / union if union > 0 else 0.0
    
    def find_similar(self, fp: str, threshold: float = 0.7) -> list[str]:
        return [key for key in self.projections if self.jaccard(fp, key) >= threshold]
    
    def is_duplicate(self, projection: SemanticProjection, threshold: float = 0.9) -> bool:
        fp = self.fingerprint(projection)
        return len(self.find_similar(fp, threshold)) > 0


class ComposePhase:
    """Sparse fingerprint-based deduplication for VRE compose phase."""
    
    def __init__(self, skip_threshold: float = 0.7):
        self.fingerprinter = SparseFingerprinter(16)
        self.skip_threshold = skip_threshold
    
    async def apply(self, projection: SemanticProjection) -> dict:
        fp = self.fingerprinter.fingerprint(projection)
        similar = self.fingerprinter.find_similar(fp, self.skip_threshold)
        
        if similar:
            return {
                "skipped": True,
                "reason": f"Similar projection found (Jaccard >= {self.skip_threshold})",
                "similar": similar,
                "fingerprint": fp,
            }
        
        self.fingerprinter.add(projection)
        return {"skipped": False, "fingerprint": fp}
    
    def reset(self) -> None:
        self.fingerprinter = SparseFingerprinter(16)
    
    def stats(self) -> dict[str, int]:
        return {"total_projections": len(self.fingerprinter.projections)}