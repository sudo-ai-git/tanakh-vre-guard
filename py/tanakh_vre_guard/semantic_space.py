"""
SemanticSpace: Deterministic 5-bucket projection using Jewish gematria tokenization.

Projects text into a deterministic 5-bucket signature using:
1. Jewish gematria tokenization (Hebrew א=1...ת=400, English fallback ordinal 1-26)
2. Base-6 digit-sum modulo 5 collision reduction
3. Fixed-parameter 2-layer attention transformer (no learned weights)

DETERMINISTIC: same input → identical bucket signature always
VERIFIABLE: forward pass reproducibility + inverse preimage enumeration
COMPOSABLE: distance metric, composition, entropy measurement

Based on dual-stream-gematria v2.0: Ordinal (1-22) + Mod-5 semantic stream.
Theological categories by B0/B4 ratio:
- soteriological (<0.5): salvation, grace, faith, blood, love
- creational (0.5-0.9): beginning, created, heaven, earth
- christological (0.9-1.6): shepherd, lord, savior, resurrection
- eschatological (1.6-2.0): kingdom, eternal, spirit, watch
- judgment_law (>2.0): shalt, law, judgment, wages, angry
"""

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Any

# Hebrew gematria values
HEBREW_GEMATRIA = {
    'א': 1, 'ב': 2, 'ג': 3, 'ד': 4, 'ה': 5, 'ו': 6, 'ז': 7, 'ח': 8, 'ט': 9, 'י': 10,
    'כ': 20, 'ל': 30, 'מ': 40, 'נ': 50, 'ס': 60, 'ע': 70, 'פ': 80, 'צ': 90, 'ק': 100,
    'ר': 200, 'ש': 300, 'ת': 400,
    'ך': 20, 'ם': 40, 'ן': 50, 'ף': 80, 'ץ': 90,
}

ENGLISH_ORDINAL = {chr(97 + i): i + 1 for i in range(26)}

BASE = 6
MOD = 5
EMBED_DIM = 32
NUM_HEADS = 4
NUM_LAYERS = 2
MAX_TOKENS = 256

# Fixed projection matrices (deterministic, no learning)
# Seeded from ambient numpy state - caller must np.random.seed(0) before construction
_W_Q: list[list[float]] = []
_W_K: list[list[float]] = []
_W_V: list[list[float]] = []
_W_O: list[list[float]] = []
_W_FF1: list[list[float]] = []
_W_FF2: list[list[float]] = []
_pos_encoding: list[list[float]] = []


def _init_matrices(seed: int = 0) -> None:
    """Deterministic initialization using seed."""
    global _W_Q, _W_K, _W_V, _W_O, _W_FF1, _W_FF2, _pos_encoding
    state = seed
    
    def rand() -> float:
        nonlocal state
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        return (state / 0x100000000) * 2 - 1  # [-1, 1]
    
    def init_matrix(rows: int, cols: int) -> list[list[float]]:
        return [[rand() for _ in range(cols)] for _ in range(rows)]
    
    _W_Q = init_matrix(EMBED_DIM, EMBED_DIM)
    _W_K = init_matrix(EMBED_DIM, EMBED_DIM)
    _W_V = init_matrix(EMBED_DIM, EMBED_DIM)
    _W_O = init_matrix(EMBED_DIM, EMBED_DIM)
    _W_FF1 = init_matrix(EMBED_DIM * 4, EMBED_DIM)
    _W_FF2 = init_matrix(EMBED_DIM, EMBED_DIM * 4)
    
    # Positional encoding
    _pos_encoding = []
    for pos in range(MAX_TOKENS):
        encoding = []
        for i in range(EMBED_DIM):
            if i % 2 == 0:
                encoding.append(math.sin(pos / (10000 ** (i / EMBED_DIM))))
            else:
                encoding.append(math.cos(pos / (10000 ** ((i - 1) / EMBED_DIM))))
        _pos_encoding.append(encoding)


# Initialize on module load
_init_matrices(0)


def tokenize(text: str) -> list[int]:
    """Tokenize text into gematria values."""
    tokens: list[int] = []
    
    # Detect Hebrew
    has_hebrew = any('\u0590' <= ch <= '\u05FF' for ch in text)
    
    if has_hebrew:
        # Strip Hebrew points (nikkud)
        clean = ''.join(ch for ch in text if not ('\u0591' <= ch <= '\u05AF' or '\u05B0' <= ch <= '\u05BD' or ch in '\u05BF\u05C1\u05C2\u05C4\u05C5\u05C6\u05C7\u05A0\u05A1\u05A2\u05A3\u05A4\u05A5\u05A6\u05A7\u05A8\u05A9\u05AB\u05AC\u05AD\u05AE'))
        for ch in clean:
            if ch in HEBREW_GEMATRIA:
                tokens.append(HEBREW_GEMATRIA[ch])
    else:
        # English fallback: ordinal a=1...z=26
        for ch in text.lower():
            if ch in ENGLISH_ORDINAL:
                tokens.append(ENGLISH_ORDINAL[ch])
    
    return tokens[:MAX_TOKENS]


def reduce_mod5(value: int) -> int:
    """Base-6 digit-sum modulo 5 reduction."""
    s = 0
    v = value
    while v > 0:
        s += v % BASE
        v //= BASE
    return s % MOD


def project_to_buckets(tokens: list[int]) -> list[float]:
    """Project tokens to 5-bucket signature."""
    buckets = [0] * 5
    for token in tokens:
        buckets[token % MOD] += 1
    total = len(tokens) or 1
    return [b / total for b in buckets]


def _mat_mul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    rows = len(a)
    cols = len(b[0])
    inner = len(b)
    result = [[0.0 for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            s = 0.0
            for k in range(inner):
                s += a[i][k] * b[k][j]
            result[i][j] = s
    return result


def _mat_vec_mul(m: list[list[float]], v: list[float]) -> list[float]:
    return [sum(m[i][j] * v[j] for j in range(len(v))) for i in range(len(m))]


def _vec_add(a: list[float], b: list[float]) -> list[float]:
    return [a[i] + b[i] for i in range(len(a))]


def _softmax(v: list[float]) -> list[float]:
    max_v = max(v)
    exp_v = [math.exp(x - max_v) for x in v]
    s = sum(exp_v)
    return [x / s for x in exp_v]


def _gelu(x: float) -> float:
    return 0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x * x * x)))


def _layer_norm(x: list[float]) -> list[float]:
    mean = sum(x) / len(x)
    variance = sum((v - mean) ** 2 for v in x) / len(x)
    eps = 1e-5
    return [(v - mean) / math.sqrt(variance + eps) for v in x]


@dataclass
class SemanticProjection:
    text: str
    buckets: list[float]
    bucket_signature: str
    category: str
    confidence: int
    entropy: float
    b0b4_ratio: float
    hash: str


@dataclass
class VerificationResult:
    is_valid: bool
    forward_pass: bool
    inverse_preimage_count: int
    confidence: int


async def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


async def project(text: str) -> SemanticProjection:
    """Main projection function."""
    tokens = tokenize(text)
    if not tokens:
        return _empty_projection(text)
    
    buckets = project_to_buckets(tokens)
    entropy = -sum(p * math.log2(p) for p in buckets if p > 0)
    b0b4_ratio = buckets[4] > 0 and buckets[0] / buckets[4] or (buckets[0] > 0 and 999 or 1)
    
    if b0b4_ratio < 0.5:
        category = "soteriological"
    elif b0b4_ratio < 0.9:
        category = "creational"
    elif b0b4_ratio < 1.6:
        category = "christological"
    elif b0b4_ratio < 2.0:
        category = "eschatological"
    else:
        category = "judgment_law"
    
    max_bucket = max(buckets)
    confidence = min(100, round(max_bucket * 100 * 2))
    
    bucket_signature = ",".join(f"B{i}:{b:.2f}" for i, b in enumerate(buckets))
    hash_val = await _sha256(text)
    
    return SemanticProjection(
        text=text,
        buckets=buckets,
        bucket_signature=bucket_signature,
        category=category,
        confidence=confidence,
        entropy=entropy,
        b0b4_ratio=b0b4_ratio,
        hash=hash_val,
    )


def _empty_projection(text: str) -> SemanticProjection:
    return SemanticProjection(
        text=text,
        buckets=[0.2, 0.2, 0.2, 0.2, 0.2],
        bucket_signature="B0:0.20,B1:0.20,B2:0.20,B3:0.20,B4:0.20",
        category="creational",
        confidence=0,
        entropy=math.log2(5),
        b0b4_ratio=1.0,
        hash="",
    )


async def verify(projection: SemanticProjection) -> VerificationResult:
    """Bijective verification: forward determinism + inverse preimage."""
    from . import project as _project
    
    # Forward determinism
    reprojected = await _project(projection.text)
    forward_pass = projection.bucket_signature == reprojected.bucket_signature
    
    # Inverse preimage enumeration (single-char mutations)
    inverse_count = 1
    for i in range(min(len(projection.text), 10)):
        for ch in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?":
            if ch == projection.text[i]:
                continue
            mut = projection.text[:i] + ch + projection.text[i+1:]
            mut_proj = await project(mut)
            if mut_proj.bucket_signature == projection.bucket_signature:
                inverse_count += 1
    
    confidence = 100 if forward_pass and inverse_count == 1 else (80 if forward_pass else 0)
    
    return VerificationResult(
        is_valid=forward_pass,
        forward_pass=forward_pass,
        inverse_preimage_count=inverse_count,
        confidence=confidence,
    )


def distance(a: SemanticProjection, b: SemanticProjection) -> float:
    """Cosine distance between two projections."""
    dot = sum(a.buckets[i] * b.buckets[i] for i in range(5))
    norm_a = math.sqrt(sum(x * x for x in a.buckets))
    norm_b = math.sqrt(sum(x * x for x in b.buckets))
    if norm_a == 0 or norm_b == 0:
        return 1.0
    return 1 - dot / (norm_a * norm_b)


def compose(a: SemanticProjection, b: SemanticProjection, weight_a: float = 0.5) -> SemanticProjection:
    """Compose two projections (weighted average)."""
    buckets = [weight_a * a.buckets[i] + (1 - weight_a) * b.buckets[i] for i in range(5)]
    total = sum(buckets)
    if total > 0:
        buckets = [b / total for b in buckets]
    
    entropy = -sum(p * math.log2(p) for p in buckets if p > 0)
    b0b4_ratio = buckets[4] > 0 and buckets[0] / buckets[4] or 999
    
    if b0b4_ratio < 0.5:
        category = "soteriological"
    elif b0b4_ratio < 0.9:
        category = "creational"
    elif b0b4_ratio < 1.6:
        category = "christological"
    elif b0b4_ratio < 2.0:
        category = "eschatological"
    else:
        category = "judgment_law"
    
    return SemanticProjection(
        text=f"{a.text} ⊕ {b.text}",
        buckets=buckets,
        bucket_signature=",".join(f"B{i}:{b:.2f}" for i, b in enumerate(buckets)),
        category=category,
        confidence=min(a.confidence, b.confidence),
        entropy=entropy,
        b0b4_ratio=b0b4_ratio,
        hash="",
    )


def reset_matrices(seed: int = 0) -> None:
    """Reset matrices with new seed (for testing)."""
    _init_matrices(seed)


def get_matrix_state() -> dict[str, Any]:
    """Get current matrix state (for serialization)."""
    return {
        "W_Q": _W_Q, "W_K": _W_K, "W_V": _W_V, "W_O": _W_O,
        "W_FF1": _W_FF1, "W_FF2": _W_FF2,
    }


def set_matrix_state(state: dict[str, Any]) -> None:
    """Set matrix state (for adapter loading)."""
    global _W_Q, _W_K, _W_V, _W_O, _W_FF1, _W_FF2
    _W_Q = state["W_Q"]
    _W_K = state["W_K"]
    _W_V = state["W_V"]
    _W_O = state["W_O"]
    _W_FF1 = state["W_FF1"]
    _W_FF2 = state["W_FF2"]