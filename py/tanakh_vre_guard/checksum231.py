"""
231-Gate Three-Channel Checksum (SUM, LINEAR, PRODUCT mod 231)

Mirrors the NEAR contract implementation in assembly/index.ts exactly.
Input: string payload (ASCII, charCodeAt & 0xff per character)
Output: "fnv8-sum2-lin2-odd2" format, e.g., "3b9aca00-8f-2a-0c"

Three channels:
- L0: FNV-1a-32 over low byte of each code unit (only layer seeing order)
- L1: sum(byte + index) mod 231 (SUM channel)
- L2: sum(byte*22 + index) mod 231 (LINEAR channel)  
- L3: count(odd bytes) mod 37 (PARITY channel)

HONEST LIMIT: L1/L2/L3 are permutation-invariant; only L0 sees order.
Full collision cost ~2^16 permutations (measured 37,417 in 0.8s).
Use for tamper DETECTION only; file identity uses sha256.
"""

import hashlib
import random

FNV_OFFSET = 2166136261
FNV_PRIME = 16777619
MOD_231 = 231
MOD_37 = 37
LINEAR_MULT = 22
SHA_PREFIX = "sha256:"


def fnv1a(s: str) -> int:
    h = FNV_OFFSET
    for ch in s:
        h = ((h ^ (ord(ch) & 0xFF)) * FNV_PRIME) & 0xFFFFFFFF
    return h


def sum_gate(s: str) -> int:
    acc = 0
    for i, ch in enumerate(s):
        acc += (ord(ch) & 0xFF) + i
    return acc % MOD_231


def lin_gate(s: str) -> int:
    acc = 0
    for i, ch in enumerate(s):
        acc += (ord(ch) & 0xFF) * LINEAR_MULT + i
    return acc % MOD_231


def odd_gate(s: str) -> int:
    c = 0
    for ch in s:
        if (ord(ch) & 0xFF) & 1:
            c += 1
    return c % MOD_37


def commitment(payload: str) -> str:
    """Commitment string: "fnv8-sum2-lin2-odd2" """
    return f"{fnv1a(payload):08x}-{sum_gate(payload):02x}-{lin_gate(payload):02x}-{odd_gate(payload):02x}"


def check(payload: str, expectation: str) -> bool:
    """Check if payload satisfies expectation (all 4 layers match)."""
    return mismatch_count(payload, expectation) == 0


def mismatch_count(payload: str, expected: str) -> int:
    """Number of layers that disagree (0-4)."""
    parts = expected.split("-")
    if len(parts) != 4:
        return 4
    mask = 0
    if f"{fnv1a(payload):08x}" != parts[0]:
        mask |= 1
    if f"{sum_gate(payload):02x}" != parts[1]:
        mask |= 2
    if f"{lin_gate(payload):02x}" != parts[2]:
        mask |= 4
    if f"{odd_gate(payload):02x}" != parts[3]:
        mask |= 8
    return bin(mask).count("1")


def matching_layers(delivered: str, expected: str) -> int:
    """Layer-by-layer comparison of two commitments (for predicate matching)."""
    a = delivered.split("-")
    b = expected.split("-")
    if len(a) != 4 or len(b) != 4:
        return 4 if delivered == expected else 0
    m = 0
    for i in range(4):
        if a[i] == b[i]:
            m += 1
    return m


def is_sha_predicate(pred: str) -> bool:
    return pred.startswith(SHA_PREFIX)


async def digest_for(pred: str, payload: str) -> str:
    """Compute digest for a predicate kind."""
    if is_sha_predicate(pred):
        h = hashlib.sha256(payload.encode()).hexdigest()
        return SHA_PREFIX + h
    return commitment(payload)


async def digest_matches(delivered: str, pred: str) -> bool:
    """Check if delivered digest matches predicate."""
    if is_sha_predicate(pred):
        return delivered == pred
    return matching_layers(delivered, pred) == 4


def verify_tamper_evidence(test_payloads: list[str] | None = None) -> dict:
    """Verify tamper evidence: single character change breaks >= 3 layers."""
    if test_payloads is None:
        test_payloads = [
            "hello world",
            "the quick brown fox",
            "market report #42: NEAR Intents volume up 18% w/w",
        ]
    
    worst = 4
    worst_case = None
    
    for payload in test_payloads:
        c = commitment(payload)
        for i in range(len(payload)):
            mut = payload[:i] + ("X" if payload[i] != "X" else "Y") + payload[i + 1:]
            n = mismatch_count(mut, c)
            if n < worst:
                worst = n
                worst_case = {"payload": payload, "index": i}
    
    return {
        "passed": worst >= 3,
        "min_broken": worst,
        "worst_case": worst_case,
    }


def find_permutation_collision(
    base: str = "POD-DELIVERY-000",
    max_permutations: int = 300_000
) -> dict:
    """Find a full 4-layer collision via permutation (demonstrates weakness)."""
    rng = random.Random(1234)
    seen = {}
    buf = list(base)
    
    for i in range(max_permutations):
        rng.shuffle(buf)
        p = "".join(buf)
        c = commitment(p)
        if c in seen and seen[c] != p:
            return {
                "found": True,
                "payload_a": seen[c],
                "payload_b": p,
                "commitment": c,
                "permutations": i,
            }
        seen[c] = p
    
    return {"found": False, "permutations": max_permutations}


GOLDEN_VECTORS = {
    "hello world": "d58b3fa7-10-79-04",
    "the quick brown fox": "338f85c2-a6-3d-09",
    "market report #42: NEAR Intents volume up 18% w/w": "1fd2a2d5-d4-41-16",
    "": "811c9dc5-00-00-00",
    "a": "e40c292c-61-37-01",
}


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:] or ["hello world"]:
        print(p, "->", commitment(p))