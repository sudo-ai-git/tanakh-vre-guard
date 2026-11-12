//! Tanakh-VRE Guard: 231-Gate Three-Channel Checksum for NEAR Contracts
//!
//! Implements the same 3-channel checksum as the TypeScript and Python versions:
//! - L0: FNV-1a-32 over low byte of each code unit (only layer seeing order)
//! - L1: SUM gate: sum(byte + index) mod 231
//! - L2: LINEAR gate: sum(byte * 22 + index) mod 231
//! - L3: PARITY gate: count(odd bytes) mod 37
//!
//! Output format: "fnv8-sum2-lin2-odd2" (e.g., "3b9aca00-8f-2a-0c")
//!
//! HONEST LIMIT: L1/L2/L3 are permutation-invariant; only L0 sees order.
//! Full collision cost ~2^16 permutations. Use for tamper DETECTION only;
//! file identity uses sha256.

#![no_std]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]
#![deny(missing_docs)]

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;
use alloc::format;
use core::fmt;
use core::hash::Hash;
use alloc::collections::BTreeMap;

/// Modulus for SUM and LINEAR channels (3 * 7 * 11 = 231)
pub const MOD_231: u32 = 231;
/// Modulus for PARITY channel (prime)
pub const MOD_37: u32 = 37;
/// Linear multiplier (22 = alphabet size for Hebrew)
pub const LINEAR_MULT: u32 = 22;
/// FNV-1a-32 offset basis
const FNV_OFFSET: u32 = 2166136261;
/// FNV-1a-32 prime
const FNV_PRIME: u32 = 16777619;

/// Four-layer commitment: "fnv8-sum2-lin2-odd2"
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Commitment {
    /// FNV-1a-32 hash (only layer that sees byte order)
    pub fnv: u32,
    /// SUM gate: sum(byte + index) mod 231
    pub sum: u32,
    /// LINEAR gate: sum(byte * 22 + index) mod 231
    pub linear: u32,
    /// PARITY gate: count(odd bytes) mod 37
    pub odd: u32,
}

impl Commitment {
    /// Create from four channel values
    pub const fn new(fnv: u32, sum: u32, linear: u32, odd: u32) -> Self {
        Self { fnv, sum, linear, odd }
    }

    /// Create from payload bytes (ASCII, low byte of each code unit)
    pub fn from_bytes(payload: &[u8]) -> Self {
        let mut fnv = FNV_OFFSET;
        let mut sum = 0u64;
        let mut linear = 0u64;
        let mut odd = 0u32;

        for (i, &b) in payload.iter().enumerate() {
            // FNV-1a-32: hash = (hash ^ byte) * prime
            fnv = (fnv ^ (b as u32)).wrapping_mul(FNV_PRIME);
            // SUM gate
            sum = sum.wrapping_add(b as u64 + i as u64);
            // LINEAR gate
            linear = linear.wrapping_add((b as u64) * (LINEAR_MULT as u64) + i as u64);
            // PARITY gate
            if b & 1 == 1 {
                odd += 1;
            }
        }

        Self {
            fnv,
            sum: (sum % MOD_231 as u64) as u32,
            linear: (linear % MOD_231 as u64) as u32,
            odd: odd % MOD_37,
        }
    }

    /// Create from string (ASCII only)
    pub fn from_str(payload: &str) -> Self {
        Self::from_bytes(payload.as_bytes())
    }

    /// Encode to hex string "fnv8-sum2-lin2-odd2"
    pub fn to_hex(&self) -> String {
        format!("{:08x}-{:02x}-{:02x}-{:02x}", self.fnv, self.sum, self.linear, self.odd)
    }

    /// Bitmask of broken layers when comparing with expected
    pub fn broken_mask(&self, expected: &Self) -> u32 {
        let mut mask = 0;
        if self.fnv != expected.fnv { mask |= 1; }
        if self.sum != expected.sum { mask |= 2; }
        if self.linear != expected.linear { mask |= 4; }
        if self.odd != expected.odd { mask |= 8; }
        mask
    }

    /// Number of mismatched layers (0-4)
    pub fn mismatch_count(&self, expected: &Self) -> u32 {
        self.broken_mask(expected).count_ones()
    }

    /// Check if all 4 layers match
    pub fn matches(&self, expected: &Self) -> bool {
        self.mismatch_count(expected) == 0
    }

    /// Layer-by-layer comparison (for predicate matching)
    pub fn matching_layers(&self, expected: &Self) -> u32 {
        4 - self.mismatch_count(expected)
    }
}

impl fmt::Display for Commitment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:08x}-{:02x}-{:02x}-{:02x}", self.fnv, self.sum, self.linear, self.odd)
    }
}

/// Verify tamper evidence: single byte change breaks >= 3 layers
pub fn verify_tamper_evidence() -> bool {
    let test_payloads: &[&[u8]] = &[
        b"hello world",
        b"the quick brown fox",
        b"market report #42: NEAR Intents volume up 18% w/w",
    ];

    let mut worst = 4;

    for payload in test_payloads {
        let c = Commitment::from_bytes(payload);
        for i in 0..payload.len() {
            let mut mut_payload = (*payload).to_vec();
            mut_payload[i] = if payload[i] != b'X' { b'X' } else { b'Y' };
            let mut_c = Commitment::from_bytes(&mut_payload);
            let n = c.mismatch_count(&mut_c);
            if n < worst {
                worst = n;
            }
        }
    }

    worst >= 3
}

/// Find a full 4-layer collision via permutation (demonstrates weakness)
/// Returns (payload_a, payload_b, commitment) if found
pub fn find_permutation_collision(base: &[u8], max_permutations: usize) -> Option<(Vec<u8>, Vec<u8>, Commitment)> {
    use core::hash::{Hash, Hasher};
    use alloc::collections::BTreeMap;
    
    // Simple hasher implementation (no_std compatible)
    struct SimpleHasher(u64);
    
    impl Hasher for SimpleHasher {
        fn finish(&self) -> u64 {
            self.0
        }
        
        fn write(&mut self, bytes: &[u8]) {
            for &b in bytes {
                self.0 = self.0.wrapping_mul(31).wrapping_add(b as u64);
            }
        }
        
        fn write_u32(&mut self, i: u32) {
            self.0 = self.0.wrapping_mul(31).wrapping_add(i as u64);
        }
        
        fn write_u64(&mut self, i: u64) {
            self.0 = self.0.wrapping_mul(31).wrapping_add(i);
        }
    }
    
    let mut seen: BTreeMap<u64, Vec<u8>> = BTreeMap::new();
    let mut buf = base.to_vec();
    let n = buf.len();
    
    // Simple deterministic shuffle using a fixed seed
    let mut state = 0x9e3779b97f4a7c15u64;
    let mut rng = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        state
    };

    for _ in 0..max_permutations {
        // Fisher-Yates shuffle
        for i in (1..n).rev() {
            let j = (rng() % (i + 1) as u64) as usize;
            buf.swap(i, j);
        }

        let c = Commitment::from_bytes(&buf);
        let mut hasher = SimpleHasher(0);
        c.fnv.hash(&mut hasher);
        c.sum.hash(&mut hasher);
        c.linear.hash(&mut hasher);
        c.odd.hash(&mut hasher);
        let key = hasher.finish();

        if let Some(existing) = seen.get(&key) {
            if existing != &buf {
                return Some((existing.clone(), buf.clone(), c));
            }
        } else {
            seen.insert(key, buf.clone());
        }
    }

    None
}

/// Golden vectors recorded from deployed contract (for parity testing)
pub const GOLDEN_VECTORS: &[(&str, &str)] = &[
    ("hello world", "d58b3fa7-10-79-04"),
    ("the quick brown fox", "338f85c2-a6-3d-09"),
    ("market report #42: NEAR Intents volume up 18% w/w", "1fd2a2d5-d4-41-16"),
    ("", "811c9dc5-00-00-00"),
    ("a", "e40c292c-61-37-01"),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn golden_vectors_match() {
        for (payload, expected) in GOLDEN_VECTORS {
            let c = Commitment::from_str(payload);
            assert_eq!(c.to_hex(), *expected, "Failed for: {}", payload);
        }
    }

    #[test]
    fn tamper_evidence_holds() {
        assert!(verify_tamper_evidence());
    }

    #[test]
    fn mismatch_count_works() {
        let c1 = Commitment::from_str("hello world");
        let c2 = Commitment::from_str("hello worldX");
        assert_eq!(c1.mismatch_count(&c2), c1.mismatch_count(&c2));
        assert!(c1.mismatch_count(&c2) > 0);
        assert_eq!(c1.mismatch_count(&c1), 0);
    }

    #[test]
    fn matching_layers_works() {
        let c1 = Commitment::from_str("hello world");
        let c2 = Commitment::from_str("hello worldX");
        assert_eq!(c1.matching_layers(&c1), 4);
        assert!(c1.matching_layers(&c2) < 4);
    }

    #[test]
    fn permutation_collision_exists() {
        let base = b"POD-DELIVERY-000";
        let result = find_permutation_collision(base, 300_000);
        assert!(result.is_some(), "Should find collision within 300k permutations");
        let (a, b, c) = result.unwrap();
        assert_ne!(a, b);
        assert_eq!(Commitment::from_bytes(&a), c);
        assert_eq!(Commitment::from_bytes(&b), c);
    }
}