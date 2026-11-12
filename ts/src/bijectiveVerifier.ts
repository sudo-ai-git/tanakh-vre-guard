/**
 * BijectiveVerifier: Forward determinism + inverse preimage enumeration.
 * 
 * Verifies SemanticSpace projections by:
 * 1. Forward pass: re-project text → same bucket signature?
 * 2. Inverse: enumerate preimages mapping to same signature
 * 3. Confidence: unique preimage = high confidence
 * 
 * This is the "structural checksum" layer of VRE - deterministic, not factual.
 * A flat lie passes bijective verification; factual truth needs oracle grounding.
 */

import { SemanticProjection, VerificationResult } from "./semanticSpace.js";
import { project, verify as semanticVerify, distance, compose } from "./semanticSpace.js";

/** Bijective verification result with detailed breakdown */
export interface BijectiveVerificationResult extends VerificationResult {
  forwardPass: boolean;
  inversePreimageCount: number;
  preimageSamples: string[];      // Sample texts with same signature
  semanticDrift: number;          // Distance from nearest neighbor
  collisionResistance: number;    // Bits of entropy in signature
}

/** 
 * Verify a projection bijectively.
 * - Re-runs forward projection (deterministic check)
 * - Enumerates inverse preimages in neighborhood
 * - Measures collision resistance
 */
export async function bijectiveVerify(
  projection: SemanticProjection,
  neighborhoodRadius: number = 10
): Promise<BijectiveVerificationResult> {
  // 1. Forward determinism check
  const forwardResult = await semanticVerify(projection);
  
  // 2. Inverse preimage enumeration
  const preimageSamples: string[] = [projection.text];
  let inversePreimageCount = 1;
  
  // Single-character mutations
  for (let i = 0; i < Math.min(projection.text.length, neighborhoodRadius); i++) {
    for (const ch of "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?") {
      if (ch === projection.text[i]) continue;
      const mut = projection.text.slice(0, i) + ch + projection.text.slice(i + 1);
      const mutProj = await project(mut);
      if (mutProj.bucketSignature === projection.bucketSignature) {
        inversePreimageCount++;
        if (preimageSamples.length < 5) preimageSamples.push(mut);
      }
    }
  }
  
  // Transposition mutations (adjacent swaps)
  for (let i = 0; i < Math.min(projection.text.length - 1, 5); i++) {
    if (projection.text[i] !== projection.text[i + 1]) {
      const mut = projection.text.slice(0, i) + projection.text[i + 1] + projection.text[i] + projection.text.slice(i + 2);
      const mutProj = await project(mut);
      if (mutProj.bucketSignature === projection.bucketSignature) {
        inversePreimageCount++;
        if (preimageSamples.length < 5) preimageSamples.push(mut);
      }
    }
  }
  
  // 3. Semantic drift to nearest neighbor with different text
  let semanticDrift = 1.0;
  // Sample random mutations to find nearest different signature
  for (let i = 0; i < 100; i++) {
    const mut = randomMutation(projection.text);
    const mutProj = await project(mut);
    if (mutProj.bucketSignature !== projection.bucketSignature) {
      const d = distance(projection, mutProj);
      if (d < semanticDrift) semanticDrift = d;
    }
  }
  
  // 4. Collision resistance (entropy bits)
  // 5 buckets, distribution entropy
  const entropy = -projection.buckets.reduce((s, p) => s + (p > 0 ? p * Math.log2(p) : 0), 0);
  const collisionResistance = entropy * 5;  // Rough: entropy per bucket * num buckets
  
  // Confidence: high if forward pass + unique preimage
  const confidence = forwardResult.forwardPass && inversePreimageCount === 1 ? 100 :
                     forwardResult.forwardPass ? 80 : 0;
  
  return {
    isValid: forwardResult.forwardPass,
    forwardPass: forwardResult.forwardPass,
    inversePreimageCount,
    preimageSamples,
    semanticDrift,
    collisionResistance,
    confidence,
  };
}

/** Verify a chain of projections (composed reasoning) */
export async function verifyChain(projections: SemanticProjection[]): Promise<{
  allValid: boolean;
  results: BijectiveVerificationResult[];
  compositionDrift: number;
}> {
  const results: BijectiveVerificationResult[] = [];
  let allValid = true;
  
  for (const proj of projections) {
    const result = await bijectiveVerify(proj);
    results.push(result);
    if (!result.isValid) allValid = false;
  }
  
  // Composition drift: distance between composed and sequential
  if (projections.length >= 2) {
    const composed = compose(projections[0], projections[1]);
    const sequential = await project(projections[0].text + " " + projections[1].text);
    const compositionDrift = distance(composed, sequential);
    return { allValid, results, compositionDrift };
  }
  
  return { allValid, results, compositionDrift: 0 };
}

/** Generate random mutation of text */
function randomMutation(text: string): string {
  const chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?";
  const idx = Math.floor(Math.random() * text.length);
  const ch = chars[Math.floor(Math.random() * chars.length)];
  return text.slice(0, idx) + ch + text.slice(idx + 1);
}

/** 
 * SparseFingerprinter: Top-k Jaccard for compose phase deduplication.
 * Used in VRE compose phase to skip redundant LLM calls.
 */
export class SparseFingerprinter {
  private k: number;
  private projections: Map<string, SemanticProjection> = new Map();
  
  constructor(k: number = 16) {
    this.k = k;
  }
  
  /** Add projection, return fingerprint */
  add(projection: SemanticProjection): string {
    const fp = this.fingerprint(projection);
    this.projections.set(fp, projection);
    return fp;
  }
  
  /** Compute top-k fingerprint from bucket signature */
  fingerprint(projection: SemanticProjection): string {
    // Extract top-k bucket indices by weight
    const indexed = projection.buckets
      .map((w, i) => ({ index: i, weight: w }))
      .sort((a, b) => b.weight - a.weight)
      .slice(0, this.k)
      .map(x => `${x.index}:${x.weight.toFixed(3)}`)
      .join("|");
    return indexed;
  }
  
  /** Jaccard similarity between two fingerprints */
  jaccard(fp1: string, fp2: string): number {
    const set1 = new Set(fp1.split("|"));
    const set2 = new Set(fp2.split("|"));
    const intersection = [...set1].filter(x => set2.has(x)).length;
    const union = set1.size + set2.size - intersection;
    return union > 0 ? intersection / union : 0;
  }
  
  /** Find similar projections above threshold */
  findSimilar(fp: string, threshold: number = 0.7): string[] {
    const results: string[] = [];
    for (const [key] of this.projections) {
      if (this.jaccard(fp, key) >= threshold) {
        results.push(key);
      }
    }
    return results;
  }
  
  /** Check if projection is duplicate (for compose skip) */
  isDuplicate(projection: SemanticProjection, threshold: number = 0.9): boolean {
    const fp = this.fingerprint(projection);
    return this.findSimilar(fp, threshold).length > 0;
  }
}

/** 
 * ComposePhase: Sparse fingerprint-based deduplication for VRE compose phase.
 * Reduces LLM calls on repetitive outputs.
 */
export class ComposePhase {
  private fingerprinter: SparseFingerprinter;
  private skipThreshold: number;
  
  constructor(skipThreshold: number = 0.7) {
    this.fingerprinter = new SparseFingerprinter(16);
    this.skipThreshold = skipThreshold;
  }
  
  /** 
   * Apply compose phase: skip if similar projection exists.
   * Returns { skipped: boolean, reason?: string, similar?: string[] }
   */
  async apply(projection: SemanticProjection): Promise<{
    skipped: boolean;
    reason?: string;
    similar?: string[];
    fingerprint: string;
  }> {
    const fp = this.fingerprinter.fingerprint(projection);
    const similar = this.fingerprinter.findSimilar(fp, this.skipThreshold);
    
    if (similar.length > 0) {
      return {
        skipped: true,
        reason: `Similar projection found (Jaccard >= ${this.skipThreshold})`,
        similar,
        fingerprint: fp,
      };
    }
    
    this.fingerprinter.add(projection);
    return { skipped: false, fingerprint: fp };
  }
  
  /** Reset fingerprinter (for new session) */
  reset(): void {
    this.fingerprinter = new SparseFingerprinter(16);
  }
  
  /** Get stats */
  stats(): { totalProjections: number } {
    return { totalProjections: this.fingerprinter["projections"].size };
  }
}