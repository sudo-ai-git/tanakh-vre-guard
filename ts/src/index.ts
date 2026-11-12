/**
 * @near/tanakh-vre-guard — Shared verification library for NEAR hackathon projects.
 * 
 * Provides mathematical verification primitives derived from Tanakh 231-gate analysis
 * and the Verifiable Reasoning Engine (VRE) framework.
 * 
 * Exports:
 * - checksum231: 3-channel SUM/LINEAR/PRODUCT mod 231 (tamper detection)
 * - semanticSpace: 5-bucket deterministic projection (Mod-5 semantic)
 * - causalMemory: Append-only causal chain with counterfactuals
 * - dualStream: Ordinal+Mod-5 semantic + Mod-37/73 structural (100% joint unique)
 * - bijectiveVerifier: Forward determinism + inverse preimage verification
 * - composePhase: Sparse fingerprint deduplication
 * 
 * Integration points for NEAR projects:
 * - Co-op: wrap AgentRuntime.plan()/execute() with bijectiveVerify
 * - Setoff: wrap solver.solve()/agent.decide() with 231-gate checksum
 * - near-agent-guard: wrap Policy.evaluate()/AuditLog.append() with dual-stream
 */

// 231-Gate Three-Channel Checksum
export {
  commitment,
  check,
  mismatchCount,
  matchingLayers,
  brokenMask,
  SHA_PREFIX,
  isShaPredicate,
  digestFor,
  digestMatches,
  verifyTamperEvidence,
  findPermutationCollision,
  GOLDEN_VECTORS,
} from "./checksum231.js";

// SemanticSpace: 5-bucket deterministic projection
export {
  project,
  verify,
  distance,
  compose,
  resetMatrices,
  getMatrixState,
  setMatrixState,
  type SemanticProjection,
  type VerificationResult,
} from "./semanticSpace.js";

// CausalMemory: Append-only causal chain
export {
  storeStep,
  getStep,
  getRecentSteps,
  retrieveSemantic,
  getCausalChain,
  createCounterfactual,
  exportMemory,
  importMemory,
  getMemoryStats,
  persist,
  load,
  type CausalStep,
} from "./causalMemory.js";

// DualStreamGematria: Ordinal+Mod-5 + Mod-37/73
export {
  analyzeDualStream,
  compareManuscripts,
  THEOLOGICAL_CATEGORIES,
  type TheologicalCategory,
  type SemanticStream,
  type StructuralStream,
  type CrossCorrelation,
  type SentinelReport,
  type DualStreamResult,
} from "./dualStream.js";

// BijectiveVerifier: Forward determinism + inverse preimage
export {
  bijectiveVerify,
  verifyChain,
  SparseFingerprinter,
  ComposePhase,
  type BijectiveVerificationResult,
} from "./bijectiveVerifier.js";

// Version
export const VERSION = "1.0.0";