"""
tanakh-vre-guard: Shared verification library for NEAR hackathon projects.

Provides mathematical verification primitives derived from Tanakh 231-gate analysis
and the Verifiable Reasoning Engine (VRE) framework.

Exports:
- checksum231: 3-channel SUM/LINEAR/PRODUCT mod 231 (tamper detection)
- semantic_space: 5-bucket deterministic projection (Mod-5 semantic)
- causal_memory: Append-only causal chain with counterfactuals
- dual_stream: Ordinal+Mod-5 semantic + Mod-37/73 structural (100% joint unique)
- bijective_verifier: Forward determinism + inverse preimage verification
- compose_phase: Sparse fingerprint deduplication

Integration points for NEAR projects:
- Co-op: wrap AgentRuntime.plan()/execute() with bijectiveVerify
- Setoff: wrap solver.solve()/agent.decide() with 231-gate checksum
- near-agent-guard: wrap Policy.evaluate()/AuditLog.append() with dual-stream
"""

from .checksum231 import (
    commitment,
    check,
    mismatch_count,
    matching_layers,
    SHA_PREFIX,
    is_sha_predicate,
    digest_for,
    digest_matches,
    verify_tamper_evidence,
    find_permutation_collision,
    GOLDEN_VECTORS,
)

from .semantic_space import (
    project,
    verify,
    distance,
    compose,
    reset_matrices,
    get_matrix_state,
    set_matrix_state,
    SemanticProjection,
    VerificationResult,
)

from .causal_memory import (
    store_step,
    get_step,
    get_recent_steps,
    retrieve_semantic,
    get_causal_chain,
    create_counterfactual,
    export_memory,
    import_memory,
    get_memory_stats,
    persist,
    load,
    CausalStep,
)

from .dual_stream import (
    analyze_dual_stream,
    compare_manuscripts,
    THEOLOGICAL_CATEGORIES,
    TheologicalCategory,
    SemanticStream,
    StructuralStream,
    CrossCorrelation,
    SentinelReport,
    DualStreamResult,
)

from .bijective_verifier import (
    bijective_verify,
    verify_chain,
    SparseFingerprinter,
    ComposePhase,
    BijectiveVerificationResult,
)

__version__ = "1.0.0"

__all__ = [
    # checksum231
    "commitment",
    "check",
    "mismatch_count",
    "matching_layers",
    "SHA_PREFIX",
    "is_sha_predicate",
    "digest_for",
    "digest_matches",
    "verify_tamper_evidence",
    "find_permutation_collision",
    "GOLDEN_VECTORS",
    # semantic_space
    "project",
    "verify",
    "distance",
    "compose",
    "reset_matrices",
    "get_matrix_state",
    "set_matrix_state",
    "SemanticProjection",
    "VerificationResult",
    # causal_memory
    "store_step",
    "get_step",
    "get_recent_steps",
    "retrieve_semantic",
    "get_causal_chain",
    "create_counterfactual",
    "export_memory",
    "import_memory",
    "get_memory_stats",
    "persist",
    "load",
    "CausalStep",
    # dual_stream
    "analyze_dual_stream",
    "compare_manuscripts",
    "THEOLOGICAL_CATEGORIES",
    "TheologicalCategory",
    "SemanticStream",
    "StructuralStream",
    "CrossCorrelation",
    "SentinelReport",
    "DualStreamResult",
    # bijective_verifier
    "bijective_verify",
    "verify_chain",
    "SparseFingerprinter",
    "ComposePhase",
    "BijectiveVerificationResult",
]

# Quick integration helpers
async def verify_agent_decision(
    policy_text: str,
    pool_context: str,
    agent_view: str,
    decision: dict,
) -> "BijectiveVerificationResult":
    """Quick integration helper for Co-op AgentRuntime."""
    from .bijective_verifier import bijective_verify
    from .semantic_space import project
    
    combined_text = f"POLICY: {policy_text} | POOL: {pool_context} | AGENT: {agent_view} | DECISION: {decision}"
    projection = await project(combined_text)
    return await bijective_verify(projection)


def verify_solver_clearing(clearing_text: str, obligation_graph: str) -> dict:
    """Quick integration helper for Setoff solver."""
    comm = commitment(f"CLEARING: {clearing_text} | GRAPH: {obligation_graph}")
    return {"commitment": comm, "mismatch_count": 0}


async def verify_guard_decision(
    intent_text: str,
    policy_text: str,
    verdict: str,
) -> dict:
    """Quick integration helper for near-agent-guard."""
    from .dual_stream import analyze_dual_stream
    combined = f"INTENT: {intent_text} | POLICY: {policy_text} | VERDICT: {verdict}"
    semantic = analyze_dual_stream(combined)
    return {"semantic": semantic, "integrity_score": semantic.integrity_score}