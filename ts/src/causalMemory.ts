/**
 * CausalMemory: Append-only causal chain with counterfactual branching.
 * 
 * Stores reasoning steps as nodes in a DAG with:
 * - Parent→child links (causal provenance)
 * - Counterfactual branches (what-if exploration)
 * - SemanticSpace projections for semantic retrieval
 * - VerificationResult for trust tracking
 * 
 * Used by VRE for:
 * - Memory alignment (Layer 2 of self-reflection)
 * - Counterfactual exploration
 * - Semantic retrieval by bucket distance
 * - Compose phase fingerprint matching
 */

// Declare Node.js globals for TypeScript
declare const process: {
  versions?: { node?: string };
} | undefined;
declare const require: (module: string) => any;

import { SemanticProjection, VerificationResult, distance } from "./semanticSpace.js";

export interface CausalStep {
  id: string;
  timestamp: number;
  text: string;
  projection: SemanticProjection;
  verification: VerificationResult;
  parents: string[];           // Causal parents (must have ≥1 for training)
  children: string[];          // Causal children
  counterfactuals: string[];   // Alternative branches from this step
  metadata: {
    source: "agent" | "contract" | "oracle" | "human" | "system" | "counterfactual";
    action?: string;           // e.g., "join_pool", "settle", "evaluate"
    txHash?: string;           // On-chain transaction hash
    poolId?: number;           // For Co-op/Setoff context
    oracleType?: string;       // For oracle-grounded steps
    factualEngine?: string;    // "bijective" | "factual" | "none"
    factualTs?: number;        // Timestamp of factual verification
  };
}

/** In-memory index (persisted to localStorage/IndexedDB in production) */
class CausalMemoryIndex {
  private steps: Map<string, CausalStep> = new Map();
  private byTimestamp: string[] = [];  // Sorted by timestamp
  private byPool: Map<number, Set<string>> = new Map();
  private bySource: Map<string, Set<string>> = new Map();
  private maxSteps = 200;  // Timestamp window (matches memory_replay daemon)

  /** Generate deterministic ID from content + timestamp */
  generateId(text: string, timestamp: number, parents: string[]): string {
    const parentHash = parents.sort().join("|");
    const seed = `${text}|${timestamp}|${parentHash}`;
    // Simple hash for ID
    let hash = 0;
    for (let i = 0; i < seed.length; i++) {
      hash = ((hash << 5) - hash + seed.charCodeAt(i)) | 0;
    }
    return `step_${Math.abs(hash).toString(36)}_${timestamp.toString(36)}`;
  }

  /** Store a new step (returns the step ID) */
  store(step: Omit<CausalStep, "id">): string {
    const id = this.generateId(step.text, step.timestamp, step.parents);
    const fullStep: CausalStep = { ...step, id };
    
    // Enforce parent requirement for training eligibility
    if (step.parents.length === 0 && step.metadata.source !== "system") {
      console.warn(`CausalMemory: step ${id} has no parents; will not enter replay training`);
    }
    
    this.steps.set(id, fullStep);
    this.byTimestamp.push(id);
    this.byTimestamp.sort((a, b) => 
      this.steps.get(a)!.timestamp - this.steps.get(b)!.timestamp
    );
    
    // Index by pool
    if (step.metadata.poolId !== undefined) {
      const set = this.byPool.get(step.metadata.poolId) || new Set();
      set.add(id);
      this.byPool.set(step.metadata.poolId, set);
    }
    
    // Index by source
    const srcSet = this.bySource.get(step.metadata.source) || new Set();
    srcSet.add(id);
    this.bySource.set(step.metadata.source, srcSet);
    
    // Link parents
    for (const parentId of step.parents) {
      const parent = this.steps.get(parentId);
      if (parent) {
        parent.children.push(id);
      }
    }
    
    // Evict oldest if over limit
    if (this.byTimestamp.length > this.maxSteps) {
      const oldest = this.byTimestamp.shift()!;
      this.steps.delete(oldest);
      // Clean up indices
      for (const set of this.byPool.values()) set.delete(oldest);
      for (const set of this.bySource.values()) set.delete(oldest);
    }
    
    return id;
  }

  /** Retrieve a step by ID */
  get(id: string): CausalStep | undefined {
    return this.steps.get(id);
  }

  /** Get recent steps (for context retrieval) */
  getRecent(count: number = 10): CausalStep[] {
    return this.byTimestamp
      .slice(-count)
      .map(id => this.steps.get(id)!)
      .filter(Boolean);
  }

  /** Get steps by pool ID */
  getByPool(poolId: number): CausalStep[] {
    const ids = this.byPool.get(poolId) || new Set();
    return Array.from(ids).map(id => this.steps.get(id)!).filter(Boolean);
  }

  /** Get steps by source */
  getBySource(source: string): CausalStep[] {
    const ids = this.bySource.get(source) || new Set();
    return Array.from(ids).map(id => this.steps.get(id)!).filter(Boolean);
  }

  /** Semantic retrieval: find steps within bucket distance threshold */
  retrieveBySemantic(query: SemanticProjection, threshold: number = 0.3, limit: number = 5): CausalStep[] {
    const results: { step: CausalStep; dist: number }[] = [];
    
    for (const step of this.steps.values()) {
      const dist = distance(query, step.projection);
      if (dist <= threshold) {
        results.push({ step, dist });
      }
    }
    
    results.sort((a, b) => a.dist - b.dist);
    return results.slice(0, limit).map(r => r.step);
  }

  /** Get causal chain from a step back to roots */
  getCausalChain(id: string): CausalStep[] {
    const chain: CausalStep[] = [];
    let current = this.steps.get(id);
    while (current) {
      chain.push(current);
      if (current.parents.length === 0) break;
      // Follow first parent (could be enhanced to follow all)
      current = this.steps.get(current.parents[0]);
    }
    return chain.reverse(); // Root to leaf
  }

  /** Get all descendants of a step */
  getDescendants(id: string): CausalStep[] {
    const descendants: CausalStep[] = [];
    const queue = [id];
    
    while (queue.length > 0) {
      const currentId = queue.shift()!;
      const current = this.steps.get(currentId);
      if (!current) continue;
      
      for (const childId of current.children) {
        const child = this.steps.get(childId);
        if (child) {
          descendants.push(child);
          queue.push(childId);
        }
      }
    }
    
    return descendants;
  }

  /** Create counterfactual branch from a step */
  createCounterfactual(
    parentId: string, 
    counterfactualText: string,
    projection: SemanticProjection,
    verification: VerificationResult,
    metadata: CausalStep["metadata"]
  ): string {
    const parent = this.steps.get(parentId);
    if (!parent) throw new Error(`Parent step ${parentId} not found`);
    
    const timestamp = Date.now();
    const counterfactualId = this.generateId(counterfactualText, timestamp, [parentId]);
    
    const step: CausalStep = {
      id: counterfactualId,
      timestamp,
      text: counterfactualText,
      projection,
      verification,
      parents: [parentId],
      children: [],
      counterfactuals: [],
      metadata: { ...metadata, source: "counterfactual" },
    };
    
    this.steps.set(counterfactualId, step);
    parent.counterfactuals.push(counterfactualId);
    
    return counterfactualId;
  }

  /** Export for persistence (serializable) */
  export(): CausalStep[] {
    return Array.from(this.steps.values());
  }

  /** Import from persistence */
  import(steps: CausalStep[]): void {
    this.steps.clear();
    this.byTimestamp = [];
    this.byPool.clear();
    this.bySource.clear();
    
    for (const step of steps) {
      this.steps.set(step.id, step);
      this.byTimestamp.push(step.id);
      
      if (step.metadata.poolId !== undefined) {
        const set = this.byPool.get(step.metadata.poolId) || new Set();
        set.add(step.id);
        this.byPool.set(step.metadata.poolId, set);
      }
      
      const srcSet = this.bySource.get(step.metadata.source) || new Set();
      srcSet.add(step.id);
      this.bySource.set(step.metadata.source, srcSet);
    }
    
    this.byTimestamp.sort((a, b) => 
      this.steps.get(a)!.timestamp - this.steps.get(b)!.timestamp
    );
  }

  /** Stats for monitoring */
  stats(): {
    totalSteps: number;
    bySource: Record<string, number>;
    byPool: Record<number, number>;
    avgChildren: number;
    maxDepth: number;
  } {
    const bySource: Record<string, number> = {};
    const byPool: Record<number, number> = {};
    let totalChildren = 0;
    let maxDepth = 0;
    
    for (const step of this.steps.values()) {
      bySource[step.metadata.source] = (bySource[step.metadata.source] || 0) + 1;
      if (step.metadata.poolId !== undefined) {
        byPool[step.metadata.poolId] = (byPool[step.metadata.poolId] || 0) + 1;
      }
      totalChildren += step.children.length;
      
      // Compute depth
      let depth = 0;
      let current = step;
      while (current.parents.length > 0) {
        depth++;
        current = this.steps.get(current.parents[0])!;
        if (!current) break;
      }
      maxDepth = Math.max(maxDepth, depth);
    }
    
    return {
      totalSteps: this.steps.size,
      bySource,
      byPool,
      avgChildren: totalChildren / this.steps.size || 0,
      maxDepth,
    };
  }
}

// Singleton instance
const causalMemory = new CausalMemoryIndex();

/** Store a verified reasoning step */
export function storeStep(
  text: string,
  projection: SemanticProjection,
  verification: VerificationResult,
  parents: string[],
  metadata: CausalStep["metadata"]
): string {
  return causalMemory.store({
    timestamp: Date.now(),
    text,
    projection,
    verification,
    parents,
    children: [],
    counterfactuals: [],
    metadata,
  });
}

/** Retrieve step by ID */
export function getStep(id: string): CausalStep | undefined {
  return causalMemory.get(id);
}

/** Get recent steps for context */
export function getRecentSteps(count: number = 10): CausalStep[] {
  return causalMemory.getRecent(count);
}

/** Semantic retrieval by bucket distance */
export function retrieveSemantic(
  query: SemanticProjection, 
  threshold: number = 0.3, 
  limit: number = 5
): CausalStep[] {
  return causalMemory.retrieveBySemantic(query, threshold, limit);
}

/** Get causal chain from step to root */
export function getCausalChain(id: string): CausalStep[] {
  return causalMemory.getCausalChain(id);
}

/** Create counterfactual branch */
export function createCounterfactual(
  parentId: string,
  counterfactualText: string,
  projection: SemanticProjection,
  verification: VerificationResult,
  metadata: CausalStep["metadata"]
): string {
  return causalMemory.createCounterfactual(parentId, counterfactualText, projection, verification, metadata);
}

/** Export all steps for persistence */
export function exportMemory(): CausalStep[] {
  return causalMemory.export();
}

/** Import steps from persistence */
export function importMemory(steps: CausalStep[]): void {
  causalMemory.import(steps);
}

/** Memory stats */
export function getMemoryStats(): ReturnType<CausalMemoryIndex["stats"]> {
  return causalMemory.stats();
}

/** Persist to localStorage (browser) or file (Node) */
export async function persist(key: string = "causal_memory"): Promise<void> {
  const data = JSON.stringify(exportMemory());
  // Declare localStorage for TypeScript
  const ls = typeof globalThis !== "undefined" && (globalThis as any).localStorage;
  if (ls) {
    ls.setItem(key, data);
  } else if (typeof process !== "undefined" && process.versions?.node) {
    // Node.js environment
    try {
      const fs = require("fs");
      fs.writeFileSync(`${key}.json`, data);
    } catch {
      console.warn("CausalMemory: fs not available, cannot persist to file");
    }
  } else {
    console.warn("CausalMemory: not in browser or Node.js, cannot persist");
  }
}

/** Load from persistence */
export async function load(key: string = "causal_memory"): Promise<void> {
  let data: string;
  const ls = typeof globalThis !== "undefined" && (globalThis as any).localStorage;
  if (ls) {
    data = ls.getItem(key) || "[]";
  } else if (typeof process !== "undefined" && process.versions?.node) {
    try {
      const fs = require("fs");
      data = fs.readFileSync(`${key}.json`, "utf-8");
    } catch {
      data = "[]";
    }
  } else {
    data = "[]";
  }
  importMemory(JSON.parse(data));
}