"""
CausalMemory: Append-only causal chain with counterfactual branching.

Stores reasoning steps as nodes in a DAG with:
- Parent→child links (causal provenance)
- Counterfactual branches (what-if exploration)
- SemanticSpace projections for semantic retrieval
- VerificationResult for trust tracking

Used by VRE for:
- Memory alignment (Layer 2 of self-reflection)
- Counterfactual exploration
- Semantic retrieval by bucket distance
- Compose phase fingerprint matching
"""

import json
import time
from dataclasses import dataclass, field
from typing import Any
from collections import defaultdict

from .semantic_space import SemanticProjection, VerificationResult, distance


@dataclass
class CausalStep:
    id: str
    timestamp: int
    text: str
    projection: SemanticProjection
    verification: VerificationResult
    parents: list[str] = field(default_factory=list)
    children: list[str] = field(default_factory=list)
    counterfactuals: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class CausalMemoryIndex:
    """In-memory index (persisted to localStorage/IndexedDB in production)."""
    
    def __init__(self, max_steps: int = 200):
        self.steps: dict[str, CausalStep] = {}
        self.by_timestamp: list[str] = []
        self.by_pool: dict[int, set[str]] = defaultdict(set)
        self.by_source: dict[str, set[str]] = defaultdict(set)
        self.max_steps = max_steps
    
    def generate_id(self, text: str, timestamp: int, parents: list[str]) -> str:
        parent_hash = "|".join(sorted(parents))
        seed = f"{text}|{timestamp}|{parent_hash}"
        h = 0
        for ch in seed:
            h = ((h << 5) - h + ord(ch)) & 0xFFFFFFFF
        return f"step_{abs(h):x}_{timestamp:x}"
    
    def store(self, step: CausalStep) -> str:
        # Enforce parent requirement for training eligibility
        if not step.parents and step.metadata.get("source") != "system":
            print(f"CausalMemory: step {step.id} has no parents; will not enter replay training")
        
        self.steps[step.id] = step
        self.by_timestamp.append(step.id)
        self.by_timestamp.sort(key=lambda id: self.steps[id].timestamp)
        
        if step.metadata.get("pool_id") is not None:
            self.by_pool[step.metadata["pool_id"]].add(step.id)
        
        src = step.metadata.get("source", "unknown")
        self.by_source[src].add(step.id)
        
        # Link parents
        for parent_id in step.parents:
            parent = self.steps.get(parent_id)
            if parent:
                parent.children.append(step.id)
        
        # Evict oldest if over limit
        if len(self.by_timestamp) > self.max_steps:
            oldest = self.by_timestamp.pop(0)
            self.steps.pop(oldest, None)
            for s in self.by_pool.values():
                s.discard(oldest)
            for s in self.by_source.values():
                s.discard(oldest)
        
        return step.id
    
    def get(self, id: str) -> CausalStep | None:
        return self.steps.get(id)
    
    def get_recent(self, count: int = 10) -> list[CausalStep]:
        return [self.steps[id] for id in self.by_timestamp[-count:] if id in self.steps]
    
    def get_by_pool(self, pool_id: int) -> list[CausalStep]:
        ids = self.by_pool.get(pool_id, set())
        return [self.steps[id] for id in ids if id in self.steps]
    
    def get_by_source(self, source: str) -> list[CausalStep]:
        ids = self.by_source.get(source, set())
        return [self.steps[id] for id in ids if id in self.steps]
    
    def retrieve_by_semantic(self, query: SemanticProjection, threshold: float = 0.3, limit: int = 5) -> list[CausalStep]:
        results = []
        for step in self.steps.values():
            d = distance(query, step.projection)
            if d <= threshold:
                results.append((step, d))
        results.sort(key=lambda x: x[1])
        return [r[0] for r in results[:limit]]
    
    def get_causal_chain(self, id: str) -> list[CausalStep]:
        chain = []
        current = self.steps.get(id)
        while current:
            chain.append(current)
            if not current.parents:
                break
            current = self.steps.get(current.parents[0])
        return list(reversed(chain))
    
    def get_descendants(self, id: str) -> list[CausalStep]:
        descendants = []
        queue = [id]
        while queue:
            current_id = queue.pop(0)
            current = self.steps.get(current_id)
            if not current:
                continue
            for child_id in current.children:
                child = self.steps.get(child_id)
                if child:
                    descendants.append(child)
                    queue.append(child_id)
        return descendants
    
    def create_counterfactual(self, parent_id: str, counterfactual_text: str,
                               projection: 'SemanticProjection',
                               verification: 'VerificationResult',
                               metadata: dict[str, Any]) -> str:
        parent = self.steps.get(parent_id)
        if not parent:
            raise ValueError(f"Parent step {parent_id} not found")
        
        timestamp = int(time.time() * 1_000_000)
        counterfactual_id = self.generate_id(counterfactual_text, timestamp, [parent_id])
        
        step = CausalStep(
            id=counterfactual_id,
            timestamp=timestamp,
            text=counterfactual_text,
            projection=projection,
            verification=verification,
            parents=[parent_id],
            children=[],
            counterfactuals=[],
            metadata={**metadata, "source": "counterfactual"},
        )
        
        self.steps[counterfactual_id] = step
        parent.counterfactuals.append(counterfactual_id)
        
        return counterfactual_id
    
    def generate_id(self, text: str, timestamp: int, parents: list[str]) -> str:
        parent_hash = "|".join(sorted(parents))
        seed = f"{text}|{timestamp}|{parent_hash}"
        h = 0
        for ch in seed:
            h = ((h << 5) - h + ord(ch)) & 0xFFFFFFFF
        return f"step_{abs(h):x}_{timestamp:x}"
    
    def export(self) -> list[CausalStep]:
        return list(self.steps.values())
    
    def import_steps(self, steps: list[CausalStep]) -> None:
        self.steps.clear()
        self.by_timestamp.clear()
        self.by_pool.clear()
        self.by_source.clear()
        
        for step in steps:
            self.steps[step.id] = step
            self.by_timestamp.append(step.id)
            
            if step.metadata.get("pool_id") is not None:
                self.by_pool[step.metadata["pool_id"]].add(step.id)
            
            src = step.metadata.get("source", "unknown")
            self.by_source[src].add(step.id)
        
        self.by_timestamp.sort(key=lambda id: self.steps[id].timestamp)
    
    def stats(self) -> dict[str, Any]:
        by_source = defaultdict(int)
        by_pool = defaultdict(int)
        total_children = 0
        max_depth = 0
        
        for step in self.steps.values():
            by_source[step.metadata.get("source", "unknown")] += 1
            if step.metadata.get("pool_id") is not None:
                by_pool[step.metadata["pool_id"]] += 1
            total_children += len(step.children)
            
            depth = 0
            current = step
            while current.parents:
                depth += 1
                current = self.steps.get(current.parents[0])
                if not current:
                    break
            max_depth = max(max_depth, depth)
        
        return {
            "total_steps": len(self.steps),
            "by_source": dict(by_source),
            "by_pool": dict(by_pool),
            "avg_children": total_children / len(self.steps) if self.steps else 0,
            "max_depth": max_depth,
        }


# Singleton instance
_causal_memory = CausalMemoryIndex()


def store_step(
    text: str,
    projection: 'SemanticProjection',
    verification: 'VerificationResult',
    parents: list[str],
    metadata: dict[str, Any],
) -> str:
    step = CausalStep(
        id="",  # Will be set in store
        timestamp=int(time.time() * 1_000_000),
        text=text,
        projection=projection,
        verification=verification,
        parents=parents,
        children=[],
        counterfactuals=[],
        metadata=metadata,
    )
    return _causal_memory.store(step)


def get_step(id: str) -> CausalStep | None:
    return _causal_memory.get(id)


def get_recent_steps(count: int = 10) -> list[CausalStep]:
    return _causal_memory.get_recent(count)


def retrieve_semantic(query: 'SemanticProjection', threshold: float = 0.3, limit: int = 5) -> list[CausalStep]:
    return _causal_memory.retrieve_by_semantic(query, threshold, limit)


def get_causal_chain(id: str) -> list[CausalStep]:
    return _causal_memory.get_causal_chain(id)


def create_counterfactual(
    parent_id: str,
    counterfactual_text: str,
    projection: 'SemanticProjection',
    verification: 'VerificationResult',
    metadata: dict[str, Any],
) -> str:
    return _causal_memory.create_counterfactual(parent_id, counterfactual_text, projection, verification, metadata)


def export_memory() -> list[CausalStep]:
    return _causal_memory.export()


def import_memory(steps: list[CausalStep]) -> None:
    _causal_memory.import_steps(steps)


def get_memory_stats() -> dict[str, Any]:
    return _causal_memory.stats()


async def persist(key: str = "causal_memory") -> None:
    data = json.dumps([step.__dict__ for step in export_memory()])
    try:
        import localStorage
        localStorage.setItem(key, data)
    except ImportError:
        # Node.js: would use fs.writeFile
        import fs
        fs.writeFileSync(f"{key}.json", data)


async def load(key: str = "causal_memory") -> None:
    try:
        import localStorage
        data = localStorage.getItem(key) or "[]"
    except ImportError:
        import fs
        try:
            data = fs.readFileSync(f"{key}.json", "utf-8")
        except:
            data = "[]"
    import_memory(json.loads(data))