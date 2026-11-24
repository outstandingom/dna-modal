"""
DNA Memory Sharding Engine: Multi-Tenant Partitioning & Shard Management.

Provides horizontal scaling, tenant isolation, consistent hashing, and
cross-shard querying across independent DNAMemorySystem instances.
"""

import os
import hashlib
import datetime
import threading
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field

from dna_memory.system import DNAMemorySystem


@dataclass
class ShardMetrics:
    """Metrics and state summary for an individual shard."""
    shard_id: str
    db_path: str
    tenant_count: int
    concept_count: int
    relation_count: int
    assembly_count: int
    last_accessed: str
    is_global: bool = False


class ShardManager:
    """
    Manages horizontal partitioning of the DNA Memory architecture.
    Uses consistent hashing (SHA-256 ring) to route tenants/users to isolated shards,
    while maintaining a global reference shard for universal domain ontology.
    """

    def __init__(
        self,
        num_shards: int = 4,
        base_dir: str = "./brain_data/shards",
        dims: int = 128,
        seed: int = 42,
        max_active_shards: int = 16
    ):
        self.num_shards = max(1, num_shards)
        self.base_dir = os.path.abspath(base_dir)
        self.dims = dims
        self.seed = seed
        self.max_active_shards = max_active_shards
        self._lock = threading.RLock()

        os.makedirs(self.base_dir, exist_ok=True)

        # Active shard cache: shard_id -> DNAMemorySystem
        self._active_shards: Dict[str, DNAMemorySystem] = {}
        # Tenant to shard lookup cache: tenant_id -> shard_id
        self._tenant_shard_map: Dict[str, str] = {}
        # Shard tenant sets: shard_id -> Set[tenant_id]
        self._shard_tenants: Dict[str, Set[str]] = {
            f"shard_{i}": set() for i in range(self.num_shards)
        }

        # Initialize Global / Reference Shard for shared domain ontology
        self.global_db_path = os.path.join(self.base_dir, "global_reference.db")
        self.global_shard = DNAMemorySystem(
            dims=dims,
            seed=seed,
            db_path=self.global_db_path
        )

    def get_shard_id(self, tenant_id: str) -> str:
        """
        Computes the target shard ID for a tenant/user using consistent SHA-256 hashing.
        """
        clean_tenant = str(tenant_id).strip()
        if not clean_tenant:
            return "shard_0"

        # Check cached mapping
        if clean_tenant in self._tenant_shard_map:
            return self._tenant_shard_map[clean_tenant]

        hash_int = int(hashlib.sha256(clean_tenant.encode("utf-8")).hexdigest()[:16], 16)
        shard_idx = hash_int % self.num_shards
        shard_id = f"shard_{shard_idx}"

        with self._lock:
            self._tenant_shard_map[clean_tenant] = shard_id
            self._shard_tenants[shard_id].add(clean_tenant)

        return shard_id

    def get_shard(self, tenant_id: str) -> DNAMemorySystem:
        """
        Retrieves or initializes the isolated DNAMemorySystem shard for the given tenant.
        """
        shard_id = self.get_shard_id(tenant_id)

        with self._lock:
            if shard_id not in self._active_shards:
                shard_db = os.path.join(self.base_dir, f"{shard_id}.db")
                sys_instance = DNAMemorySystem(
                    dims=self.dims,
                    seed=self.seed + int(shard_id.split("_")[-1]),
                    db_path=shard_db
                )
                self._active_shards[shard_id] = sys_instance

            return self._active_shards[shard_id]

    def record_tenant_fact(
        self,
        tenant_id: str,
        relation_type: str,
        fact_value: str,
        confidence: float = 1.0,
        source: str = "dialogue"
    ) -> Any:
        """
        Routes and persists a user fact directly to the tenant's isolated shard.
        """
        shard = self.get_shard(tenant_id)
        return shard.user_memory.remember_fact(
            user_id=tenant_id,
            relation_type=relation_type,
            fact_value=fact_value,
            confidence=confidence,
            source=source
        )

    def query_tenant(self, tenant_id: str, query_text: str) -> Dict[str, Any]:
        """
        Federated query: Resolves query against the tenant's private shard and
        supplements with universal facts from the global shard.
        """
        shard = self.get_shard(tenant_id)

        # 1. Resolve tenant personal facts
        tenant_context = shard.user_memory.synthesize_agent_grounding(
            user_id=tenant_id,
            query_text=query_text
        )

        # 2. Search tenant hybrid index
        tenant_candidates = shard.search_hybrid(query_text, limit=3)

        # 3. Search global reference index
        global_candidates = self.global_shard.search_hybrid(query_text, limit=3)

        return {
            "tenant_id": tenant_id,
            "shard_id": self.get_shard_id(tenant_id),
            "user_profile": tenant_context.user_profile,
            "direct_answers": tenant_context.direct_answers,
            "prompt_context": tenant_context.prompt_context_string,
            "tenant_hits": [c.to_dict() for c in tenant_candidates],
            "global_reference_hits": [c.to_dict() for c in global_candidates]
        }

    def get_sharding_stats(self) -> Dict[str, Any]:
        """
        Returns full observability metrics for all shards and tenant distribution.
        """
        with self._lock:
            stats = []
            total_tenants = len(self._tenant_shard_map)
            total_concepts = 0
            total_relations = 0

            # Global shard metrics
            global_metrics = ShardMetrics(
                shard_id="global_reference",
                db_path=self.global_db_path,
                tenant_count=0,
                concept_count=self.global_shard.dna_registry.count(),
                relation_count=self.global_shard.relation_registry.count(),
                assembly_count=self.global_shard.assembly_engine.count(),
                last_accessed=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                is_global=True
            )
            total_concepts += global_metrics.concept_count
            total_relations += global_metrics.relation_count

            # Individual partition shards
            for i in range(self.num_shards):
                sid = f"shard_{i}"
                tenants = self._shard_tenants.get(sid, set())
                c_cnt = 0
                r_cnt = 0
                a_cnt = 0
                db_path = os.path.join(self.base_dir, f"{sid}.db")

                if sid in self._active_shards:
                    shard_sys = self._active_shards[sid]
                    c_cnt = shard_sys.dna_registry.count()
                    r_cnt = shard_sys.relation_registry.count()
                    a_cnt = shard_sys.assembly_engine.count()

                total_concepts += c_cnt
                total_relations += r_cnt

                stats.append({
                    "shard_id": sid,
                    "db_path": db_path,
                    "tenant_count": len(tenants),
                    "concept_count": c_cnt,
                    "relation_count": r_cnt,
                    "assembly_count": a_cnt,
                    "is_active": sid in self._active_shards
                })

            return {
                "num_shards": self.num_shards,
                "total_tenants": total_tenants,
                "total_concepts": total_concepts,
                "total_relations": total_relations,
                "global_shard": {
                    "shard_id": "global_reference",
                    "concept_count": global_metrics.concept_count,
                    "relation_count": global_metrics.relation_count
                },
                "shards": stats
            }
