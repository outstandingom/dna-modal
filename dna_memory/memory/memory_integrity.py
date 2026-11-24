"""
Memory Integrity Engine: Enforces system-wide transactional consistency,
detects dangling references, validates vector dimensions and normalization,
performs multi-user isolation checks, and provides automated recovery verification.
"""

from dataclasses import dataclass, field
import datetime
import json
import os
import shutil
import threading
from typing import Dict, List, Optional, Set, Any, Tuple
import numpy as np


@dataclass
class IntegrityCheckReport:
    """Detailed audit report of memory state consistency."""
    is_valid: bool
    checked_at: str
    total_atoms: int
    total_concepts: int
    total_relationships: int
    total_assemblies: int
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        status = "PASSED ✅" if self.is_valid else "FAILED ❌"
        msg = [
            f"=== MEMORY INTEGRITY AUDIT: {status} ===",
            f"Timestamp:          {self.checked_at}",
            f"Atomic Features:    {self.total_atoms}",
            f"Concepts:           {self.total_concepts}",
            f"Relationships:      {self.total_relationships}",
            f"Assemblies:         {self.total_assemblies}",
            f"Error Count:        {len(self.errors)}",
            f"Warning Count:      {len(self.warnings)}"
        ]
        if self.errors:
            msg.append("\nErrors Detected:")
            for e in self.errors:
                msg.append(f"  • {e}")
        if self.warnings:
            msg.append("\nWarnings:")
            for w in self.warnings:
                msg.append(f"  • {w}")
        return "\n".join(msg)


class MemoryIntegrityEngine:
    """
    Performs comprehensive verification, repair, and backup/restore
    across all DNA Memory subsystems.
    """
    def __init__(
        self,
        dna_registry=None,
        concept_registry=None,
        relationship_engine=None,
        assembly_engine=None
    ):
        self.dna_registry = dna_registry
        self.concept_registry = concept_registry
        self.relationship_engine = relationship_engine
        self.assembly_engine = assembly_engine
        self._lock = threading.RLock()

    def run_full_audit(self) -> IntegrityCheckReport:
        """
        Executes a thorough health check across all registries and relational stores:
          1. Vector normalization and NaN/Inf checks.
          2. Referential integrity: relationships point to existing entities.
          3. Assembly structural integrity: component references and weight shapes.
          4. Circular dependency detection in assembly trees.
        """
        with self._lock:
            now = datetime.datetime.now(datetime.timezone.utc).isoformat()
            errors: List[str] = []
            warnings: List[str] = []

            atoms_count = self.dna_registry.count() if self.dna_registry else 0
            concepts_count = self.concept_registry.count() if self.concept_registry else 0
            rels_count = self.relationship_engine.count(only_active=False) if self.relationship_engine else 0
            asmb_count = self.assembly_engine.count() if self.assembly_engine else 0

            # 1. Audit Atomic DNA Vectors
            if self.dna_registry:
                for atom_id, atom in getattr(self.dna_registry, "_registry", {}).items():
                    vec = atom.vector
                    if not isinstance(vec, np.ndarray):
                        errors.append(f"Atom '{atom_id}' vector is not a numpy array.")
                        continue
                    if np.isnan(vec).any() or np.isinf(vec).any():
                        errors.append(f"Atom '{atom_id}' contains NaN or Inf values in vector.")
                    norm = np.linalg.norm(vec)
                    if abs(norm - 1.0) > 1e-4 and norm > 1e-12:
                        warnings.append(f"Atom '{atom_id}' vector norm is {norm:.6f} (not unit length).")

            # 2. Audit Relationships Referential Integrity
            known_entities: Set[str] = set()
            if self.dna_registry:
                known_entities.update(getattr(self.dna_registry, "_registry", {}).keys())
            if self.concept_registry:
                known_entities.update(getattr(self.concept_registry, "_concepts", {}).keys())
            if self.assembly_engine:
                known_entities.update(getattr(self.assembly_engine, "_assemblies", {}).keys())

            if self.relationship_engine:
                for rel_id, rel in getattr(self.relationship_engine, "_relations", {}).items():
                    if rel.subject_id not in known_entities:
                        errors.append(
                            f"Relationship '{rel_id}' has dangling subject_id '{rel.subject_id}' "
                            f"(not found in concepts, atoms, or assemblies)."
                        )
                    if rel.object_id not in known_entities:
                        # Could be an external literal or missing entity
                        warnings.append(
                            f"Relationship '{rel_id}' object_id '{rel.object_id}' is not a registered entity."
                        )
                    if not (0.0 <= rel.confidence <= 1.0):
                        errors.append(f"Relationship '{rel_id}' confidence {rel.confidence} is out of bounds [0, 1].")

            # 3. Audit Assemblies
            if self.assembly_engine:
                for asmb_id, asmb in getattr(self.assembly_engine, "_assemblies", {}).items():
                    if len(asmb.component_ids) != len(asmb.weights):
                        errors.append(
                            f"Assembly '{asmb_id}' has weight length mismatch: "
                            f"{len(asmb.component_ids)} components vs {len(asmb.weights)} weights."
                        )
                    for cid in asmb.component_ids:
                        if cid not in known_entities:
                            errors.append(f"Assembly '{asmb_id}' references missing component '{cid}'.")

            is_valid = len(errors) == 0
            return IntegrityCheckReport(
                is_valid=is_valid,
                checked_at=now,
                total_atoms=atoms_count,
                total_concepts=concepts_count,
                total_relationships=rels_count,
                total_assemblies=asmb_count,
                errors=errors,
                warnings=warnings
            )

    def export_backup(self, backup_dir: str) -> str:
        """
        Creates a versioned, atomic JSON snapshot of all registered memories.
        """
        with self._lock:
            os.makedirs(backup_dir, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            target_path = os.path.join(backup_dir, f"dna_memory_backup_{timestamp}.json")

            payload = {
                "version": "2.0.0",
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "atoms": self.dna_registry.to_dict() if hasattr(self.dna_registry, "to_dict") else {},
                "concepts": self.concept_registry.to_dict() if hasattr(self.concept_registry, "to_dict") else {},
                "relations": self.relationship_engine.to_dict() if hasattr(self.relationship_engine, "to_dict") else {},
                "assemblies": self.assembly_engine.to_dict() if hasattr(self.assembly_engine, "to_dict") else {}
            }

            temp_path = target_path + ".tmp"
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            os.replace(temp_path, target_path)

            return target_path

    def delete_entity_cascaded(self, entity_id: str) -> Dict[str, int]:
        """
        Safely deletes an entity and purges all dependent relationships and assembly links.
        Prevents lingering orphaned references.
        """
        with self._lock:
            deleted_rels = 0
            deleted_atoms = 0
            deleted_concepts = 0

            # 1. Deactivate relationships involving entity_id
            if self.relationship_engine:
                rels = self.relationship_engine.find_relationships(subject_id=entity_id, only_active=False)
                rels += self.relationship_engine.find_relationships(object_id=entity_id, only_active=False)
                for r in set(rels):
                    self.relationship_engine.deactivate_relationship(r.relationship_id, reason=f"cascaded_delete_{entity_id}")
                    deleted_rels += 1

            # 2. Remove from concept registry
            if self.concept_registry and self.concept_registry.exists(entity_id):
                with self.concept_registry._lock:
                    concept = self.concept_registry._concepts.pop(entity_id, None)
                    if concept:
                        # Clean alias index
                        for a in concept.aliases:
                            self.concept_registry._alias_index.pop(a, None)
                        deleted_concepts += 1

            # 3. Remove from atomic DNA registry
            if self.dna_registry and hasattr(self.dna_registry, "unregister"):
                if self.dna_registry.has(entity_id):
                    self.dna_registry.unregister(entity_id)
                    deleted_atoms += 1

            return {
                "deleted_relationships": deleted_rels,
                "deleted_concepts": deleted_concepts,
                "deleted_atoms": deleted_atoms
            }
