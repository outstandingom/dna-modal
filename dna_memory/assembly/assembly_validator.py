"""
Assembly Validator Engine: Validates structural integrity, component existence,
relationship references, weight dimension consistency, and circular dependency detection
in recursive DNA hierarchical assemblies.
"""

from typing import List, Dict, Set, Optional, Any
import numpy as np


class AssemblyStructuralError(ValueError):
    """Raised when an assembly fails structural validation."""
    pass


class CircularDependencyError(AssemblyStructuralError):
    """Raised when a circular composition loop is detected in the assembly DAG."""
    pass


class AssemblyValidator:
    """
    Guarantees that assemblies remain mathematically consistent, reference-valid,
    and strictly acyclic (Directed Acyclic Graph).
    """

    @staticmethod
    def validate_weights(component_ids: List[str], weights: Optional[List[float]]) -> List[float]:
        """
        Validates that weights length strictly matches component_ids length,
        and all weights are finite non-negative numbers.
        """
        if weights is None:
            return [1.0] * len(component_ids)

        if len(weights) != len(component_ids):
            raise AssemblyStructuralError(
                f"Weight dimension mismatch: Expected {len(component_ids)} weights, "
                f"received {len(weights)}."
            )

        validated = []
        for i, w in enumerate(weights):
            if not isinstance(w, (int, float)) or np.isnan(w) or np.isinf(w):
                raise AssemblyStructuralError(f"Invalid weight at index {i}: '{w}' is not a finite number.")
            if w < 0:
                raise AssemblyStructuralError(f"Negative weight at index {i}: '{w}'. Weights must be >= 0.")
            validated.append(float(w))

        return validated

    @staticmethod
    def validate_component_references(
        component_ids: List[str],
        dna_registry=None,
        assembly_engine=None
    ) -> bool:
        """
        Verifies that every constituent component exists either as an AtomicDNA in the registry
        or as a child DNAAssembly in the assembly engine.
        """
        missing = []
        for cid in component_ids:
            exists_in_atoms = bool(dna_registry and dna_registry.has(cid))
            exists_in_assemblies = bool(assembly_engine and assembly_engine.has_assembly(cid))
            if not exists_in_atoms and not exists_in_assemblies:
                missing.append(cid)

        if missing:
            raise AssemblyStructuralError(
                f"Missing component reference(s): {missing}. Components must be registered prior to assembly."
            )
        return True

    @staticmethod
    def validate_relation_references(relation_ids: List[str], relation_registry=None) -> bool:
        """
        Verifies that every referenced relation exists in the relation registry.
        """
        if not relation_registry or not relation_ids:
            return True

        missing = []
        for rid in relation_ids:
            # Check either find_by_id or has method
            has_rel = hasattr(relation_registry, "has") and relation_registry.has(rid)
            if not has_rel and hasattr(relation_registry, "find_by_id") and relation_registry.find_by_id(rid):
                has_rel = True
            if not has_rel and hasattr(relation_registry, "_relations") and rid in relation_registry._relations:
                has_rel = True

            if not has_rel:
                missing.append(rid)

        if missing:
            raise AssemblyStructuralError(f"Missing relation reference(s): {missing}.")
        return True

    @staticmethod
    def detect_circular_dependencies(
        target_assembly_id: str,
        proposed_components: List[str],
        assembly_lookup_fn
    ) -> None:
        """
        Traverses child components recursively using Depth-First Search (DFS)
        to detect if target_assembly_id appears in any sub-assembly path.
        """
        visited: Set[str] = set()
        path: List[str] = [target_assembly_id]

        def dfs(current_id: str):
            if current_id == target_assembly_id and len(path) > 1:
                loop_str = " -> ".join(path)
                raise CircularDependencyError(
                    f"Circular assembly composition detected! Path: {loop_str}"
                )

            if current_id in visited:
                return

            visited.add(current_id)

            # Retrieve assembly if it is a composite node
            child_assembly = assembly_lookup_fn(current_id)
            if child_assembly is not None:
                children = getattr(child_assembly, "component_ids", [])
                for child in children:
                    path.append(child)
                    dfs(child)
                    path.pop()

        for comp_id in proposed_components:
            path.append(comp_id)
            dfs(comp_id)
            path.pop()
