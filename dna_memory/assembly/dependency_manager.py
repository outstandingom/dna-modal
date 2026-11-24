"""
Assembly Dependency Manager: Maintains bi-directional dependency graphs,
propagates updates up the composition hierarchy, invalidates stale cached vectors,
and supports transactional rollback on composition failures.
"""

from typing import Dict, List, Set, Optional, Tuple, Any
import copy
import threading


class DependencyManager:
    """
    Tracks and manages dependencies between Atomic components and DNAAssemblies:
      - Forward mapping: assembly_id -> Set of child component_ids
      - Inverted mapping: component_id -> Set of parent assembly_ids
    Enables automatic upward recomposition and transactional rollback.
    """
    def __init__(self):
        self._child_to_parents: Dict[str, Set[str]] = {}  # component_id -> parent assembly_ids
        self._parent_to_children: Dict[str, Set[str]] = {}  # assembly_id -> child component_ids
        self._lock = threading.RLock()

    def register_assembly(self, assembly_id: str, component_ids: List[str]):
        """Registers parent-child links for an assembly."""
        with self._lock:
            # Clear previous children if updating an existing assembly
            self.unregister_assembly(assembly_id)

            self._parent_to_children[assembly_id] = set(component_ids)
            for cid in component_ids:
                self._child_to_parents.setdefault(cid, set()).add(assembly_id)

    def unregister_assembly(self, assembly_id: str):
        """Cleans up all references to an assembly from inverted indexes."""
        with self._lock:
            if assembly_id in self._parent_to_children:
                old_children = self._parent_to_children[assembly_id]
                for cid in old_children:
                    if cid in self._child_to_parents:
                        self._child_to_parents[cid].discard(assembly_id)
                        if not self._child_to_parents[cid]:
                            del self._child_to_parents[cid]
                del self._parent_to_children[assembly_id]

    def get_parent_assemblies(self, component_id: str) -> Set[str]:
        """Returns direct parent assemblies that depend on this component."""
        with self._lock:
            return set(self._child_to_parents.get(component_id, set()))

    def get_all_ancestor_assemblies(self, component_id: str) -> List[str]:
        """
        Returns all recursive ancestor assemblies in topological update order
        (parents before grandparents) to allow safe upward recomposition.
        """
        with self._lock:
            ordered_ancestors: List[str] = []
            visited: Set[str] = set()
            queue = list(self.get_parent_assemblies(component_id))

            while queue:
                current_parent = queue.pop(0)
                if current_parent not in visited:
                    visited.add(current_parent)
                    ordered_ancestors.append(current_parent)
                    # Enqueue grandparents
                    grandparents = self.get_parent_assemblies(current_parent)
                    queue.extend(grandparents - visited)

            return ordered_ancestors

    def get_child_components(self, assembly_id: str) -> Set[str]:
        """Returns immediate constituent children of an assembly."""
        with self._lock:
            return set(self._parent_to_children.get(assembly_id, set()))

    def create_snapshot(self) -> Dict[str, Any]:
        """Creates a deep copy snapshot for rollback purposes."""
        with self._lock:
            return {
                "child_to_parents": {k: set(v) for k, v in self._child_to_parents.items()},
                "parent_to_children": {k: set(v) for k, v in self._parent_to_children.items()}
            }

    def restore_snapshot(self, snapshot: Dict[str, Any]):
        """Restores dependencies from a previous snapshot."""
        with self._lock:
            self._child_to_parents = {k: set(v) for k, v in snapshot.get("child_to_parents", {}).items()}
            self._parent_to_children = {k: set(v) for k, v in snapshot.get("parent_to_children", {}).items()}
