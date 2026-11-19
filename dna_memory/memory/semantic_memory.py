"""
Semantic Memory: Long-term consolidated store of concepts, assemblies, and relational knowledge.
"""

from dna_memory.core.dna_registry import DNARegistry
from dna_memory.relations.relation_registry import RelationRegistry
from dna_memory.assembly.recursive_composition import AssemblyEngine

class SemanticMemory:
    """
    Unified interface to the system's long-term consolidated memory representations.
    """
    def __init__(
        self,
        dna_registry: DNARegistry,
        relation_registry: RelationRegistry,
        assembly_engine: AssemblyEngine
    ):
        self.dna_registry = dna_registry
        self.relation_registry = relation_registry
        self.assembly_engine = assembly_engine

    def get_concept_summary(self, concept_id: str) -> dict:
        """Returns unified summary of an entity or concept."""
        atom = self.dna_registry.get(concept_id)
        asmb = self.assembly_engine.get(concept_id)
        rels_out = self.relation_registry.retrieve_relations(subject_id=concept_id)
        rels_in = self.relation_registry.retrieve_relations(object_id=concept_id)

        return {
            "id": concept_id,
            "atomic": atom.to_dict() if atom else None,
            "assembly": asmb.to_dict() if asmb else None,
            "outbound_relations": [r.to_dict() for r in rels_out],
            "inbound_relations": [r.to_dict() for r in rels_in]
        }
