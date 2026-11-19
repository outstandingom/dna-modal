"""
Automated Test Suite for Atomic DNA, Composition Engine, and Relational DNA.
Verifies:
  1. Atomic DNA creation and vector properties.
  2. Bundling operations (superposition and cosine preservation).
  3. Binding operations (quasi-orthogonality and unbinding recovery).
  4. Authoritative composition manifest and decomposition.
  5. Relational DNA creation, tripartite binding, and query retrieval.
  6. Recursive assemblies and incremental recomposition.
"""

import unittest
import numpy as np
from dna_memory.core.atomic_dna import AtomicDNA, DNAFeatureMetadata
from dna_memory.core.dna_encoder import DNAEncoder
from dna_memory.core.dna_registry import DNARegistry
from dna_memory.core.vector_operations import bundle, bind, unbind, cosine_similarity, normalize
from dna_memory.composition.composition_engine import CompositionEngine
from dna_memory.composition.decomposition_engine import DecompositionEngine
from dna_memory.relations.relation_registry import RelationRegistry
from dna_memory.assembly.recursive_composition import AssemblyEngine

class TestCompositionalDNAMemory(unittest.TestCase):

    def setUp(self):
        self.encoder = DNAEncoder(dims=128, seed=42)
        self.dna_reg = DNARegistry(self.encoder)
        self.comp_engine = CompositionEngine(self.dna_reg)
        self.decomp_engine = DecompositionEngine(self.dna_reg)
        self.rel_reg = RelationRegistry(self.dna_reg)
        self.asmb_engine = AssemblyEngine(self.dna_reg, self.rel_reg)

    def test_01_atomic_dna_creation(self):
        """Verify atomic DNA registers with normalized vector and metadata."""
        atom = self.dna_reg.register("coffee", dna_type="product", confidence=0.99)
        self.assertIsNotNone(atom.id)
        self.assertEqual(atom.value, "coffee")
        self.assertEqual(len(atom.vector), 128)
        self.assertAlmostEqual(float(np.linalg.norm(atom.vector)), 1.0, places=5)
        self.assertEqual(atom.confidence, 0.99)

    def test_02_bundling_superposition(self):
        """Verify bundling retains positive similarity to all constituent components."""
        v1 = self.encoder.encode_text("Bean")
        v2 = self.encoder.encode_text("Aroma")
        v3 = self.encoder.encode_text("Taste")
        v4 = self.encoder.encode_text("Caffeine")

        bundled = bundle([v1, v2, v3, v4])
        self.assertAlmostEqual(float(np.linalg.norm(bundled)), 1.0, places=5)

        # In 128-dim, each of 4 bundled orthogonal vectors retains ~1/sqrt(4) = ~0.5 similarity
        sim1 = cosine_similarity(bundled, v1)
        sim2 = cosine_similarity(bundled, v2)
        sim3 = cosine_similarity(bundled, v3)
        sim4 = cosine_similarity(bundled, v4)

        self.assertGreater(sim1, 0.3)
        self.assertGreater(sim2, 0.3)
        self.assertGreater(sim3, 0.3)
        self.assertGreater(sim4, 0.3)

    def test_03_binding_quasi_orthogonality_and_unbinding(self):
        """Verify HRR circular convolution binding produces quasi-orthogonality and is unbindable."""
        role = self.encoder.encode_text("ROLE:Customer")
        filler = self.encoder.encode_text("VALUE:Rahul")

        bound = bind(role, filler)

        # Quasi-orthogonal to both role and filler:
        sim_to_role = abs(cosine_similarity(bound, role))
        sim_to_filler = abs(cosine_similarity(bound, filler))
        self.assertLess(sim_to_role, 0.3, "Bound representation must be quasi-orthogonal to role")
        self.assertLess(sim_to_filler, 0.3, "Bound representation must be quasi-orthogonal to filler")

        # Unbinding:
        recovered_filler = unbind(role, bound)
        recovery_sim = cosine_similarity(recovered_filler, filler)
        self.assertGreater(recovery_sim, 0.5, "Unbinding must successfully recover the filler vector")

    def test_04_composition_and_lossless_decomposition(self):
        """Verify Coffee DNA composed from Bean, Aroma, Taste, Caffeine can be losslessly decomposed."""
        atom_bean = self.dna_reg.register("Arabica_Bean", "ingredient")
        atom_aroma = self.dna_reg.register("Roasted_Aroma", "sensory")
        atom_taste = self.dna_reg.register("Bitter_Sweet_Taste", "sensory")
        atom_caffeine = self.dna_reg.register("High_Caffeine", "chemical")

        composite_coffee = self.comp_engine.compose(
            name="Coffee_DNA",
            components=[atom_bean, atom_aroma, atom_taste, atom_caffeine],
            operation="bundle"
        )

        self.assertEqual(len(composite_coffee.manifest.component_ids), 4)

        decomposed = self.decomp_engine.decompose_lossless(composite_coffee)
        decomposed_values = [a.value for a in decomposed]

        self.assertIn("Arabica_Bean", decomposed_values)
        self.assertIn("Roasted_Aroma", decomposed_values)
        self.assertIn("Bitter_Sweet_Taste", decomposed_values)
        self.assertIn("High_Caffeine", decomposed_values)

    def test_05_relational_dna_tripartite(self):
        """Verify first-class relational DNA: Rahul -> purchased -> Coffee."""
        atom_customer = self.dna_reg.register("Rahul_001", "customer")
        atom_product = self.dna_reg.register("Coffee_Arabica", "product")

        rel = self.rel_reg.create_relation(
            subject_id=atom_customer.id,
            relation_type="purchased",
            object_id=atom_product.id,
            confidence=1.0,
            strength=1.5
        )

        self.assertIsNotNone(rel.id)
        self.assertEqual(rel.relation_type, "purchased")
        self.assertEqual(len(rel.vector), 128)

        # Query relation
        retrieved = self.rel_reg.retrieve_relations(subject_id=atom_customer.id, relation_type="purchased")
        self.assertEqual(len(retrieved), 1)
        self.assertEqual(retrieved[0].object_id, atom_product.id)

        # Reverse relation
        inv = self.rel_reg.reverse_relation(rel.id)
        self.assertEqual(inv.relation_type, "purchased_by")
        self.assertEqual(inv.subject_id, atom_product.id)
        self.assertEqual(inv.object_id, atom_customer.id)

    def test_06_recursive_assembly(self):
        """Verify hierarchical assemblies: components can be added, removed, and recomposed."""
        c1 = self.dna_reg.register("FeatureA", "property")
        c2 = self.dna_reg.register("FeatureB", "property")

        asmb = self.asmb_engine.create_assembly(
            assembly_type="BaseAssembly",
            component_ids=[c1.id, c2.id]
        )
        self.assertEqual(len(asmb.component_ids), 2)
        initial_vec = asmb.vector.copy()

        # Add 3rd component
        c3 = self.dna_reg.register("FeatureC", "property")
        asmb = self.asmb_engine.add_component(asmb.id, c3.id, weight=2.0)
        self.assertEqual(len(asmb.component_ids), 3)
        self.assertEqual(asmb.version, 2)
        # Vector should have updated
        self.assertFalse(np.allclose(asmb.vector, initial_vec))

        # Check decomposition
        decomp = self.asmb_engine.decompose(asmb.id)
        comp_ids = [c["data"].id for c in decomp["components"]]
        self.assertEqual(comp_ids, [c1.id, c2.id, c3.id])

        # Find parent assemblies
        parents = self.asmb_engine.find_parent_assemblies(c1.id)
        self.assertIn(asmb.id, parents)

if __name__ == "__main__":
    unittest.main()
