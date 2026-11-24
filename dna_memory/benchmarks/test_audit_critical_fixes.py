"""
Unit tests validating resolutions for the 5 Critical Issues raised in the Architecture Audit:
  1. Concept identity collision safety & registry integrity
  2. Relationship lifecycle supersession & contradiction resolution in UniversalUserMemory
  3. Authoritative UserMemoryManifest population, serialization, and reconstruction
  4. Assembly inverted index stale reference cleanup upon recomposition
  5. SineBrainBridge dual-engine interface between System A (HRR) and System B (Sine Brain)
"""

import unittest
import os
import tempfile
from dna_memory.system import DNAMemorySystem
from dna_memory.memory.universal_memory import UserMemoryManifest


class TestAuditCriticalFixes(unittest.TestCase):
    def setUp(self):
        self.sys = DNAMemorySystem(dims=128, seed=42)

    def tearDown(self):
        if hasattr(self.sys, "persistence") and hasattr(self.sys.persistence, "close"):
            self.sys.persistence.close()

    def test_1_collision_safe_fact_registration(self):
        """Two facts with identical initial prefixes must not collide or silently overwrite."""
        # Fact 1: long string starting with 'Senior Cloud Engineer and Architect for Enterprise Systems'
        # Fact 2: different role starting with identical 30-char prefix
        val_1 = "Senior Cloud Engineer and Architect for Enterprise Systems - Alpha Division"
        val_2 = "Senior Cloud Engineer and Architect for Enterprise Systems - Beta Division"

        rel1 = self.sys.user_memory.remember_fact("user_om", "has_role", val_1)
        rel2 = self.sys.user_memory.remember_fact("user_om", "has_role", val_2)

        # The two object IDs must be distinct due to SHA-256 content hashing
        self.assertNotEqual(rel1.object_id, rel2.object_id)

        # Both atoms must exist independently in the DNA registry
        atom1 = self.sys.dna_registry.get(rel1.object_id)
        atom2 = self.sys.dna_registry.get(rel2.object_id)
        self.assertIsNotNone(atom1)
        self.assertIsNotNone(atom2)
        self.assertEqual(atom1.value, val_1)
        self.assertEqual(atom2.value, val_2)

    def test_2_relationship_lifecycle_supersession(self):
        """Changing company must mark old company superseded and keep new company active."""
        user = "user_om"
        self.sys.user_memory.remember_fact(user, "has_company", "Company A")
        profile1 = self.sys.user_memory.reconstruct_profile(user)
        self.assertEqual(profile1.get("has_company"), "Company A")

        # Now update company to Company B
        self.sys.user_memory.remember_fact(user, "has_company", "Company B")
        profile2 = self.sys.user_memory.reconstruct_profile(user)

        # Profile reconstruction must return currently valid Company B, not Company A
        self.assertEqual(profile2.get("has_company"), "Company B")

        # In lifecycle engine, verify Company A is SUPERSEDED and Company B is ACTIVE
        active_rels = self.sys.relationship_lifecycle.query_relationships(
            subject_id=user,
            predicate="has_company",
            status="ACTIVE"
        )
        self.assertEqual(len(active_rels), 1)
        obj_atom = self.sys.dna_registry.get(active_rels[0].object_id)
        self.assertEqual(obj_atom.value, "Company B")

        superseded_rels = self.sys.relationship_lifecycle.query_relationships(
            subject_id=user,
            predicate="has_company",
            status="SUPERSEDED"
        )
        self.assertGreaterEqual(len(superseded_rels), 1)

    def test_3_authoritative_manifest_population_and_persistence(self):
        """UserMemoryManifest must be actively populated, versioned, and serializable."""
        user = "user_alice"
        self.sys.user_memory.remember_fact(user, "has_name", "Alice Chen")
        self.sys.user_memory.remember_fact(user, "has_role", "Principal Architect")

        manifest = self.sys.user_memory.get_manifest(user)
        self.assertIsNotNone(manifest)
        self.assertEqual(manifest.identity_facts["has_name"], "Alice Chen")
        self.assertEqual(manifest.identity_facts["has_role"], "Principal Architect")
        self.assertGreaterEqual(manifest.version, 2)

        # Save manifest to temp JSON file and reload
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            saved = self.sys.user_memory.save_manifest(user, tmp_path)
            self.assertTrue(saved)
            self.assertTrue(os.path.exists(tmp_path))

            # Load into fresh manifest
            loaded = self.sys.user_memory.load_manifest(tmp_path)
            self.assertEqual(loaded.user_id, user)
            self.assertEqual(loaded.identity_facts["has_name"], "Alice Chen")
            self.assertEqual(loaded.identity_facts["has_role"], "Principal Architect")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_4_assembly_inverted_index_cleanup(self):
        """Updating or recompiling an assembly must prune stale entries from _component_to_assemblies."""
        eng = self.sys.assembly_engine
        asmb_id = "test_profile_asmb"

        # Version 1 with components C1, C2, C3
        eng.create_assembly(
            assembly_type="TestProfile",
            component_ids=["comp_1", "comp_2", "comp_3"],
            custom_id=asmb_id
        )
        self.assertIn(asmb_id, eng._component_to_assemblies["comp_1"])
        self.assertIn(asmb_id, eng._component_to_assemblies["comp_2"])
        self.assertIn(asmb_id, eng._component_to_assemblies["comp_3"])

        # Version 2 with components C2 and C4 (C1 and C3 removed)
        eng.create_assembly(
            assembly_type="TestProfile",
            component_ids=["comp_2", "comp_4"],
            custom_id=asmb_id
        )

        # Stale components C1 and C3 must NO LONGER point to asmb_id
        self.assertNotIn("comp_1", eng._component_to_assemblies)
        self.assertNotIn("comp_3", eng._component_to_assemblies)
        # Active components C2 and C4 must point to asmb_id
        self.assertIn(asmb_id, eng._component_to_assemblies["comp_2"])
        self.assertIn(asmb_id, eng._component_to_assemblies["comp_4"])

    def test_5_sine_brain_bridge_sync(self):
        """Bridge must synchronize concepts between System A and System B."""
        # Create a mock System B environment
        class MockBrainEnv:
            def __init__(self):
                self.concepts = {}
                self.relationships = {}
                self.synapses = {("concept_a", "concept_b"): 0.85}

            def predict_heuristic(self, user_id, query):
                return {"predicted_action": "route_to_billing", "confidence": 0.92}

        mock_brain = MockBrainEnv()
        self.sys.brain_bridge.set_brain_env(mock_brain)

        # Register some facts in System A
        self.sys.user_memory.remember_fact("user_bob", "has_company", "Acme Corp")
        report = self.sys.brain_bridge.sync_memory_to_brain(limit=10)
        self.assertGreater(report.concepts_synced_to_brain, 0)

        # Ingest synapses from System B into System A
        ingested = self.sys.brain_bridge.ingest_brain_synapses_to_memory(min_weight=0.7)
        self.assertEqual(ingested, 1)

        # Unified context grounding
        ctx = self.sys.brain_bridge.unified_context_grounding("user_bob", "where do I work?")
        self.assertEqual(ctx["structural_facts"].get("has_company"), "Acme Corp")
        self.assertEqual(ctx["heuristic_routing"].get("predicted_action"), "route_to_billing")


if __name__ == "__main__":
    unittest.main()
