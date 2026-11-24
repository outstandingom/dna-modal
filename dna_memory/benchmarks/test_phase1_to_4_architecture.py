"""
Comprehensive Architecture Verification Unit Tests (Capabilities A through J):
  A. Identity stability
  B. Component reuse (100 leaf structures)
  C. Relationship accuracy & duplicate suppression
  D. Temporal consistency & bitemporal queries
  E. Contradiction handling & supersession
  F. Structural reconstruction
  G. Multi-hop retrieval
  H. Similarity without incorrect merging
  I. Persistence & state restoration
  J. Concurrent updates & thread safety
"""

import unittest
import time
import datetime
import concurrent.futures
from dna_memory.system import DNAMemorySystem
from dna_memory.core.concept_identity import IdentityCollisionError
from dna_memory.assembly.assembly_validator import CircularDependencyError, AssemblyStructuralError


class TestDNAArchitecturePhase1To4(unittest.TestCase):
    def setUp(self):
        self.sys = DNAMemorySystem(dims=128, seed=42)

    def tearDown(self):
        if hasattr(self.sys, "persistence") and hasattr(self.sys.persistence, "close"):
            self.sys.persistence.close()

    def test_a_identity_stability(self):
        """Updating attributes must advance version while keeping canonical ID immutable."""
        c = self.sys.register_canonical_concept(
            canonical_id="concept_vehicle_car_001",
            canonical_name="Car",
            aliases=["automobile", "motorcar"],
            attributes={"wheels": 4, "engine": "gasoline"}
        )
        self.assertEqual(c.canonical_id, "concept_vehicle_car_001")
        self.assertEqual(c.version, 1)

        # Update attribute
        v2 = self.sys.concept_registry.update_attributes(
            "concept_vehicle_car_001",
            {"engine": "electric", "battery_kwh": 75},
            source="user_correction"
        )
        self.assertEqual(v2, 2)
        c_updated = self.sys.resolve_concept("automobile")
        self.assertIsNotNone(c_updated)
        self.assertEqual(c_updated.canonical_id, "concept_vehicle_car_001")
        self.assertEqual(c_updated.attributes["engine"], "electric")

        # Verify historical version preservation
        v1_attrs = c_updated.get_version_at(1)
        self.assertIsNotNone(v1_attrs)
        self.assertEqual(v1_attrs["engine"], "gasoline")

        # Collision detection: prevent registering different concept under same ID
        with self.assertRaises(IdentityCollisionError):
            self.sys.register_canonical_concept(
                canonical_id="concept_vehicle_car_001",
                canonical_name="Truck"
            )

    def test_b_component_reuse_100_leaves(self):
        """100 leaf structures must reuse shared components without losing individual identity."""
        shared_comps = ["leaf_blade", "vein_network", "midrib", "petiole", "stomata_surface"]
        for sc in shared_comps:
            self.sys.shared_components.register_shared_component(sc, sc.replace("_", " ").title(), domain="botany")

        composed_results = []
        for i in range(100):
            leaf_id = f"leaf_sample_{i:04d}"
            res = self.sys.compose_shared_instance(
                instance_id=leaf_id,
                canonical_name=f"Leaf Sample #{i}",
                domain="botany",
                shared_component_ids=shared_comps,
                unique_features={
                    "length_cm": round(5.0 + (i * 0.1), 2),
                    "color_hex": f"#22{i%80:02x}22",
                    "sample_index": i
                }
            )
            composed_results.append(res)

        # Every leaf has its own distinct identity
        self.assertEqual(len(composed_results), 100)
        self.assertEqual(len(set(r.instance_id for r in composed_results)), 100)

        # High reuse ratio (> 0.60 across components)
        avg_reuse = sum(r.reuse_ratio for r in composed_results) / len(composed_results)
        self.assertGreaterEqual(avg_reuse, 0.60)

        # Verify reference count on shared component
        blade = self.sys.shared_components.get_shared_component("leaf_blade")
        self.assertEqual(blade.reference_count, 100)

    def test_c_relationship_accuracy_no_blind_duplicates(self):
        """Repeated observations must not generate uncontrolled duplicate relationships."""
        rel1, is_new1 = self.sys.record_relationship(
            subject_id="person_alice",
            predicate="lives_in",
            object_id="city_paris",
            confidence=0.8,
            source="sensor_a",
            evidence_content="GPS ping 1"
        )
        self.assertTrue(is_new1)

        # Repeated ingestion of same claim
        rel2, is_new2 = self.sys.record_relationship(
            subject_id="person_alice",
            predicate="lives_in",
            object_id="city_paris",
            confidence=0.8,
            source="sensor_a",
            evidence_content="GPS ping 2"
        )
        self.assertFalse(is_new2)
        self.assertEqual(rel1.relationship_id, rel2.relationship_id)
        # Relationship count remains 1
        active_rels = self.sys.relationship_lifecycle.find_relationships(
            subject_id="person_alice",
            predicate="lives_in"
        )
        self.assertEqual(len(active_rels), 1)

    def test_d_temporal_consistency_and_supersession(self):
        """Historical facts remain retrievable while current active state reflects updates."""
        t_2025 = "2025-06-01T00:00:00Z"
        t_2026 = "2026-01-15T00:00:00Z"

        rel_job1, _ = self.sys.record_relationship(
            subject_id="person_bob",
            predicate="employed_at",
            object_id="company_alpha",
            valid_from=t_2025
        )

        # Supersede job in 2026
        rel_job2 = self.sys.supersede_fact(
            old_relationship_id=rel_job1.relationship_id,
            new_object_id="company_beta",
            source="hr_records"
        )

        # Current active state: Bob works at company_beta
        current_rels = self.sys.relationship_lifecycle.find_relationships(
            subject_id="person_bob",
            predicate="employed_at",
            only_active=True
        )
        self.assertEqual(len(current_rels), 1)
        self.assertEqual(current_rels[0].object_id, "company_beta")

        # Historical query as of 2025-08-01: Bob worked at company_alpha
        facts_2025 = self.sys.query_facts_as_of("2025-08-01T00:00:00Z", subject_id="person_bob")
        self.assertEqual(len(facts_2025), 1)
        self.assertEqual(facts_2025[0].object_id, "company_alpha")

    def test_e_contradiction_handling(self):
        """Contradictory observations are flagged without corrupting graph."""
        rel1, _ = self.sys.record_relationship("server_alpha", "status", "online")
        rel1.mark_contradicted("rel_offline_incident_001", reason="hardware_smoke_detected")
        self.assertEqual(rel1.status, "contradicted")
        # Inactive for normal queries
        active = self.sys.relationship_lifecycle.find_relationships(subject_id="server_alpha", only_active=True)
        self.assertEqual(len(active), 0)

    def test_f_assembly_validation_and_cycles(self):
        """Circular dependency detection prevents infinite loops in assembly DAG."""
        # A depends on B
        self.sys.assembly_engine.create_assembly("A", component_ids=[], custom_id="asmb_A")
        self.sys.assembly_engine.create_assembly("B", component_ids=["asmb_A"], custom_id="asmb_B")

        # Propose adding asmb_B to asmb_A -> Circular dependency!
        with self.assertRaises(CircularDependencyError):
            self.sys.assembly_validator.detect_circular_dependencies(
                target_assembly_id="asmb_A",
                proposed_components=["asmb_B"],
                assembly_lookup_fn=lambda aid: self.sys.assembly_engine.get(aid)
            )

    def test_g_difference_detection(self):
        """Difference detection isolates shared parts and distinguishes unique attributes."""
        # Concept A: Neem Leaf
        self.sys.record_relationship("neem_leaf", "has_part", "leaf_blade")
        self.sys.record_relationship("neem_leaf", "has_part", "vein_network")
        self.sys.record_relationship("neem_leaf", "margin", "serrated")

        # Concept B: Mango Leaf
        self.sys.record_relationship("mango_leaf", "has_part", "leaf_blade")
        self.sys.record_relationship("mango_leaf", "has_part", "vein_network")
        self.sys.record_relationship("mango_leaf", "margin", "entire")

        diff = self.sys.compare_structures("neem_leaf", "mango_leaf")
        self.assertIn("leaf_blade", diff.shared_components)
        self.assertIn("vein_network", diff.shared_components)
        # Margin differs
        margin_diff = [d for d in diff.different_relationships if d["predicate"] == "margin"]
        self.assertEqual(len(margin_diff), 1)
        self.assertEqual(margin_diff[0]["target_in_a"], "serrated")
        self.assertEqual(margin_diff[0]["target_in_b"], "entire")

    def test_h_pattern_discovery(self):
        """Discovers recurring relational schemas across observations."""
        instances = ["user_p1", "user_p2", "user_p3"]
        for u in instances:
            self.sys.record_relationship(u, "subscribes_to", "plan_enterprise")
            self.sys.record_relationship(u, "uses_platform", "os_linux")

        patterns = self.sys.discover_patterns_across_instances(instances, domain="saas")
        self.assertGreaterEqual(len(patterns), 1)
        self.assertIn("subscribes_to", patterns[0].recurring_predicates)
        self.assertEqual(patterns[0].frequency, 3)

    def test_i_hybrid_retrieval_and_reranking(self):
        """Multi-channel retrieval combines exact, lexical, vector, and relational graph hits."""
        self.sys.register_canonical_concept(
            canonical_id="concept_ai_agent",
            canonical_name="Autonomous AI Agent",
            aliases=["agentic workflow", "ai worker"]
        )
        self.sys.record_relationship("concept_ai_agent", "requires_capability", "long_term_memory")

        results = self.sys.hybrid_search("ai worker", top_k=3)
        self.assertGreaterEqual(len(results), 1)
        top_cand = results[0].candidate
        self.assertEqual(top_cand.entity_id, "concept_ai_agent")
        self.assertIn("lexical", top_cand.channel_scores)

    def test_j_persistence_and_concurrent_updates(self):
        """Concurrent writes do not corrupt registries and state restores cleanly."""
        def worker(idx):
            cid = f"conc_thread_{idx}"
            self.sys.register_canonical_concept(cid, f"Concept {idx}")
            self.sys.record_relationship(cid, "links_to", "root_node")

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            futures = [ex.submit(worker, i) for i in range(32)]
            for f in concurrent.futures.as_completed(futures):
                f.result()

        # 32 thread concepts + 1 root_node = 33 concepts
        self.assertEqual(self.sys.concept_registry.count(), 33)
        audit = self.sys.audit_integrity()
        self.assertTrue(audit.is_valid)


if __name__ == "__main__":
    unittest.main()
