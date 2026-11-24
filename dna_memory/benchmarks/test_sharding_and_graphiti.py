"""
Unit tests validating:
  1. DNA Memory Sharding Engine (consistent hashing, tenant isolation, global reference shard)
  2. Graphiti & Neo4j AuraDB Integration Adapter (episodic ingestion, shadow store, DNA sync)
"""

import unittest
import os
import shutil
import tempfile
from dna_memory.system import DNAMemorySystem
from dna_memory.core.sharding import ShardManager
from dna_memory.integrations.graphiti_adapter import GraphitiAdapter


class TestShardingAndGraphiti(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="dna_sharding_test_")
        self.sharder = ShardManager(num_shards=4, base_dir=self.tmp_dir, dims=128, seed=42)
        self.graphiti = GraphitiAdapter()  # Uses shadow mode for zero-dependency test

    def tearDown(self):
        if hasattr(self.sharder, "global_shard") and hasattr(self.sharder.global_shard, "persistence"):
            self.sharder.global_shard.persistence.close()
        for shard_sys in list(self.sharder._active_shards.values()):
            if hasattr(shard_sys, "persistence"):
                shard_sys.persistence.close()
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_1_consistent_hashing_and_isolation(self):
        """Tenants must deterministically map to consistent shards."""
        user_a = "user_alpha_101"
        user_b = "user_beta_202"

        shard_a = self.sharder.get_shard_id(user_a)
        shard_b = self.sharder.get_shard_id(user_b)

        # Consistent mapping: repeated call must return identical shard ID
        self.assertEqual(self.sharder.get_shard_id(user_a), shard_a)
        self.assertEqual(self.sharder.get_shard_id(user_b), shard_b)

        # Record facts in tenant shards
        self.sharder.record_tenant_fact(user_a, "favorite_color", "Indigo")
        self.sharder.record_tenant_fact(user_b, "favorite_color", "Emerald")

        # Shard systems must hold the respective fact
        sys_a = self.sharder.get_shard(user_a)
        sys_b = self.sharder.get_shard(user_b)

        profile_a = sys_a.user_memory.reconstruct_profile(user_a)
        self.assertEqual(profile_a.get("favorite_color"), "Indigo")

        profile_b = sys_b.user_memory.reconstruct_profile(user_b)
        self.assertEqual(profile_b.get("favorite_color"), "Emerald")

    def test_2_sharding_observability_stats(self):
        """Stats report must reflect all partitions and global shard."""
        self.sharder.record_tenant_fact("tenant_x", "works_at", "Growhaz Tech")
        stats = self.sharder.get_sharding_stats()

        self.assertEqual(stats["num_shards"], 4)
        self.assertGreaterEqual(stats["total_tenants"], 1)
        self.assertIn("global_shard", stats)
        self.assertEqual(len(stats["shards"]), 4)

    def test_3_graphiti_episodic_extraction(self):
        """Graphiti must extract entities (org, role, tech) and temporal relations from text."""
        message = "I joined Growhaz as a Principal AI Architect working with Python and Neo4j."
        res = self.graphiti.add_episode(user_id="om_uikey", message=message)

        self.assertEqual(res["status"], "RECORDED")
        entities = res["extracted_entities"]
        relations = res["extracted_relations"]

        entity_names = [e["name"] for e in entities]
        self.assertIn("om_uikey", entity_names)
        self.assertTrue(any("Growhaz" in n for n in entity_names))
        self.assertTrue(any("Principal AI Architect" in n for n in entity_names))

        predicates = [r["predicate"] for r in relations]
        self.assertIn("works_at", predicates)
        self.assertIn("has_role", predicates)

    def test_4_graphiti_search_and_temporal_lookup(self):
        """Search must find recorded nodes and edges."""
        self.graphiti.add_episode("alice", "I work at DeepMind as a Senior Researcher with Python.")
        results = self.graphiti.search("DeepMind", user_id="alice")

        self.assertGreaterEqual(results["total_matches"], 1)
        self.assertTrue(any("DeepMind" in e["name"] for e in results["entities"]))

    def test_5_graphiti_sync_to_dna_modal(self):
        """Extracted Graphiti entities and relations must cleanly train/populate DNA Modal."""
        dna_sys = DNAMemorySystem(dims=128, seed=42)

        # Seed Graphiti with an episode
        self.graphiti.add_episode("bob_builder", "I work at CyberForge as Chief Security Officer using Rust.")

        # Sync to DNA Modal
        report = self.graphiti.sync_to_dna_memory(dna_sys)

        self.assertEqual(report.status, "SUCCESS")
        self.assertGreater(report.entities_synced, 0)
        self.assertGreater(report.relations_synced, 0)

        # Check DNA Modal concept registry
        all_concepts = list(dna_sys.concept_registry._concepts.keys())
        self.assertTrue(any("CyberForge" in cid or "cyberforge" in cid for cid in all_concepts))


if __name__ == "__main__":
    unittest.main()
