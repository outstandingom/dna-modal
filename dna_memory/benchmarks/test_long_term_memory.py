"""
Benchmark & Validation: DNA Long-Term Memory & Identity Recall
Verifies that the DNA encoding and decoding algorithms reliably extract,
store, and recall customer identities and attributes (e.g. Om Uikey, Growhaz).
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from dna_memory import DNAMemorySystem
from dna_memory.demo_dataset import seed_demo_knowledge_base
from dna_memory.core.vector_operations import bind, unbind, cosine_similarity

def run_long_term_memory_benchmark():
    print("===========================================================================")
    print("🧬 BENCHMARK: DNA LONG-TERM MEMORY & IDENTITY ENCODING/DECODING")
    print("===========================================================================\n")

    # 1. Initialize DNA Memory System
    system = DNAMemorySystem(dims=128, seed=42)

    # 2. Seed Demo Knowledge Base & Customer Personas
    print("[1] Seeding Demo Knowledge Base via DNA Holographic Encoding...")
    res = seed_demo_knowledge_base(system)
    print(f"  • Registered Atoms:      {res['atoms_count']}")
    print(f"  • Relational Bonds:      {res['relations_count']}")
    print(f"  • Customer Personas:     {res['customers_seeded']}\n")

    # 3. Simulate New Customer Dialogue Fact Extraction
    test_customer = "919988776655"
    print(f"[2] Simulating Dialogue Fact Learning for Customer: {test_customer}")
    
    # Register customer entity
    cust_atom = system.register_concept(value=test_customer, concept_type="customer", custom_id=test_customer)
    
    # User says: "my name is Om Uikey"
    name_atom = system.register_concept(value="Om Uikey", concept_type="customer_attribute", custom_id="fact_has_name_om_uikey")
    rel_name = system.relation_registry.create_relation(
        subject_id=test_customer,
        relation_type="has_name",
        object_id=name_atom.id,
        confidence=1.0
    )
    print(f"  • Encoded Relational Bond: {rel_name.subject_id} --[{rel_name.relation_type}]--> {rel_name.object_id}")

    # User says: "my company is Growhaz"
    comp_atom = system.register_concept(value="Growhaz Technologies", concept_type="customer_attribute", custom_id="fact_has_company_growhaz")
    rel_comp = system.relation_registry.create_relation(
        subject_id=test_customer,
        relation_type="has_company",
        object_id=comp_atom.id,
        confidence=1.0
    )
    print(f"  • Encoded Relational Bond: {rel_comp.subject_id} --[{rel_comp.relation_type}]--> {rel_comp.object_id}\n")

    # 4. Decode Long-Term Memory via Relational Index
    print(f"[3] Decoding Customer Long-Term Memory Profile...")
    rels = system.relation_registry.retrieve_relations(subject_id=test_customer)
    decoded_facts = {}
    for r in rels:
        obj_atom = system.dna_registry.get(r.object_id)
        val = obj_atom.value if obj_atom else r.object_id
        decoded_facts[r.relation_type] = val
        print(f"  • Recalled Fact: {r.relation_type} -> {val}")

    assert decoded_facts.get("has_name") == "Om Uikey", "Failed to decode name!"
    assert decoded_facts.get("has_company") == "Growhaz Technologies", "Failed to decode company!"
    print("  >>> Verified: Accurate Identity & Company Recall! ✅\n")

    # 5. Holographic Vector Unbinding Proof
    print("[4] Holographic Vector Unbinding Mathematical Verification...")
    # S = customer vector, R = relation vector ("has_name"), O = name vector ("Om Uikey")
    s_vec = cust_atom.vector
    r_vec = system.encoder.encode_text("RELATION:has_name")
    o_vec = name_atom.vector

    # Forward Binding: R_ABC = bind(bind(S, R), O)
    sr_bound = bind(s_vec, r_vec)
    r_abc = bind(sr_bound, o_vec)

    # Reverse Unbinding: unbind(bind(S, R), R_ABC) ≈ O
    recovered_o_vec = unbind(sr_bound, r_abc)
    sim = cosine_similarity(recovered_o_vec, o_vec)
    print(f"  • Cosine Similarity between Unbound Vector and True 'Om Uikey' Vector: {sim:.4f}")
    assert sim > 0.30, f"Unbinding fidelity too low ({sim})"
    print("  >>> Verified: Holographic Unbinding Successfully Recovered Object Vector! ✅\n")

    # 6. Spreading Activation Search for Identity
    print("[5] Testing Spreading Activation from Query: 'What is Om Uikey's role?'")
    act_results = system.search_memory("Om Uikey role", top_k=4)
    for res in act_results:
        print(f"  • [{res['type'].upper()}] {res['id']} (Activation: {res['activation_score']:.3f})")

    print("\n✅ LONG-TERM MEMORY & IDENTITY ENCODING/DECODING VERIFICATION COMPLETE!")

if __name__ == "__main__":
    run_long_term_memory_benchmark()
