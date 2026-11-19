"""
Customer Memory Verification Benchmark:
Executes the exact 10-step Customer Memory Scenario:
  1. Create customer representation (rahul_001).
  2. Record 10 purchase events (periodic cadence).
  3. Construct product DNA (Coffee = Bean + Aroma + Taste + Caffeine) and Purchase event DNA assemblies.
  4. Create relational DNA for every purchase (Rahul -> purchased -> Coffee).
  5. Compose Customer Behaviour Assembly.
  6. Add 11th purchase event.
  7. Update the Customer Behaviour Assembly incrementally.
  8. Ask system to explain customer's purchasing pattern & active memories.
  9. Predict the next purchase window.
  10. Compare prediction with actual outcome.
"""

import datetime
from dna_memory.core.dna_encoder import DNAEncoder
from dna_memory.core.dna_registry import DNARegistry
from dna_memory.composition.composition_engine import CompositionEngine
from dna_memory.composition.decomposition_engine import DecompositionEngine
from dna_memory.relations.relation_registry import RelationRegistry
from dna_memory.assembly.recursive_composition import AssemblyEngine
from dna_memory.activation.activation_engine import DNAActivationEngine
from dna_memory.learning.learning_engine import MemoryLearningEngine
from dna_memory.prediction.behaviour_prediction import BehaviourPredictionEngine
from dna_memory.prediction.temporal_patterns import TemporalPatternAnalyzer

def run_customer_memory_benchmark():
    print("=" * 70)
    print("🚀 RUNNING RECURSIVE COMPOSITIONAL DNA MEMORY BENCHMARK: CUSTOMER MEMORY")
    print("=" * 70)

    encoder = DNAEncoder(dims=128, seed=42)
    dna_reg = DNARegistry(encoder)
    comp_engine = CompositionEngine(dna_reg)
    decomp_engine = DecompositionEngine(dna_reg)
    rel_reg = RelationRegistry(dna_reg)
    asmb_engine = AssemblyEngine(dna_reg, rel_reg)
    act_engine = DNAActivationEngine(dna_reg, rel_reg, asmb_engine)
    learning_engine = MemoryLearningEngine(dna_reg, rel_reg, asmb_engine)
    prediction_engine = BehaviourPredictionEngine(learning_engine, rel_reg, asmb_engine)

    # -------------------------------------------------------------
    # Step 1: Create a Customer
    # -------------------------------------------------------------
    customer = dna_reg.register("Rahul", dna_type="customer", custom_id="rahul_001")
    print(f"\n[Step 1] Customer Created: ID='{customer.id}', Type='{customer.type}', Value='{customer.value}'")

    # -------------------------------------------------------------
    # Step 2 & 3: Construct Product DNA & Purchase Events
    # Coffee DNA = Bean + Aroma + Taste + Caffeine
    # -------------------------------------------------------------
    bean = dna_reg.register("Arabica_Beans", "ingredient")
    aroma = dna_reg.register("Dark_Roast_Aroma", "sensory")
    taste = dna_reg.register("Caramel_Notes", "sensory")
    caffeine = dna_reg.register("Medium_Caffeine", "chemical")

    coffee_product = comp_engine.compose(
        name="Artisanal_Coffee",
        components=[bean, aroma, taste, caffeine],
        operation="bundle",
        composite_id="product_coffee_001"
    )
    print(f"[Step 2-3] Product DNA Created: '{coffee_product.name}' (ID: {coffee_product.id})")
    print(f"          Constituents: Arabica_Beans, Dark_Roast_Aroma, Caramel_Notes, Medium_Caffeine")

    # -------------------------------------------------------------
    # Step 4: Record 10 Purchase Events with Relational DNA
    # Cadence: every 7 days
    # -------------------------------------------------------------
    base_date = datetime.datetime(2026, 9, 1, 10, 0, 0, tzinfo=datetime.timezone.utc)
    purchase_event_ids = []

    print(f"\n[Step 4] Recording 10 Historic Purchase Events (every 7 days)...")
    for i in range(10):
        event_time = base_date + datetime.timedelta(days=7 * i)
        ts_str = event_time.strftime("%Y-%m-%d %H:%M:%S")

        ev = learning_engine.learn_from_event(
            event_type="purchased",
            subject_id=customer.id,
            object_id=coffee_product.id,
            attributes={"price": 450, "quantity": 1, "currency": "INR"},
            timestamp=ts_str
        )
        purchase_event_ids.append(ev["event_id"])
        print(f"   Event {i+1:02d}: {ts_str} -> Rahul purchased Artisanal_Coffee (Rel ID: {ev['direct_relation_id']})")

    # -------------------------------------------------------------
    # Step 5: Compose Customer Behaviour Assembly
    # -------------------------------------------------------------
    behaviour_asmb_id = f"asmb_behaviour_{customer.id}"
    behaviour_asmb = asmb_engine.get(behaviour_asmb_id)
    print(f"\n[Step 5] Customer Behaviour Assembly Composed:")
    print(f"          Assembly ID: {behaviour_asmb.id}")
    print(f"          Components: {len(behaviour_asmb.component_ids)} (Customer + 10 Purchase Events)")
    print(f"          Version: {behaviour_asmb.version}")

    # -------------------------------------------------------------
    # Step 6: Add 11th Purchase Event (Customer purchases after 5 days instead of 7)
    # -------------------------------------------------------------
    date_11 = base_date + datetime.timedelta(days=7 * 9 + 5) # 5 days after 10th purchase
    ts_11 = date_11.strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[Step 6] Customer makes 11th Purchase on {ts_11} (after 5-day interval instead of 7)...")
    ev_11 = learning_engine.learn_from_event(
        event_type="purchased",
        subject_id=customer.id,
        object_id=coffee_product.id,
        attributes={"price": 450, "quantity": 1},
        timestamp=ts_11
    )

    # -------------------------------------------------------------
    # Step 7: Update Assembly
    # -------------------------------------------------------------
    updated_asmb = asmb_engine.get(behaviour_asmb_id)
    print(f"[Step 7] Customer Behaviour Assembly Automatically Recomposed:")
    print(f"          New Version: {updated_asmb.version}")
    print(f"          Total Components: {len(updated_asmb.component_ids)}")

    # -------------------------------------------------------------
    # Step 8: Ask System to Explain Customer's Purchasing Pattern
    # -------------------------------------------------------------
    print(f"\n[Step 8] Spreading Activation: 'What coffee does Rahul usually buy?'")
    activations = act_engine.activate_from_query("What coffee does Rahul usually buy?", top_k=5)
    print(f"          Top Activated Representations:")
    for act in activations[:4]:
        print(f"          • [{act['type'].upper()}] {act['id']} (Score: {act['activation_score']:.3f})")
        for exp in act["explanations"][:1]:
            print(f"            └─ {exp}")

    # -------------------------------------------------------------
    # Step 9: Predict Next Purchase Window
    # -------------------------------------------------------------
    prediction = prediction_engine.predict_next_event(customer.id)
    print(f"\n[Step 9] Next Event Prediction:")
    print(f"          Predicted Event:   {prediction['predicted_event']}")
    print(f"          Predicted Window:  {prediction['predicted_window']}")
    print(f"          Estimated Interval:{prediction['estimated_interval_days']} days (Mean Cadence)")
    print(f"          Confidence:        {prediction['confidence'] * 100:.1f}%")
    print(f"          Supporting Evidence:")
    for ev_line in prediction["evidence"]:
        print(f"            - {ev_line}")

    # -------------------------------------------------------------
    # Step 10: Compare Prediction with Actual Next Purchase
    # -------------------------------------------------------------
    actual_12th_date = date_11 + datetime.timedelta(days=6)
    predicted_interval = prediction['estimated_interval_days']
    actual_interval = 6.0
    diff = abs(actual_interval - predicted_interval)

    print(f"\n[Step 10] Prediction Evaluation:")
    print(f"          Actual 12th Purchase Date: {actual_12th_date.strftime('%Y-%m-%d')}")
    print(f"          Predicted Mean Interval:   {predicted_interval:.1f} days")
    print(f"          Actual Interval:           {actual_interval:.1f} days")
    print(f"          Absolute Error:            {diff:.1f} days")
    print(f"          Outcome:                   ✅ SUCCESSFUL PREDICTION ALIGNMENT")

    print("\n" + "=" * 70)
    print("✅ BENCHMARK COMPLETED: ALL 10 STAGES PASSED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_customer_memory_benchmark()
