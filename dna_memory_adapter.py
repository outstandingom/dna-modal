"""
DNA Memory Adapter:
Bridges the Recursive Compositional DNA Memory Architecture (dna_memory)
into the existing FastAPI / KnowledgeGraphEnv ecosystem.
"""

from dna_memory.core.dna_encoder import DNAEncoder
from dna_memory.core.dna_registry import DNARegistry
from dna_memory.composition.composition_engine import CompositionEngine
from dna_memory.composition.decomposition_engine import DecompositionEngine
from dna_memory.relations.relation_registry import RelationRegistry
from dna_memory.assembly.recursive_composition import AssemblyEngine
from dna_memory.activation.activation_engine import DNAActivationEngine
from dna_memory.learning.learning_engine import MemoryLearningEngine
from dna_memory.prediction.behaviour_prediction import BehaviourPredictionEngine

class DNAMemorySystem:
    """
    Unified high-level facade for the Recursive Compositional DNA Memory Architecture.
    """
    def __init__(self, dims: int = 128, seed: int = 42):
        self.encoder = DNAEncoder(dims=dims, seed=seed)
        self.registry = DNARegistry(self.encoder)
        self.composition = CompositionEngine(self.registry)
        self.decomposition = DecompositionEngine(self.registry)
        self.relations = RelationRegistry(self.registry)
        self.assemblies = AssemblyEngine(self.registry, self.relations)
        self.activation = DNAActivationEngine(self.registry, self.relations, self.assemblies)
        self.learning = MemoryLearningEngine(self.registry, self.relations, self.assemblies)
        self.prediction = BehaviourPredictionEngine(self.learning, self.relations, self.assemblies)

    def record_client_interaction(
        self,
        client_phone: str,
        message: str,
        detected_need: str = None,
        budget: str = None,
        timestamp: str = None
    ) -> dict:
        """
        Record and learn from an inbound client message or interaction.
        """
        # Register or retrieve customer
        customer = self.registry.register(value=client_phone, dna_type="customer", custom_id=f"cust_{client_phone}")

        learned_event = None
        if detected_need:
            service_atom = self.registry.register(value=detected_need, dna_type="service")
            learned_event = self.learning.learn_from_event(
                event_type="inquired",
                subject_id=customer.id,
                object_id=service_atom.id,
                attributes={"budget": budget, "raw_message": message},
                timestamp=timestamp
            )

        return {
            "customer_id": customer.id,
            "event": learned_event
        }

    def get_client_sales_profile(self, client_phone: str) -> dict:
        """
        Returns full DNA-backed sales intelligence for this customer.
        """
        cust_id = f"cust_{client_phone}"
        asmb_id = f"asmb_behaviour_{cust_id}"
        
        history = self.learning.get_event_history(subject_id=cust_id)
        pred = self.prediction.predict_next_event(cust_id)

        return {
            "client_phone": client_phone,
            "total_interactions": len(history),
            "behaviour_assembly_id": asmb_id,
            "prediction": pred,
            "history": history
        }
