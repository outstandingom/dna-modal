"""
DNA Memory Architecture - Recursive Compositional Memory System
Provides atomic representations, holographic vector composition, relational DNA,
recursive assemblies, and predictive learning.
"""

from dna_memory.core.atomic_dna import AtomicDNA, DNAFeatureMetadata
from dna_memory.core.dna_encoder import DNAEncoder
from dna_memory.core.dna_registry import DNARegistry
from dna_memory.core.vector_operations import bundle, bind, unbind, cosine_similarity, normalize

from dna_memory.composition.composition_engine import CompositionEngine, CompositionManifest
from dna_memory.composition.decomposition_engine import DecompositionEngine

from dna_memory.relations.relation_dna import RelationDNA
from dna_memory.relations.relation_registry import RelationRegistry

from dna_memory.assembly.recursive_composition import DNAAssembly, AssemblyEngine

from dna_memory.activation.activation_engine import DNAActivationEngine
from dna_memory.reasoning.multi_hop import FrontierReasoningEngine

from dna_memory.learning.learning_engine import MemoryLearningEngine
from dna_memory.prediction.behaviour_prediction import BehaviourPredictionEngine
from dna_memory.prediction.temporal_patterns import TemporalPatternAnalyzer
from dna_memory.activation.retrieval_index import RetrievalIndex
from dna_memory.system import DNAMemorySystem

__version__ = "2.0.0"
