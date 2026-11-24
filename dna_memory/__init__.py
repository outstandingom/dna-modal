"""
DNA Memory Architecture - Recursive Compositional Memory System
Provides atomic representations, holographic vector composition, relational DNA,
recursive assemblies, identity management, temporal reasoning, and predictive learning.
"""

from dna_memory.core.atomic_dna import AtomicDNA, DNAFeatureMetadata
from dna_memory.core.dna_encoder import DNAEncoder
from dna_memory.core.dna_registry import DNARegistry
from dna_memory.core.concept_identity import ConceptIdentity, ConceptIdentityRegistry, IdentityCollisionError, ConceptNotFoundError
from dna_memory.core.vector_operations import bundle, bind, unbind, cosine_similarity, normalize

from dna_memory.composition.composition_engine import CompositionEngine, CompositionManifest
from dna_memory.composition.decomposition_engine import DecompositionEngine
from dna_memory.composition.shared_component_engine import SharedComponentEngine, SharedComponentDefinition, InstanceCompositionResult

from dna_memory.relations.relation_dna import RelationDNA
from dna_memory.relations.relation_registry import RelationRegistry
from dna_memory.relations.relationship_lifecycle import RelationshipRecord, RelationshipLifecycleEngine, EvidenceItem

from dna_memory.assembly.recursive_composition import DNAAssembly, AssemblyEngine
from dna_memory.assembly.assembly_validator import AssemblyValidator, AssemblyStructuralError, CircularDependencyError
from dna_memory.assembly.dependency_manager import DependencyManager

from dna_memory.activation.activation_engine import DNAActivationEngine
from dna_memory.activation.retrieval_index import RetrievalIndex
from dna_memory.activation.hybrid_retrieval import HybridRetrievalEngine, SemanticEmbeddingAdapter, RetrievalCandidate
from dna_memory.activation.retrieval_reranker import RetrievalReranker, RerankedCandidate

from dna_memory.reasoning.multi_hop import FrontierReasoningEngine
from dna_memory.learning.learning_engine import MemoryLearningEngine
from dna_memory.learning.pattern_discovery import PatternDiscoveryEngine, DiscoveredPattern
from dna_memory.learning.difference_detector import DifferenceDetector, StructuralComparisonReport

from dna_memory.prediction.behaviour_prediction import BehaviourPredictionEngine
from dna_memory.prediction.temporal_patterns import TemporalPatternAnalyzer

from dna_memory.memory.temporal_memory import TemporalMemoryEngine, TemporalEvent, EntityHistoricalState
from dna_memory.memory.memory_consolidator import MemoryConsolidator, ConsolidationAudit
from dna_memory.memory.memory_integrity import MemoryIntegrityEngine, IntegrityCheckReport
from dna_memory.memory.persistence_adapter import SQLitePersistenceAdapter
from dna_memory.core.sine_brain_bridge import SineBrainBridge, BridgeSyncReport
from dna_memory.core.sharding import ShardManager, ShardMetrics
from dna_memory.integrations.graphiti_adapter import GraphitiAdapter, GraphitiEntity, GraphitiRelation, GraphitiEpisode, GraphitiSyncReport
from dna_memory.system import DNAMemorySystem

__version__ = "2.1.0"
