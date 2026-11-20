"""
Universal User Memory Architecture:
Consolidates the three core pillars of the DNA Cognitive Architecture:
  1. Recursive DNA Composition: User profiles as multi-tier hierarchical assemblies.
  2. First-Class Relational DNA: R_ABC triples with algebraic holographic unbinding.
  3. Structural Manifest + Composite Vector: Dual-representation allowing vector search & lossless structural recall.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import datetime
import re
import numpy as np

from dna_memory.core.atomic_dna import AtomicDNA
from dna_memory.core.dna_registry import DNARegistry
from dna_memory.relations.relation_registry import RelationRegistry
from dna_memory.relations.relation_dna import RelationDNA
from dna_memory.assembly.recursive_composition import AssemblyEngine, DNAAssembly
from dna_memory.core.vector_operations import bind, unbind, cosine_similarity, bundle, normalize

@dataclass
class UserMemoryManifest:
    """
    Authoritative structural manifest for a user's universal long-term memory.
    """
    user_id: str
    identity_facts: Dict[str, str]
    preferences: Dict[str, Any]
    relations: List[str]
    last_updated: str
    version: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "identity_facts": self.identity_facts,
            "preferences": self.preferences,
            "relations": self.relations,
            "last_updated": self.last_updated,
            "version": self.version,
            "metadata": self.metadata
        }

@dataclass
class AgentGroundingContext:
    """
    High-density, structured context payload for downstream AI agents.
    Contains grounded facts, user profile, active domain facts, and direct answers.
    """
    user_id: str
    user_profile: Dict[str, str]
    direct_answers: Dict[str, Any]
    relevant_domain_facts: List[str]
    active_concepts: List[str]
    prediction: Dict[str, Any]
    prompt_context_string: str

class UniversalUserMemory:
    """
    Universal Cognitive Long-Term User Memory Engine.
    Operates across any AI agent, channel, or application.
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
        self._user_manifests: Dict[str, UserMemoryManifest] = {}

    def remember_fact(
        self,
        user_id: str,
        relation_type: str,
        fact_value: str,
        confidence: float = 1.0,
        source: str = "dialogue"
    ) -> RelationDNA:
        """
        Encodes a single fact into Relational DNA (R_ABC) and recompiles the User Profile Assembly.
        """
        clean_user_id = str(user_id).strip()
        clean_val = str(fact_value).strip()
        rel_type = str(relation_type).strip().lower()

        # 1. Register/Retrieve User Concept Atom
        self.dna_registry.register(
            value=clean_user_id,
            dna_type="customer",
            source="user_memory",
            custom_id=clean_user_id
        )

        # 2. Register/Retrieve Fact Value Atom
        slug = re.sub(r'[^a-z0-9_]', '', clean_val.lower().replace(' ', '_'))[:30]
        fact_id = f"fact_{rel_type}_{slug}"
        fact_atom = self.dna_registry.register(
            value=clean_val,
            dna_type="customer_attribute",
            source=source,
            confidence=confidence,
            custom_id=fact_id
        )

        # 3. Create First-Class Relational DNA: R_ABC = bind(bind(User, rel_type), Fact)
        rel = self.relation_registry.create_relation(
            subject_id=clean_user_id,
            relation_type=rel_type,
            object_id=fact_atom.id,
            confidence=confidence,
            strength=1.0
        )

        # Reciprocal Inverse Relation: Fact --[is_rel_of]--> User
        self.relation_registry.create_relation(
            subject_id=fact_atom.id,
            relation_type=f"is_{rel_type}_of",
            object_id=clean_user_id,
            confidence=confidence,
            strength=0.9
        )

        # 4. Pillar 1 & 3: Recompile the UserProfile Assembly (Recursive Composition)
        self._recompile_user_profile_assembly(clean_user_id)

        return rel

    def unbind_relation(self, subject_id: str, relation_type: str) -> Optional[Tuple[str, float]]:
        """
        Pillar 2: Algebraic Vector Unbinding:
        Given Subject and Relation Type, unbinds the Object from Relational DNA space.
        """
        sub_atom = self.dna_registry.get(subject_id)
        if not sub_atom:
            return None

        # Probe vector: p = S * R
        r_vec = self.dna_registry.encoder.encode_text(f"RELATION:{relation_type.lower().strip()}")
        probe = bind(sub_atom.vector, r_vec)

        # Retrieve candidate relations
        rels = self.relation_registry.retrieve_relations(subject_id=subject_id, relation_type=relation_type)
        if not rels:
            return None

        best_atom = None
        best_sim = -1.0

        for r in rels:
            # Unbind O_rec = unbind(probe, r.vector)
            o_rec = unbind(probe, r.vector)
            obj_atom = self.dna_registry.get(r.object_id)
            if obj_atom is not None:
                sim = cosine_similarity(o_rec, obj_atom.vector)
                if sim > best_sim:
                    best_sim = sim
                    best_atom = obj_atom

        if best_atom:
            return best_atom.value, best_sim
        return None

    def reconstruct_profile(self, user_id: str) -> Dict[str, str]:
        """
        Pillar 3: Lossless Structural Reconstruction via Manifest & Relational Graph.
        """
        clean_user_id = str(user_id).strip()
        rels = self.relation_registry.retrieve_relations(subject_id=clean_user_id)
        # Sort by creation sequence so latest updates take absolute priority
        rels.sort(key=lambda r: r.id)
        facts = {}
        for r in reversed(rels):
            if r.relation_type not in facts and not r.relation_type.startswith("inverse_") and not r.relation_type.startswith("is_"):
                obj_atom = self.dna_registry.get(r.object_id)
                facts[r.relation_type] = obj_atom.value if obj_atom else r.object_id
        return facts

    def _recompile_user_profile_assembly(self, user_id: str):
        """
        Pillar 1: Recursively compiles UserProfile Assembly from constituent fact atoms.
        """
        clean_user_id = str(user_id).strip()
        rels = self.relation_registry.retrieve_relations(subject_id=clean_user_id)
        comp_ids = [r.object_id for r in rels if not r.relation_type.startswith("inverse_") and not r.relation_type.startswith("is_")]
        rel_ids = [r.id for r in rels]

        asmb_id = f"asmb_profile_{clean_user_id}"
        if comp_ids:
            self.assembly_engine.create_assembly(
                assembly_type="UserProfile",
                component_ids=comp_ids,
                relation_ids=rel_ids,
                metadata={
                    "user_id": clean_user_id,
                    "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
                },
                custom_id=asmb_id
            )

    def extract_from_dialogue(self, user_id: str, text: str) -> Dict[str, str]:
        """
        Universal Dialogue Fact Extractor:
        Extracts user identities, company, websites, roles, and preferences from text.
        """
        learned = {}
        m = f" {text.strip()} "

        # 1. Name Extraction
        name_match = re.search(r'(?:my name is|i am|i\'m|call me|this is)\s+([A-Za-z\s]{2,30}?)(?:\.|\,|$|\n|!|\band\b)', m, re.IGNORECASE)
        if name_match:
            candidate = name_match.group(1).strip()
            stopwords = {"looking", "interested", "here", "ready", "fine", "good", "a", "an", "the", "not", "just", "trying", "wondering", "asking", "happy", "new", "website", "having", "facing"}
            words = candidate.lower().split()
            if candidate and not any(w in stopwords for w in words) and len(candidate) > 1:
                clean_name = " ".join(w.capitalize() for w in words)
                learned["has_name"] = clean_name
                self.remember_fact(user_id, "has_name", clean_name)

        # 2. Company Extraction
        comp_match = re.search(r'(?:my company is|i work at|my startup is|my business is|our company is)\s+([A-Za-z0-9\.\-_ ]+?)(?:\.|\,|$|\n)', m, re.IGNORECASE)
        if comp_match:
            candidate_comp = comp_match.group(1).strip()
            if len(candidate_comp) > 1:
                learned["has_company"] = candidate_comp
                self.remember_fact(user_id, "has_company", candidate_comp)

        # 3. Role Extraction
        role_match = re.search(r'(?:i am a|i am the|i\'m a|i\'m the)\s+(founder|co-founder|developer|engineer|ceo|cto|owner|manager|designer|director)', m, re.IGNORECASE)
        if role_match:
            role = role_match.group(1).capitalize()
            learned["has_role"] = role
            self.remember_fact(user_id, "has_role", role)

        # 4. Website Extraction
        url_match = re.search(r'(https?://[^\s]+|[a-zA-Z0-9-]+\.(com|in|org|net|io|co|ai|app)[^\s]*)', m, re.IGNORECASE)
        if url_match:
            url = url_match.group(1).strip()
            learned["has_website"] = url
            self.remember_fact(user_id, "has_website", url)

        return learned

    def resolve_universal_query(self, user_id: str, query_text: str) -> Dict[str, Any]:
        """
        Direct Query Resolution:
        Uses holographic vector unbinding and relational graph lookup to directly answer
        queries about the user or domain without needing application-level bot if/else rules.
        """
        q = query_text.lower().strip()
        user_facts = self.reconstruct_profile(user_id)
        direct_answers = {}

        # User identity questions (only trigger on actual questions, not declarations)
        if any(p in q for p in ["what is my name", "what's my name", "who am i", "do you know my name", "remember my name", "tell me my name", "my name?"]):
            val = user_facts.get("has_name")
            if val:
                direct_answers["user_name"] = val
        if any(p in q for p in ["what is my company", "what's my company", "where do i work", "my company?"]):
            val = user_facts.get("has_company")
            if val:
                direct_answers["user_company"] = val
        if any(p in q for p in ["what is my website", "what's my website", "what website", "my website?"]):
            val = user_facts.get("has_website")
            if val:
                direct_answers["user_website"] = val

        # Domain relational queries (Founder, Payment, Headquarters, Turnaround)
        if any(p in q for p in ["founder", "who started", "who founded", "who created", "who owns"]):
            res = self.unbind_relation("company_growhaz", "founded_by")
            if res:
                direct_answers["founder"] = res[0]
            else:
                rels = self.relation_registry.retrieve_relations(subject_id="company_growhaz", relation_type="founded_by")
                if rels:
                    atom = self.dna_registry.get(rels[0].object_id)
                    direct_answers["founder"] = atom.value if atom else rels[0].object_id

        if any(p in q for p in ["payment", "pay", "upi", "credit card", "netbanking", "card"]):
            rels = self.relation_registry.retrieve_relations(subject_id="company_growhaz", relation_type="accepts_payment_method")
            if rels:
                methods = []
                for r in rels:
                    atom = self.dna_registry.get(r.object_id)
                    methods.append(atom.value if atom else r.object_id)
                direct_answers["payment_methods"] = ", ".join(methods)

        if any(p in q for p in ["turnaround", "delivery time", "how long does it take"]):
            rels = self.relation_registry.retrieve_relations(relation_type="delivery_time")
            timeline = {}
            for r in rels[:4]:
                s_atom = self.dna_registry.get(r.subject_id)
                o_atom = self.dna_registry.get(r.object_id)
                s_val = s_atom.value if s_atom else r.subject_id
                o_val = o_atom.value if o_atom else r.object_id
                timeline[s_val] = o_val
            if timeline:
                direct_answers["turnaround_times"] = timeline

        return {
            "user_id": user_id,
            "user_facts": user_facts,
            "direct_answers": direct_answers
        }

    def synthesize_agent_grounding(
        self,
        user_id: str,
        query_text: str,
        activated_nodes: Optional[List[dict]] = None
    ) -> AgentGroundingContext:
        """
        Synthesizes a complete, noise-free, grounded context payload for any AI agent.
        """
        clean_user_id = str(user_id).strip()
        user_facts = self.reconstruct_profile(clean_user_id)
        resolved = self.resolve_universal_query(clean_user_id, query_text)

        domain_facts = []
        if activated_nodes:
            top_ids = [n["id"] for n in activated_nodes[:4]]
            for nid in top_ids:
                rels = self.relation_registry.retrieve_relations(subject_id=nid)
                for r in rels[:3]:
                    if not r.relation_type.startswith("inverse_") and not r.relation_type.startswith("is_synonym"):
                        s_name = r.subject_id.replace("company_", "").replace("service_", "").replace("payment_", "").replace("_", " ").title()
                        o_atom = self.dna_registry.get(r.object_id)
                        o_name = (o_atom.value if o_atom else r.object_id).replace("price_inr_", "₹").replace("_", " ").title()
                        rel_name = r.relation_type.replace("_", " ")
                        domain_facts.append(f"{s_name} {rel_name}: {o_name}")

        # Build clean prompt context string
        parts = [
            f"[UNIVERSAL COGNITIVE MEMORY GROUNDING - USER: {clean_user_id}]",
            f"• Current Query: \"{query_text}\"",
        ]

        if user_facts:
            parts.append("• Verified User Memory Profile (Reconstructed Losslessly):")
            for k, v in user_facts.items():
                label = k.replace("has_", "").title()
                parts.append(f"  - {label}: {v}")

        if resolved.get("direct_answers"):
            parts.append("• Direct Relational Knowledge Query Results:")
            for k, v in resolved["direct_answers"].items():
                parts.append(f"  - {k.replace('_', ' ').title()}: {v}")

        if domain_facts:
            parts.append("• Activated Domain Knowledge Triples (First-Class Relational DNA):")
            for f in domain_facts[:5]:
                parts.append(f"  └─ {f}")

        parts.append("• AGENT INSTRUCTION: Answer the user's inquiry directly, accurately, and naturally using the verified knowledge above. Address the user by name if known. Never mention internal IDs.")

        return AgentGroundingContext(
            user_id=clean_user_id,
            user_profile=user_facts,
            direct_answers=resolved.get("direct_answers", {}),
            relevant_domain_facts=domain_facts,
            active_concepts=[n["id"] for n in (activated_nodes or [])],
            prediction={},
            prompt_context_string="\n".join(parts)
        )
