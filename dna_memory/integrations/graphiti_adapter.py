"""
Graphiti & Neo4j AuraDB Integration Adapter.

Provides temporal knowledge graph extraction, episodic memory management,
and bi-directional synchronization between Graphiti/Neo4j and DNA Modal.

References:
  - Graphiti by Zep: https://github.com/getzep/graphiti
  - Neo4j AuraDB: https://console.neo4j.io
"""

import os
import re
import json
import hashlib
import datetime
import threading
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

try:
    import neo4j
    from neo4j import GraphDatabase
    HAS_NEO4J_DRIVER = True
except ImportError:
    HAS_NEO4J_DRIVER = False


@dataclass
class GraphitiEntity:
    """Represents a node entity in Graphiti / Neo4j."""
    id: str
    name: str
    entity_type: str
    summary: str = ""
    created_at: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "entity_type": self.entity_type,
            "summary": self.summary,
            "created_at": self.created_at,
            "attributes": self.attributes
        }


@dataclass
class GraphitiRelation:
    """Represents a directed temporal relationship in Graphiti / Neo4j."""
    id: str
    source_id: str
    predicate: str
    target_id: str
    valid_at: str
    invalid_at: Optional[str] = None
    confidence: float = 1.0
    episodes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "predicate": self.predicate,
            "target_id": self.target_id,
            "valid_at": self.valid_at,
            "invalid_at": self.invalid_at,
            "confidence": self.confidence,
            "episodes": self.episodes
        }


@dataclass
class GraphitiEpisode:
    """Represents a conversation turn / episode node."""
    id: str
    content: str
    user_id: str
    timestamp: str

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "user_id": self.user_id,
            "timestamp": self.timestamp
        }


@dataclass
class GraphitiSyncReport:
    """Summary report of syncing Graphiti Neo4j data into DNA Modal."""
    entities_synced: int
    relations_synced: int
    duration_ms: float
    status: str
    timestamp: str


class GraphitiAdapter:
    """
    Adapter bridging Graphiti (Zep) / Neo4j AuraDB with DNA Modal.
    Supports live Neo4j AuraDB instances as well as a zero-dependency in-memory
    shadow graph for reliable fallback and testing.
    """

    def __init__(
        self,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None,
        database: str = "neo4j"
    ):
        self.uri = neo4j_uri or os.getenv("NEO4J_URI", "")
        self.user = neo4j_user or os.getenv("NEO4J_USER", os.getenv("NEO4J_USERNAME", "neo4j"))
        self.password = neo4j_password or os.getenv("NEO4J_PASSWORD", "")
        self.database = database
        self._lock = threading.RLock()

        self._driver = None
        self._is_connected = False
        self._connection_error = None

        # Local in-memory shadow store (always available for fallback & test)
        self._shadow_entities: Dict[str, GraphitiEntity] = {}
        self._shadow_relations: Dict[str, GraphitiRelation] = {}
        self._shadow_episodes: List[GraphitiEpisode] = []

        self._init_driver()

    def _init_driver(self) -> bool:
        """Attempts to initialize connection to Neo4j instance."""
        if not HAS_NEO4J_DRIVER:
            self._connection_error = "neo4j driver not installed; running in shadow in-memory mode"
            self._is_connected = False
            return False

        if not self.uri or not self.password:
            self._connection_error = "NEO4J_URI or NEO4J_PASSWORD not configured; running in shadow in-memory mode"
            self._is_connected = False
            return False

        try:
            auth = (self.user, self.password)
            self._driver = GraphDatabase.driver(self.uri, auth=auth)
            # Test connectivity
            self._driver.verify_connectivity()
            self._is_connected = True
            self._connection_error = None
            return True
        except Exception as e:
            self._driver = None
            self._is_connected = False
            self._connection_error = str(e)
            return False

    def get_status(self) -> Dict[str, Any]:
        """Returns connection and telemetry status of the Graphiti/Neo4j engine."""
        with self._lock:
            # Re-check connectivity if driver exists
            if self._driver and not self._is_connected:
                self._init_driver()

            node_count = len(self._shadow_entities)
            edge_count = len(self._shadow_relations)
            episode_count = len(self._shadow_episodes)

            if self._is_connected and self._driver:
                try:
                    with self._driver.session(database=self.database) as session:
                        res = session.run("MATCH (n) RETURN count(n) AS node_count")
                        rec = res.single()
                        if rec:
                            node_count = rec["node_count"]
                        res_r = session.run("MATCH ()-[r]->() RETURN count(r) AS rel_count")
                        rec_r = res_r.single()
                        if rec_r:
                            edge_count = rec_r["rel_count"]
                except Exception as e:
                    self._connection_error = f"Query error: {str(e)}"

            masked_uri = self.uri
            if "@" in masked_uri:
                masked_uri = masked_uri.split("@")[-1]

            return {
                "configured": bool(self.uri and self.password),
                "connected": self._is_connected,
                "neo4j_driver_installed": HAS_NEO4J_DRIVER,
                "uri": masked_uri or "not configured (console.neo4j.io)",
                "database": self.database,
                "error": self._connection_error,
                "node_count": node_count,
                "edge_count": edge_count,
                "episode_count": episode_count,
                "mode": "NEO4J_LIVE" if self._is_connected else "SHADOW_FALLBACK"
            }

    def add_episode(
        self,
        user_id: str,
        message: str,
        timestamp: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Processes a conversation turn / episode:
        Extracts temporal entities and relations, records an Episode node,
        and saves them to Neo4j (and shadow store).
        """
        now_iso = timestamp or datetime.datetime.now(datetime.timezone.utc).isoformat()
        ep_id = f"ep_{hashlib.sha256(f'{user_id}:{message}:{now_iso}'.encode()).hexdigest()[:12]}"

        episode = GraphitiEpisode(
            id=ep_id,
            content=message,
            user_id=user_id,
            timestamp=now_iso
        )

        # Extract entities & relationships via rule-based parser or LLM extraction pattern
        extracted_entities, extracted_relations = self._extract_graph_elements(user_id, message, now_iso, ep_id)

        with self._lock:
            # Save to shadow store
            self._shadow_episodes.append(episode)
            for ent in extracted_entities:
                self._shadow_entities[ent.id] = ent
            for rel in extracted_relations:
                self._shadow_relations[rel.id] = rel

            # If connected to live Neo4j, write Cypher
            if self._is_connected and self._driver:
                try:
                    self._write_to_neo4j(episode, extracted_entities, extracted_relations)
                except Exception as e:
                    self._connection_error = f"Write error: {str(e)}"

        return {
            "episode_id": ep_id,
            "user_id": user_id,
            "extracted_entities": [e.to_dict() for e in extracted_entities],
            "extracted_relations": [r.to_dict() for r in extracted_relations],
            "timestamp": now_iso,
            "status": "RECORDED"
        }

    def _extract_graph_elements(
        self,
        user_id: str,
        text: str,
        timestamp: str,
        episode_id: str
    ) -> Tuple[List[GraphitiEntity], List[GraphitiRelation]]:
        """
        Extracts entities and temporal relationships from dialogue text.
        Recognizes user identity, roles, companies, technologies, and temporal markers.
        """
        entities: List[GraphitiEntity] = []
        relations: List[GraphitiRelation] = []

        clean_user = str(user_id).strip()
        user_ent_id = f"user_{clean_user}"
        user_ent = GraphitiEntity(
            id=user_ent_id,
            name=clean_user,
            entity_type="User",
            summary=f"User entity {clean_user}"
        )
        entities.append(user_ent)

        m = f" {text.strip()} "

        # 1. Company / Employer extraction
        comp_match = re.search(
            r'(?:work at|works at|joined|employed by|company is|startup is)\s+([A-Za-z0-9\.\-_ ]{2,30}?)(?:\.|\,|$|\n|\b(?:in|as|at|for)\b)',
            m,
            re.IGNORECASE
        )
        if comp_match:
            c_name = comp_match.group(1).strip()
            cid = f"org_{re.sub(r'[^a-z0-9_]', '', c_name.lower().replace(' ', '_'))[:20]}"
            comp_ent = GraphitiEntity(
                id=cid,
                name=c_name,
                entity_type="Organization",
                summary=f"Company {c_name}"
            )
            entities.append(comp_ent)
            rel_id = f"rel_{user_ent_id}_works_at_{cid}"
            relations.append(GraphitiRelation(
                id=rel_id,
                source_id=user_ent_id,
                predicate="works_at",
                target_id=cid,
                valid_at=timestamp,
                episodes=[episode_id]
            ))

        # 2. Role extraction
        role_match = re.search(
            r'(?:role is|position is|working as|hired as|i am an?|i am the|as an?|as the)\s+([A-Za-z\s]{3,35}?)(?:\.|\,|$|\n|\b(?:at|for|working|with|in)\b)',
            m,
            re.IGNORECASE
        )
        if role_match:
            raw_role = role_match.group(1).strip()
            # Preserve capitalization of known acronyms (AI, ML, CTO, CEO)
            words = raw_role.split()
            r_name = " ".join(w.upper() if w.upper() in {"AI", "ML", "CEO", "CTO", "COO", "CFO", "VP"} else w.capitalize() for w in words)
            rid = f"role_{re.sub(r'[^a-z0-9_]', '', r_name.lower().replace(' ', '_'))[:20]}"
            role_ent = GraphitiEntity(
                id=rid,
                name=r_name,
                entity_type="Role",
                summary=f"Professional Role {r_name}"
            )
            entities.append(role_ent)
            rel_id = f"rel_{user_ent_id}_has_role_{rid}"
            relations.append(GraphitiRelation(
                id=rel_id,
                source_id=user_ent_id,
                predicate="has_role",
                target_id=rid,
                valid_at=timestamp,
                episodes=[episode_id]
            ))

        # 3. Technologies / Topics
        tech_match = re.findall(r'\b(Python|Neo4j|Graphiti|FastAPI|React|Next\.js|Docker|Rust|Solidity|AI|Cybersecurity|Blockchain|Kubernetes)\b', m, re.IGNORECASE)
        if tech_match:
            for t in set(tech_match):
                tid = f"tech_{t.lower()}"
                entities.append(GraphitiEntity(
                    id=tid,
                    name=t.capitalize(),
                    entity_type="Technology",
                    summary=f"Technology / Skill {t}"
                ))
                relations.append(GraphitiRelation(
                    id=f"rel_{user_ent_id}_uses_{tid}",
                    source_id=user_ent_id,
                    predicate="uses_technology",
                    target_id=tid,
                    valid_at=timestamp,
                    episodes=[episode_id]
                ))

        return entities, relations

    def _write_to_neo4j(
        self,
        episode: GraphitiEpisode,
        entities: List[GraphitiEntity],
        relations: List[GraphitiRelation]
    ):
        """Executes Cypher commands to persist elements into Neo4j AuraDB."""
        if not self._driver:
            return

        with self._driver.session(database=self.database) as session:
            # 1. Merge Episode node
            session.run(
                """
                MERGE (ep:Episode {id: $id})
                SET ep.content = $content, ep.user_id = $user_id, ep.timestamp = datetime($timestamp)
                """,
                id=episode.id,
                content=episode.content,
                user_id=episode.user_id,
                timestamp=episode.timestamp
            )

            # 2. Merge Entities
            for ent in entities:
                session.run(
                    """
                    MERGE (e:Entity {id: $id})
                    SET e.name = $name, e.entity_type = $entity_type, e.summary = $summary, e.updated_at = datetime()
                    """,
                    id=ent.id,
                    name=ent.name,
                    entity_type=ent.entity_type,
                    summary=ent.summary
                )

            # 3. Merge Relations with Temporal Properties
            for rel in relations:
                session.run(
                    """
                    MATCH (s:Entity {id: $source_id})
                    MATCH (t:Entity {id: $target_id})
                    MERGE (s)-[r:RELATION {id: $id}]->(t)
                    SET r.predicate = $predicate,
                        r.valid_at = datetime($valid_at),
                        r.confidence = $confidence
                    """,
                    id=rel.id,
                    source_id=rel.source_id,
                    target_id=rel.target_id,
                    predicate=rel.predicate,
                    valid_at=rel.valid_at,
                    confidence=rel.confidence
                )

    def search(self, query: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Searches the Graphiti graph for relevant entities, relationships, and episodes.
        """
        clean_q = query.strip().lower()
        matched_entities = []
        matched_relations = []

        with self._lock:
            # Search shadow entities
            for ent in self._shadow_entities.values():
                if clean_q in ent.name.lower() or clean_q in ent.entity_type.lower():
                    matched_entities.append(ent.to_dict())

            # Search shadow relations
            for rel in self._shadow_relations.values():
                if user_id and rel.source_id != f"user_{user_id}":
                    continue
                if clean_q in rel.predicate.lower() or clean_q in rel.target_id.lower() or clean_q in rel.source_id.lower():
                    matched_relations.append(rel.to_dict())

        return {
            "query": query,
            "user_id": user_id,
            "entities": matched_entities[:10],
            "relations": matched_relations[:10],
            "total_matches": len(matched_entities) + len(matched_relations)
        }

    def sync_to_dna_memory(self, dna_system: Any) -> GraphitiSyncReport:
        """
        Synchronizes all extracted Graphiti/Neo4j entities and relationships
        into DNA Modal's canonical concept identity registry and relationship lifecycle engine.
        This enables training / grounding DNA Modal directly from Graphiti.
        """
        start_time = datetime.datetime.now()
        e_synced = 0
        r_synced = 0

        with self._lock:
            entities = list(self._shadow_entities.values())
            relations = list(self._shadow_relations.values())

        # Sync Entities into DNA Modal Concept Registry
        for ent in entities:
            try:
                dna_system.concept_registry.register(
                    canonical_id=ent.id,
                    canonical_name=ent.name,
                    domain=ent.entity_type.lower(),
                    aliases=[ent.name.lower()],
                    source="graphiti_neo4j_sync"
                )
                e_synced += 1
            except Exception:
                # If already exists, update attributes
                try:
                    dna_system.concept_registry.update_attributes(
                        canonical_id=ent.id,
                        new_attributes={"summary": ent.summary},
                        source="graphiti_sync"
                    )
                    e_synced += 1
                except Exception:
                    pass

        # Sync Relationships into DNA Modal Relationship Lifecycle
        for rel in relations:
            try:
                dna_system.relationship_lifecycle.create_relationship(
                    subject_id=rel.source_id,
                    predicate=rel.predicate,
                    object_id=rel.target_id,
                    confidence=rel.confidence,
                    source="graphiti_neo4j_sync",
                    evidence_content=f"Ingested from Graphiti temporal episode at {rel.valid_at}",
                    valid_from=rel.valid_at
                )
                r_synced += 1
            except Exception:
                pass

        duration = (datetime.datetime.now() - start_time).total_seconds() * 1000.0

        return GraphitiSyncReport(
            entities_synced=e_synced,
            relations_synced=r_synced,
            duration_ms=round(duration, 2),
            status="SUCCESS",
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
        )
