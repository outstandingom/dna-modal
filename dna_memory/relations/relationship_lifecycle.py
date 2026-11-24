"""
Relationship Lifecycle Engine: Manages persistent, updateable, versioned relational bonds.
Preserves historical states, handles contradictions, tracks evidence/provenance,
and enforces non-destructive relationship evolution.
"""

from dataclasses import dataclass, field
import datetime
import hashlib
import threading
from typing import Dict, List, Optional, Set, Any, Tuple


class RelationshipValidationError(ValueError):
    """Raised when relationship parameters violate relational integrity constraints."""
    pass


class RelationshipNotFoundError(KeyError):
    """Raised when querying a non-existent relationship ID."""
    pass


@dataclass
class EvidenceItem:
    """Provenance unit containing factual evidence supporting a relationship."""
    evidence_id: str
    source: str
    content: str
    timestamp: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    confidence_weight: float = 1.0

    def to_dict(self) -> dict:
        return {
            "evidence_id": self.evidence_id,
            "source": self.source,
            "content": self.content,
            "timestamp": self.timestamp,
            "confidence_weight": self.confidence_weight
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'EvidenceItem':
        return cls(
            evidence_id=data["evidence_id"],
            source=data.get("source", "unknown"),
            content=data.get("content", ""),
            timestamp=data.get("timestamp", ""),
            confidence_weight=float(data.get("confidence_weight", 1.0))
        )


@dataclass
class RelationshipVersionSnapshot:
    """Immutable snapshot of a relationship at a specific version."""
    version: int
    confidence: float
    strength: float
    status: str
    valid_from: str
    valid_until: Optional[str]
    recorded_at: str
    source: str
    evidence_count: int
    change_reason: str


@dataclass
class RelationshipRecord:
    """
    Complete lifecycle representation of an episodic or semantic relationship.
    Supports temporal bounds, supersession, evidence aggregation, and version history.
    """
    relationship_id: str
    subject_id: str
    predicate: str
    object_id: str
    confidence: float = 0.8
    strength: float = 1.0
    source: str = "direct_observation"
    evidence: List[EvidenceItem] = field(default_factory=list)
    valid_from: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    valid_until: Optional[str] = None
    recorded_at: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    status: str = "active"  # "active", "superseded", "deactivated", "contradicted"
    version: int = 1
    superseded_by: Optional[str] = None
    supersedes: Optional[str] = None
    contradicted_by: Optional[str] = None
    history: List[RelationshipVersionSnapshot] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_currently_valid(self, as_of: Optional[str] = None) -> bool:
        """
        Evaluates temporal and lifecycle validity.
        If as_of is provided (ISO timestamp), evaluates historical validity.
        """
        if self.status in ("deactivated", "contradicted"):
            return False

        check_time = as_of or datetime.datetime.now(datetime.timezone.utc).isoformat()

        if self.valid_from > check_time:
            return False
        if self.valid_until and self.valid_until < check_time:
            return False

        return self.status == "active" or (as_of is not None and self.status == "superseded" and (not self.valid_until or check_time <= self.valid_until))

    def add_evidence(self, item: EvidenceItem, update_confidence: bool = True) -> bool:
        """
        Adds evidence supporting this relationship.
        Applies asymptotic confidence scaling to avoid blind confidence inflation:
        Repeated evidence from the SAME source yields diminishing returns.
        Independent evidence from NEW sources provides higher confidence boosts.
        """
        # Check duplicate evidence
        for ev in self.evidence:
            if ev.evidence_id == item.evidence_id or (ev.source == item.source and ev.content == item.content):
                return False

        self.evidence.append(item)

        if update_confidence:
            # Count distinct sources
            sources = set(e.source for e in self.evidence)
            # Base confidence asymptotic formula: C = 1 - (1 - C0) * 0.7^(num_independent_sources)
            base_uncertainty = 1.0 - 0.5
            source_factor = 0.65 ** len(sources)
            # Diminishing return on evidence volume
            vol_factor = 1.0 / (1.0 + 0.1 * len(self.evidence))
            new_conf = 1.0 - (base_uncertainty * source_factor * vol_factor)
            self.confidence = min(0.99, max(0.1, new_conf))

        return True

    def snapshot(self, change_reason: str):
        """Records current state to historical version ledger."""
        self.history.append(RelationshipVersionSnapshot(
            version=self.version,
            confidence=self.confidence,
            strength=self.strength,
            status=self.status,
            valid_from=self.valid_from,
            valid_until=self.valid_until,
            recorded_at=self.recorded_at,
            source=self.source,
            evidence_count=len(self.evidence),
            change_reason=change_reason
        ))

    def supersede(self, new_relationship_id: str, effective_time: Optional[str] = None, reason: str = "superseded_by_newer_fact"):
        """Retires this relationship because a newer fact replaces it."""
        self.snapshot(reason)
        now = effective_time or datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.status = "superseded"
        self.valid_until = now
        self.superseded_by = new_relationship_id
        self.version += 1

    def deactivate(self, reason: str = "explicit_deactivation"):
        """Deactivates a relationship (retracted or invalidated)."""
        self.snapshot(reason)
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.status = "deactivated"
        self.valid_until = now
        self.version += 1

    def mark_contradicted(self, conflicting_relationship_id: str, reason: str = "contradicted_by_evidence"):
        """Flags relationship as contradicted by conflicting observation."""
        self.snapshot(reason)
        self.status = "contradicted"
        self.contradicted_by = conflicting_relationship_id
        self.version += 1

    def to_dict(self) -> dict:
        return {
            "relationship_id": self.relationship_id,
            "subject_id": self.subject_id,
            "predicate": self.predicate,
            "object_id": self.object_id,
            "confidence": self.confidence,
            "strength": self.strength,
            "source": self.source,
            "evidence": [e.to_dict() for e in self.evidence],
            "valid_from": self.valid_from,
            "valid_until": self.valid_until,
            "recorded_at": self.recorded_at,
            "status": self.status,
            "version": self.version,
            "superseded_by": self.superseded_by,
            "supersedes": self.supersedes,
            "contradicted_by": self.contradicted_by,
            "history": [
                {
                    "version": h.version,
                    "confidence": h.confidence,
                    "strength": h.strength,
                    "status": h.status,
                    "valid_from": h.valid_from,
                    "valid_until": h.valid_until,
                    "recorded_at": h.recorded_at,
                    "source": h.source,
                    "evidence_count": h.evidence_count,
                    "change_reason": h.change_reason
                }
                for h in self.history
            ],
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'RelationshipRecord':
        ev_items = [EvidenceItem.from_dict(e) for e in data.get("evidence", [])]
        history = [
            RelationshipVersionSnapshot(
                version=h["version"],
                confidence=h["confidence"],
                strength=h["strength"],
                status=h["status"],
                valid_from=h["valid_from"],
                valid_until=h.get("valid_until"),
                recorded_at=h["recorded_at"],
                source=h["source"],
                evidence_count=h["evidence_count"],
                change_reason=h.get("change_reason", "")
            )
            for h in data.get("history", [])
        ]
        return cls(
            relationship_id=data["relationship_id"],
            subject_id=data["subject_id"],
            predicate=data["predicate"],
            object_id=data["object_id"],
            confidence=float(data.get("confidence", 0.8)),
            strength=float(data.get("strength", 1.0)),
            source=data.get("source", "direct_observation"),
            evidence=ev_items,
            valid_from=data.get("valid_from", ""),
            valid_until=data.get("valid_until"),
            recorded_at=data.get("recorded_at", ""),
            status=data.get("status", "active"),
            version=data.get("version", 1),
            superseded_by=data.get("superseded_by"),
            supersedes=data.get("supersedes"),
            contradicted_by=data.get("contradicted_by"),
            history=history,
            metadata=data.get("metadata", {})
        )


class RelationshipLifecycleEngine:
    """
    Manages relationship lifecycles, transactional updates, contradiction detection,
    and index integrity.
    """
    def __init__(self, concept_registry=None):
        self._relations: Dict[str, RelationshipRecord] = {}     # rel_id -> record
        self._subject_index: Dict[str, Set[str]] = {}           # subject_id -> {rel_ids}
        self._object_index: Dict[str, Set[str]] = {}            # object_id -> {rel_ids}
        self._predicate_index: Dict[str, Set[str]] = {}         # predicate -> {rel_ids}
        self._spo_index: Dict[Tuple[str, str, str], str] = {}   # (s, p, o) -> rel_id
        self._concept_registry = concept_registry               # Optional ConceptIdentityRegistry
        self._lock = threading.RLock()

    def _generate_rel_id(self, subject_id: str, predicate: str, object_id: str) -> str:
        h = hashlib.sha256(f"{subject_id}:{predicate}:{object_id}".encode("utf-8")).hexdigest()[:16]
        return f"rel_{h}"

    def create_relationship(
        self,
        subject_id: str,
        predicate: str,
        object_id: str,
        confidence: float = 0.8,
        strength: float = 1.0,
        source: str = "direct_observation",
        evidence_content: Optional[str] = None,
        valid_from: Optional[str] = None,
        valid_until: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[RelationshipRecord, bool]:
        """
        Creates or updates a relationship.
        If an active (s, p, o) relationship already exists:
          - Ingests new evidence without creating duplicate relations.
          - Re-evaluates calibrated confidence.
          - Returns (existing_record, False).
        If new:
          - Validates subject & object references (if concept_registry is present).
          - Inserts and returns (new_record, True).
        """
        with self._lock:
            s = subject_id.strip()
            p = predicate.strip().lower()
            o = object_id.strip()

            if not s or not p or not o:
                raise RelationshipValidationError(f"Invalid SPO triple: subject='{s}', predicate='{p}', object='{o}'")

            # Validate and ensure subject & object reference integrity in concept registry
            if self._concept_registry:
                if not self._concept_registry.exists(s):
                    self._concept_registry.register(canonical_id=s, canonical_name=s.replace("_", " ").title(), source=source)
                if not self._concept_registry.exists(o):
                    self._concept_registry.register(canonical_id=o, canonical_name=o.replace("_", " ").title(), source=source)

            spo_key = (s, p, o)
            now = datetime.datetime.now(datetime.timezone.utc).isoformat()
            v_from = valid_from or now

            # Duplicate check
            if spo_key in self._spo_index:
                existing_id = self._spo_index[spo_key]
                existing = self._relations[existing_id]
                if existing.status == "active":
                    if evidence_content:
                        ev_id = f"ev_{hashlib.sha256(f'{source}:{evidence_content}:{now}'.encode()).hexdigest()[:12]}"
                        existing.add_evidence(EvidenceItem(
                            evidence_id=ev_id,
                            source=source,
                            content=evidence_content,
                            timestamp=now,
                            confidence_weight=1.0
                        ))
                    return existing, False

            rel_id = self._generate_rel_id(s, p, o)
            # In case an older inactive version had this ID, append timestamp suffix
            if rel_id in self._relations:
                rel_id = f"{rel_id}_{int(datetime.datetime.now().timestamp() * 1000)}"

            record = RelationshipRecord(
                relationship_id=rel_id,
                subject_id=s,
                predicate=p,
                object_id=o,
                confidence=max(0.0, min(1.0, confidence)),
                strength=max(0.0, min(1.0, strength)),
                source=source,
                valid_from=v_from,
                valid_until=valid_until,
                recorded_at=now,
                status="active",
                version=1,
                metadata=dict(metadata or {})
            )

            if evidence_content:
                ev_id = f"ev_{hashlib.sha256(f'{source}:{evidence_content}:{now}'.encode()).hexdigest()[:12]}"
                record.add_evidence(EvidenceItem(
                    evidence_id=ev_id,
                    source=source,
                    content=evidence_content,
                    timestamp=now,
                    confidence_weight=1.0
                ))

            self._relations[rel_id] = record
            self._spo_index[spo_key] = rel_id
            self._subject_index.setdefault(s, set()).add(rel_id)
            self._object_index.setdefault(o, set()).add(rel_id)
            self._predicate_index.setdefault(p, set()).add(rel_id)

            return record, True

    def supersede_relationship(
        self,
        old_relationship_id: str,
        new_object_id: str,
        source: str = "direct_observation",
        evidence_content: Optional[str] = None,
        effective_time: Optional[str] = None
    ) -> RelationshipRecord:
        """
        Atomically supersedes an old relationship with a new fact.
        e.g., worked_at CompanyA -> superseded by CompanyB.
        """
        with self._lock:
            if old_relationship_id not in self._relations:
                raise RelationshipNotFoundError(f"Relationship '{old_relationship_id}' not found.")

            old_rel = self._relations[old_relationship_id]
            now = effective_time or datetime.datetime.now(datetime.timezone.utc).isoformat()

            # Create new relationship
            new_rel, _ = self.create_relationship(
                subject_id=old_rel.subject_id,
                predicate=old_rel.predicate,
                object_id=new_object_id,
                confidence=old_rel.confidence,
                source=source,
                evidence_content=evidence_content,
                valid_from=now
            )

            # Retire old relationship
            old_rel.supersede(new_rel.relationship_id, effective_time=now)
            new_rel.supersedes = old_rel.relationship_id

            # Remove old from active SPO index
            old_spo = (old_rel.subject_id, old_rel.predicate, old_rel.object_id)
            if self._spo_index.get(old_spo) == old_rel.relationship_id:
                del self._spo_index[old_spo]

            return new_rel

    def deactivate_relationship(self, relationship_id: str, reason: str = "explicit_deactivation") -> bool:
        """Deactivates relationship and removes from active index."""
        with self._lock:
            if relationship_id not in self._relations:
                raise RelationshipNotFoundError(f"Relationship '{relationship_id}' not found.")
            rel = self._relations[relationship_id]
            rel.deactivate(reason=reason)
            spo = (rel.subject_id, rel.predicate, rel.object_id)
            if self._spo_index.get(spo) == relationship_id:
                del self._spo_index[spo]
            return True

    def get(self, relationship_id: str) -> RelationshipRecord:
        with self._lock:
            if relationship_id not in self._relations:
                raise RelationshipNotFoundError(f"Relationship '{relationship_id}' not found.")
            return self._relations[relationship_id]

    def find_relationships(
        self,
        subject_id: Optional[str] = None,
        predicate: Optional[str] = None,
        object_id: Optional[str] = None,
        as_of: Optional[str] = None,
        only_active: bool = True
    ) -> List[RelationshipRecord]:
        """
        Searches relationships matching optional criteria.
        Evaluates temporal validity if as_of is supplied.
        """
        with self._lock:
            candidates: Optional[Set[str]] = None

            if subject_id is not None:
                s_matches = self._subject_index.get(subject_id.strip(), set())
                candidates = set(s_matches) if candidates is None else candidates & s_matches

            if predicate is not None:
                p_matches = self._predicate_index.get(predicate.strip().lower(), set())
                candidates = set(p_matches) if candidates is None else candidates & p_matches

            if object_id is not None:
                o_matches = self._object_index.get(object_id.strip(), set())
                candidates = set(o_matches) if candidates is None else candidates & o_matches

            if candidates is None:
                candidates = set(self._relations.keys())

            results = []
            for rid in candidates:
                rel = self._relations[rid]
                if only_active:
                    if rel.is_currently_valid(as_of=as_of):
                        results.append(rel)
                else:
                    results.append(rel)

            return results

    def query_relationships(
        self,
        subject_id: Optional[str] = None,
        predicate: Optional[str] = None,
        object_id: Optional[str] = None,
        status: Optional[str] = None,
        as_of: Optional[str] = None
    ) -> List[RelationshipRecord]:
        """Query relationships with explicit status filtering."""
        with self._lock:
            matches = self.find_relationships(
                subject_id=subject_id,
                predicate=predicate,
                object_id=object_id,
                as_of=as_of,
                only_active=False
            )
            if status is not None:
                matches = [m for m in matches if m.status.lower() == status.lower()]
            return matches

    def count(self, only_active: bool = True) -> int:
        with self._lock:
            if not only_active:
                return len(self._relations)
            return sum(1 for r in self._relations.values() if r.status == "active")

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "relations": {rid: r.to_dict() for rid, r in self._relations.items()}
            }

    @classmethod
    def from_dict(cls, data: dict, concept_registry=None) -> 'RelationshipLifecycleEngine':
        engine = cls(concept_registry=concept_registry)
        for rid, r_data in data.get("relations", {}).items():
            record = RelationshipRecord.from_dict(r_data)
            engine._relations[rid] = record
            engine._subject_index.setdefault(record.subject_id, set()).add(rid)
            engine._object_index.setdefault(record.object_id, set()).add(rid)
            engine._predicate_index.setdefault(record.predicate, set()).add(rid)
            if record.status == "active":
                engine._spo_index[(record.subject_id, record.predicate, record.object_id)] = rid
        return engine
