"""
Concept Identity Engine: Manages stable canonical concept identities, aliases,
versioning, and separation of immutable identity from mutable attributes.

Prevents identity corruption, collision, and silent overwrites across long-term memory.
"""

from dataclasses import dataclass, field
import datetime
import hashlib
import re
import threading
from typing import Dict, List, Optional, Set, Any, Tuple


class IdentityCollisionError(ValueError):
    """Raised when an attempt is made to overwrite an existing concept identity improperly."""
    pass


class ConceptNotFoundError(KeyError):
    """Raised when a requested canonical concept ID does not exist."""
    pass


@dataclass
class ConceptVersionRecord:
    """Historical record of an attribute update for a concept."""
    version: int
    attributes: Dict[str, Any]
    source: str
    timestamp: str
    change_reason: str


@dataclass
class ConceptIdentity:
    """
    Represents the stable canonical identity of a concept.
    Identity (canonical_id, domain) is immutable.
    Attributes, aliases, and versions mutate under strict lifecycle rules.
    """
    canonical_id: str
    canonical_name: str
    domain: str = "general"
    aliases: Set[str] = field(default_factory=set)
    attributes: Dict[str, Any] = field(default_factory=dict)
    source: str = "direct_observation"
    version: int = 1
    created_at: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    history: List[ConceptVersionRecord] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_alias(self, alias: str) -> bool:
        """Adds a surface lexical expression pointing to this canonical concept."""
        clean_alias = alias.strip().lower()
        if not clean_alias:
            return False
        if clean_alias not in self.aliases:
            self.aliases.add(clean_alias)
            self.updated_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
            return True
        return False

    def update_attributes(self, new_attributes: Dict[str, Any], source: str, reason: str = "attribute_update") -> int:
        """
        Updates mutable attributes, recording a new version in the historical ledger.
        Preserves the canonical_id and immutable identity.
        """
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        # Record current version in history
        self.history.append(ConceptVersionRecord(
            version=self.version,
            attributes=dict(self.attributes),
            source=self.source,
            timestamp=self.updated_at,
            change_reason=reason
        ))
        # Apply updates
        self.attributes.update(new_attributes)
        self.version += 1
        self.source = source
        self.updated_at = now
        return self.version

    def get_version_at(self, version_num: int) -> Optional[Dict[str, Any]]:
        """Returns the attribute dictionary as it existed at a specific version."""
        if version_num == self.version:
            return dict(self.attributes)
        for h in self.history:
            if h.version == version_num:
                return dict(h.attributes)
        return None

    def to_dict(self) -> dict:
        return {
            "canonical_id": self.canonical_id,
            "canonical_name": self.canonical_name,
            "domain": self.domain,
            "aliases": sorted(list(self.aliases)),
            "attributes": self.attributes,
            "source": self.source,
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "history": [
                {
                    "version": h.version,
                    "attributes": h.attributes,
                    "source": h.source,
                    "timestamp": h.timestamp,
                    "change_reason": h.change_reason
                }
                for h in self.history
            ],
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'ConceptIdentity':
        history_records = [
            ConceptVersionRecord(
                version=h["version"],
                attributes=h["attributes"],
                source=h["source"],
                timestamp=h["timestamp"],
                change_reason=h.get("change_reason", "")
            )
            for h in data.get("history", [])
        ]
        return cls(
            canonical_id=data["canonical_id"],
            canonical_name=data["canonical_name"],
            domain=data.get("domain", "general"),
            aliases=set(data.get("aliases", [])),
            attributes=data.get("attributes", {}),
            source=data.get("source", "direct_observation"),
            version=data.get("version", 1),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            history=history_records,
            metadata=data.get("metadata", {})
        )


class ConceptIdentityRegistry:
    """
    Central registry for canonical concept identities.
    Guarantees:
      - 1:1 mapping between canonical ID and concept.
      - N:1 mapping from aliases / expressions to canonical IDs.
      - Collision detection preventing silent overwrites.
      - Distinct concept protection (e.g. 'car' vs 'truck').
    """
    def __init__(self):
        self._concepts: Dict[str, ConceptIdentity] = {}          # canonical_id -> ConceptIdentity
        self._alias_index: Dict[str, str] = {}                  # alias (lowercase) -> canonical_id
        self._lock = threading.RLock()

    def mint_id(
        self,
        canonical_name: str,
        domain: str = "general",
        source: str = "direct_observation",
        concept_type: Optional[str] = None
    ) -> ConceptIdentity:
        """Helper to register or retrieve a concept automatically deriving a clean collision-safe ID."""
        with self._lock:
            resolved = self.resolve(canonical_name)
            if resolved:
                return resolved
            prefix = f"concept_{concept_type}_" if concept_type else "concept_"
            slug = re.sub(r'[^a-z0-9_]', '', canonical_name.lower().replace(' ', '_'))[:30]
            cid = f"{prefix}{slug}"
            if cid in self._concepts and self._concepts[cid].canonical_name.lower() != canonical_name.lower():
                import hashlib
                h = hashlib.sha256(canonical_name.encode('utf-8')).hexdigest()[:6]
                cid = f"{cid}_{h}"
            return self.register(
                canonical_id=cid,
                canonical_name=canonical_name,
                domain=domain,
                aliases=[canonical_name],
                source=source
            )

    def register(
        self,
        canonical_id: str,
        canonical_name: str,
        domain: str = "general",
        aliases: Optional[List[str]] = None,
        attributes: Optional[Dict[str, Any]] = None,
        source: str = "direct_observation",
        metadata: Optional[Dict[str, Any]] = None
    ) -> ConceptIdentity:
        """
        Registers a new canonical concept.
        Fails with IdentityCollisionError if canonical_id already exists with different name.
        """
        with self._lock:
            cid = canonical_id.strip()
            if not cid:
                raise ValueError("canonical_id cannot be empty")

            if cid in self._concepts:
                existing = self._concepts[cid]
                # If name or domain differ, reject silent overwrite
                if existing.canonical_name != canonical_name:
                    raise IdentityCollisionError(
                        f"ID Collision: Concept '{cid}' already exists with name '{existing.canonical_name}', "
                        f"cannot overwrite with '{canonical_name}'."
                    )
                # Same concept: add any new aliases or attributes
                if aliases:
                    for a in aliases:
                        self.add_alias(cid, a)
                if attributes:
                    existing.update_attributes(attributes, source=source, reason="merge_registration")
                return existing

            concept = ConceptIdentity(
                canonical_id=cid,
                canonical_name=canonical_name,
                domain=domain,
                aliases=set(),
                attributes=dict(attributes or {}),
                source=source,
                metadata=dict(metadata or {})
            )

            # Register canonical name as an alias
            clean_name = canonical_name.strip().lower()
            self._concepts[cid] = concept
            self._alias_index[clean_name] = cid

            # Register provided aliases
            if aliases:
                for a in aliases:
                    self.add_alias(cid, a)

            return concept

    def get(self, canonical_id: str) -> ConceptIdentity:
        """Retrieve concept by exact canonical ID."""
        with self._lock:
            if canonical_id not in self._concepts:
                raise ConceptNotFoundError(f"Concept '{canonical_id}' not found in identity registry.")
            return self._concepts[canonical_id]

    def exists(self, canonical_id: str) -> bool:
        """Check if canonical ID exists."""
        with self._lock:
            return canonical_id in self._concepts

    def resolve(self, expression: str) -> Optional[ConceptIdentity]:
        """
        Resolves an expression or alias to its canonical concept.
        Handles direct canonical_id lookup, exact alias lookup, and case-insensitivity.
        """
        with self._lock:
            # 1. Exact canonical ID
            if expression in self._concepts:
                return self._concepts[expression]

            # 2. Alias lookup
            clean = expression.strip().lower()
            if clean in self._alias_index:
                cid = self._alias_index[clean]
                return self._concepts.get(cid)

            return None

    def add_alias(self, canonical_id: str, alias: str) -> bool:
        """
        Associates an alias with a canonical concept.
        Fails if the alias is already bound to a *different* canonical concept.
        """
        with self._lock:
            concept = self.get(canonical_id)
            clean_alias = alias.strip().lower()
            if not clean_alias:
                return False

            if clean_alias in self._alias_index:
                existing_cid = self._alias_index[clean_alias]
                if existing_cid != canonical_id:
                    raise IdentityCollisionError(
                        f"Alias conflict: Expression '{alias}' is already bound to concept '{existing_cid}'. "
                        f"Cannot bind to '{canonical_id}' without explicit merging."
                    )
                return False  # Already registered

            concept.add_alias(clean_alias)
            self._alias_index[clean_alias] = canonical_id
            return True

    def update_attributes(
        self,
        canonical_id: str,
        new_attributes: Dict[str, Any],
        source: str = "direct_observation",
        reason: str = "attribute_update"
    ) -> int:
        """Updates attributes on a concept, advancing its version."""
        with self._lock:
            concept = self.get(canonical_id)
            return concept.update_attributes(new_attributes, source=source, reason=reason)

    def count(self) -> int:
        with self._lock:
            return len(self._concepts)

    def list_all(self) -> List[ConceptIdentity]:
        with self._lock:
            return list(self._concepts.values())

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "concepts": {cid: c.to_dict() for cid, c in self._concepts.items()},
                "aliases": dict(self._alias_index)
            }

    @classmethod
    def from_dict(cls, data: dict) -> 'ConceptIdentityRegistry':
        registry = cls()
        concepts_data = data.get("concepts", {})
        for cid, c_data in concepts_data.items():
            registry._concepts[cid] = ConceptIdentity.from_dict(c_data)
        registry._alias_index = dict(data.get("aliases", {}))
        return registry
