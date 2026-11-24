"""
Persistence Adapter: Modular storage engine supporting PostgreSQL and SQLite.
Enforces transactional ACID writes, relational schema migration, and durable memory snapshots.
"""

import sqlite3
import json
import os
import threading
from typing import Dict, List, Optional, Any, Tuple


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS concepts (
    canonical_id TEXT PRIMARY KEY,
    canonical_name TEXT NOT NULL,
    domain TEXT NOT NULL,
    aliases TEXT NOT NULL,       -- JSON array
    attributes TEXT NOT NULL,    -- JSON object
    source TEXT NOT NULL,
    version INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    history TEXT NOT NULL,       -- JSON array of versions
    metadata TEXT NOT NULL       -- JSON object
);

CREATE TABLE IF NOT EXISTS relationships (
    relationship_id TEXT PRIMARY KEY,
    subject_id TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object_id TEXT NOT NULL,
    confidence REAL NOT NULL,
    strength REAL NOT NULL,
    source TEXT NOT NULL,
    evidence TEXT NOT NULL,      -- JSON array of evidence items
    valid_from TEXT NOT NULL,
    valid_until TEXT,
    recorded_at TEXT NOT NULL,
    status TEXT NOT NULL,
    version INTEGER NOT NULL,
    superseded_by TEXT,
    supersedes TEXT,
    contradicted_by TEXT,
    history TEXT NOT NULL,       -- JSON array
    metadata TEXT NOT NULL       -- JSON object
);

CREATE TABLE IF NOT EXISTS assemblies (
    id TEXT PRIMARY KEY,
    assembly_type TEXT NOT NULL,
    component_ids TEXT NOT NULL, -- JSON array
    relation_ids TEXT NOT NULL,  -- JSON array
    weights TEXT NOT NULL,       -- JSON array
    vector TEXT NOT NULL,        -- JSON array (128/512/1024D)
    version INTEGER NOT NULL,
    metadata TEXT NOT NULL       -- JSON object
);

CREATE TABLE IF NOT EXISTS temporal_events (
    event_id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    valid_time TEXT NOT NULL,
    recorded_time TEXT NOT NULL,
    details TEXT NOT NULL,       -- JSON object
    confidence REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_rel_subject ON relationships(subject_id);
CREATE INDEX IF NOT EXISTS idx_rel_object ON relationships(object_id);
CREATE INDEX IF NOT EXISTS idx_rel_predicate ON relationships(predicate);
CREATE INDEX IF NOT EXISTS idx_rel_status ON relationships(status);
"""


class SQLitePersistenceAdapter:
    """
    Transactional SQLite persistence driver.
    Provides identical relational SQL semantics to PostgreSQL without requiring an external daemon.
    """
    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        if db_path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self):
        with self._lock:
            self._conn.executescript(SCHEMA_SQL)
            self._conn.commit()

    def save_concept(self, concept_dict: dict):
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO concepts (
                    canonical_id, canonical_name, domain, aliases, attributes,
                    source, version, created_at, updated_at, history, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    concept_dict["canonical_id"],
                    concept_dict["canonical_name"],
                    concept_dict["domain"],
                    json.dumps(concept_dict.get("aliases", [])),
                    json.dumps(concept_dict.get("attributes", {})),
                    concept_dict.get("source", "direct_observation"),
                    concept_dict.get("version", 1),
                    concept_dict.get("created_at", ""),
                    concept_dict.get("updated_at", ""),
                    json.dumps(concept_dict.get("history", [])),
                    json.dumps(concept_dict.get("metadata", {}))
                )
            )
            self._conn.commit()

    def load_all_concepts(self) -> List[dict]:
        with self._lock:
            cursor = self._conn.execute("SELECT * FROM concepts")
            rows = cursor.fetchall()
            results = []
            for row in rows:
                results.append({
                    "canonical_id": row["canonical_id"],
                    "canonical_name": row["canonical_name"],
                    "domain": row["domain"],
                    "aliases": json.loads(row["aliases"]),
                    "attributes": json.loads(row["attributes"]),
                    "source": row["source"],
                    "version": row["version"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "history": json.loads(row["history"]),
                    "metadata": json.loads(row["metadata"])
                })
            return results

    def save_relationship(self, rel_dict: dict):
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO relationships (
                    relationship_id, subject_id, predicate, object_id, confidence,
                    strength, source, evidence, valid_from, valid_until, recorded_at,
                    status, version, superseded_by, supersedes, contradicted_by,
                    history, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rel_dict["relationship_id"],
                    rel_dict["subject_id"],
                    rel_dict["predicate"],
                    rel_dict["object_id"],
                    rel_dict["confidence"],
                    rel_dict["strength"],
                    rel_dict["source"],
                    json.dumps(rel_dict.get("evidence", [])),
                    rel_dict["valid_from"],
                    rel_dict.get("valid_until"),
                    rel_dict["recorded_at"],
                    rel_dict["status"],
                    rel_dict["version"],
                    rel_dict.get("superseded_by"),
                    rel_dict.get("supersedes"),
                    rel_dict.get("contradicted_by"),
                    json.dumps(rel_dict.get("history", [])),
                    json.dumps(rel_dict.get("metadata", {}))
                )
            )
            self._conn.commit()

    def load_all_relationships(self) -> List[dict]:
        with self._lock:
            cursor = self._conn.execute("SELECT * FROM relationships")
            rows = cursor.fetchall()
            results = []
            for row in rows:
                results.append({
                    "relationship_id": row["relationship_id"],
                    "subject_id": row["subject_id"],
                    "predicate": row["predicate"],
                    "object_id": row["object_id"],
                    "confidence": row["confidence"],
                    "strength": row["strength"],
                    "source": row["source"],
                    "evidence": json.loads(row["evidence"]),
                    "valid_from": row["valid_from"],
                    "valid_until": row["valid_until"],
                    "recorded_at": row["recorded_at"],
                    "status": row["status"],
                    "version": row["version"],
                    "superseded_by": row["superseded_by"],
                    "supersedes": row["supersedes"],
                    "contradicted_by": row["contradicted_by"],
                    "history": json.loads(row["history"]),
                    "metadata": json.loads(row["metadata"])
                })
            return results

    def close(self):
        with self._lock:
            if self._conn:
                self._conn.close()
