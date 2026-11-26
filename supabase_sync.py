"""Durable Supabase PostgreSQL persistence for DNA Modal Knowledge Graph.

Reads WA_SUPABASE_URL / WA_SUPABASE_SECRET_KEY (or SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY)
from environment variables. Degrades gracefully to local in-memory mode if unset.
"""

import os
import re
import logging
import threading
from datetime import datetime, timezone
from typing import Optional

import httpx

logger = logging.getLogger("dna_modal_supabase")

_CLIENT: Optional[httpx.Client] = None
_LOCK = threading.Lock()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(val: str) -> str:
    clean = re.sub(r"[^a-z0-9]+", "_", str(val or "").strip().lower()).strip("_")
    return clean[:64] or "node"


def _get_config() -> tuple[str, str]:
    url = (os.getenv("WA_SUPABASE_URL") or os.getenv("SUPABASE_URL") or "").strip().rstrip("/")
    key = (
        os.getenv("WA_SUPABASE_SECRET_KEY")
        or os.getenv("WA_SUPABASE_SERVICE_ROLE_KEY")
        or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        or os.getenv("WA_SUPABASE_PUBLISHABLE_KEY")
        or ""
    ).strip()
    return url, key


def _get_client() -> httpx.Client:
    global _CLIENT
    with _LOCK:
        if _CLIENT is None:
            _CLIENT = httpx.Client(timeout=8.0)
        return _CLIENT


def _headers(key: str, prefer: Optional[str] = None) -> dict[str, str]:
    h = {"apikey": key, "Content-Type": "application/json"}
    if key.startswith("eyJ"):
        h["Authorization"] = f"Bearer {key}"
    if prefer:
        h["Prefer"] = prefer
    return h


def hydrate_graph_from_supabase(env) -> int:
    """Restore persisted relationships from Supabase wa_graph_edges into KnowledgeGraphEnv."""
    url, key = _get_config()
    if not url or not key or env is None:
        return 0
    try:
        client = _get_client()
        resp = client.get(
            f"{url}/rest/v1/wa_graph_edges",
            headers=_headers(key),
            params={"select": "source_name,target_name,relation", "limit": 1000},
        )
        if resp.status_code != 200:
            logger.warning(f"[SupabaseSync] Hydrate skipped HTTP {resp.status_code}")
            return 0
        rows = resp.json() if isinstance(resp.json(), list) else []
        restored = 0
        for row in rows:
            src = str(row.get("source_name") or "").strip()
            tgt = str(row.get("target_name") or "").strip()
            rel = str(row.get("relation") or "RELATED").strip()
            if not src or not tgt:
                continue
            for c in (src, tgt):
                if c.lower() not in env.concept_memory.concepts:
                    pf = [env.feature_registry.register(c)]
                    sf = [env.feature_registry.register(c)]
                    env.concept_memory.register(c, pf, sf)
            env.concept_memory.add_weighted_relationship(src, tgt, weight=1.0, color=rel)
            restored += 1
        logger.info(f"[SupabaseSync] Hydrated {restored} edges from Supabase.")
        return restored
    except Exception as e:
        logger.warning(f"[SupabaseSync] Hydrate error: {e}")
        return 0


def persist_relationship_async(concept_a: str, concept_b: str, relation: str = "RELATED", tenant_id: str = "default") -> None:
    """Persist a relationship triple to Supabase wa_graph_nodes + wa_graph_edges without blocking /relationships/add."""
    url, key = _get_config()
    if not url or not key or not concept_a or not concept_b:
        return

    def _worker():
        try:
            client = _get_client()
            now = _now_iso()
            tid = str(tenant_id or "default").strip() or "default"
            src_key = f"{tid}:concept:{_slug(concept_a)}"
            tgt_key = f"{tid}:concept:{_slug(concept_b)}"
            nodes = [
                {"node_key": src_key, "tenant_id": tid, "label": "Concept", "name": concept_a[:120], "meta": {}, "updated_at": now},
                {"node_key": tgt_key, "tenant_id": tid, "label": "Concept", "name": concept_b[:120], "meta": {}, "updated_at": now},
            ]
            edges = [
                {
                    "edge_key": f"{src_key}|{relation}|{tgt_key}",
                    "tenant_id": tid,
                    "source_key": src_key,
                    "source_name": concept_a[:120],
                    "relation": str(relation or "RELATED")[:40],
                    "target_key": tgt_key,
                    "target_name": concept_b[:120],
                    "target_label": "Concept",
                    "updated_at": now,
                }
            ]
            hdrs = _headers(key, prefer="resolution=merge-duplicates,return=minimal")
            client.post(f"{url}/rest/v1/wa_graph_nodes", headers=hdrs, json=nodes)
            client.post(f"{url}/rest/v1/wa_graph_edges", headers=hdrs, json=edges)
        except Exception as e:
            logger.debug(f"[SupabaseSync] Persist error: {e}")

    threading.Thread(target=_worker, daemon=True).start()
