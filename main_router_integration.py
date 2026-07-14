"""
main_router_integration.py
───────────────────────────
Copy-paste snippets showing the MINIMAL changes needed to wire GraphRouter
into the existing main.py.  No existing logic is removed or replaced.

Search for each ── PATCH N ── marker in main.py and apply the diff.
"""

# ════════════════════════════════════════════════════════════════════════════
# PATCH 1 — Import  (top of main.py, after existing imports)
# ════════════════════════════════════════════════════════════════════════════
from graph_router import GraphRouter, RoutingResult   # ADD THIS LINE


# ════════════════════════════════════════════════════════════════════════════
# PATCH 2 — KnowledgeGraphEnv.__init__
#
# Find the line:   self._train_projector()
# Add AFTER it:
# ════════════════════════════════════════════════════════════════════════════
class KnowledgeGraphEnv_PATCH2:          # shown as a stub — apply to real class
    def __init__(self, start_trainer=True):
        # ... (existing init code unchanged) ...

        self._train_projector()          # ← existing line

        # ── GRAPH ROUTER (DNA attention mechanism) ───────────────────────
        self.graph_router = GraphRouter(dims=DIMS)
        self.graph_router.bind_letter_vec(self.letter_vec)
        self._last_routing: dict = {}    # stores {session_id: RoutingResult}
        # ────────────────────────────────────────────────────────────────


# ════════════════════════════════════════════════════════════════════════════
# PATCH 3 — routed_reasoning helper on KnowledgeGraphEnv
#
# Add this method to KnowledgeGraphEnv (after extract_and_learn):
# ════════════════════════════════════════════════════════════════════════════
class KnowledgeGraphEnv_PATCH3:
    def routed_reasoning(
        self,
        query_text: str,
        session_id: str = "",
        query_vector=None,
        max_hops: int = 3,
        color_filter=None,
    ) -> dict:
        """
        Router-guided multi-hop reasoning.

        Replaces bare calls to self.reasoning_engine.multi_hop_reasoning()
        in the /agent endpoint with a version that:
          1. Uses GraphRouter to select the best entry nodes
          2. Seeds multi-hop reasoning from those entry nodes
          3. Merges and normalises the combined activation map
          4. Stores the RoutingResult for later reinforce() call

        Returns merged activation dict  {concept_name: score}
        """
        activation, routing = self.graph_router.routed_reasoning(
            query_text=query_text,
            concept_memory=self.concept_memory,
            reasoning_engine=self.reasoning_engine,
            query_vector=query_vector,
            max_hops=max_hops,
            color_filter=color_filter,
        )
        # Cache routing result keyed by session for reinforce() later
        key = session_id or "__global__"
        self._last_routing[key] = routing
        return activation

    def reinforce_router(self, reward: float, session_id: str = "") -> None:
        """
        Call after grading to propagate reward back into routing policy.

        Example (inside /agent endpoint, after calling a grader):
            score = GRADERS["task_medium"](response_text)
            _api_env.reinforce_router(score, session_id=req.session_id or "")
        """
        key = session_id or "__global__"
        routing = self._last_routing.get(key)
        if routing is not None:
            self.graph_router.reinforce(routing, reward, self.concept_memory)


# ════════════════════════════════════════════════════════════════════════════
# PATCH 4 — /agent endpoint
#
# Inside the existing independent_fallback() function, replace the bare
# multi_hop_reasoning calls with routed_reasoning:
# ════════════════════════════════════════════════════════════════════════════
#
# BEFORE (inside independent_fallback):
#     for concept_name in learned_concepts:
#         hop_res = _api_env.reasoning_engine.multi_hop_reasoning(concept_name, max_hops=2)
#
# AFTER:
#     for concept_name in learned_concepts:
#         # Use router-guided reasoning for better entry-point selection
#         hop_res = _api_env.routed_reasoning(
#             concept_name,
#             session_id=req.session_id or "",
#             max_hops=2,
#         )
#
# Then after grading (e.g. after score = GRADERS["task_medium"](response)):
#     _api_env.reinforce_router(score, session_id=req.session_id or "")
#
# ════════════════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════════════
# PATCH 5 — New FastAPI endpoints
#
# Add these two endpoints to the FastAPI app in main.py:
# ════════════════════════════════════════════════════════════════════════════
from fastapi import HTTPException                  # already imported in main.py
from pydantic import BaseModel                     # already imported in main.py
from typing import Optional                        # already imported in main.py


class RouterReinforceRequest(BaseModel):
    reward: float
    session_id: Optional[str] = None


# ── /router/status  (GET) ───────────────────────────────────────────────────
def router_status_endpoint():
    """
    @app.get("/router/status")
    Returns GraphRouter learning statistics and top routing preferences.
    """
    if _api_env is None:
        raise HTTPException(status_code=503, detail="Environment still initializing.")
    return _api_env.graph_router.status()


# ── /router/reinforce  (POST) ────────────────────────────────────────────────
def router_reinforce_endpoint(req: RouterReinforceRequest):
    """
    @app.post("/router/reinforce")
    Manually propagate a reward signal into the routing policy.
    Useful for explicit human feedback or external reward sources.
    """
    if _api_env is None:
        raise HTTPException(status_code=503, detail="Environment still initializing.")
    _api_env.reinforce_router(req.reward, session_id=req.session_id or "")
    return {"status": "reinforced", "reward": req.reward}


# ── Register the endpoints (add to FastAPI app) ─────────────────────────────
#
# app.add_api_route("/router/status",    router_status_endpoint,    methods=["GET"])
# app.add_api_route("/router/reinforce", router_reinforce_endpoint, methods=["POST"])
#
# Or equivalently as decorators:
#
# @app.get("/router/status")
# async def router_status():
#     if _api_env is None:
#         raise HTTPException(status_code=503)
#     return _api_env.graph_router.status()
#
# @app.post("/router/reinforce")
# async def router_reinforce(req: RouterReinforceRequest):
#     if _api_env is None:
#         raise HTTPException(status_code=503)
#     _api_env.reinforce_router(req.reward, session_id=req.session_id or "")
#     return {"status": "reinforced", "reward": req.reward}
