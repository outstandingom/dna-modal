import os
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from graders import task_easy, task_medium, task_hard, TASKS, GRADERS

class KnowledgeGraphEnv:
    def __init__(self, start_trainer: bool = True):
        pass
    def reset(self) -> str:
        return "Dummy observation"
    def step(self, action: str):
        return "Dummy observation", 0.0, False, {}
    def state(self) -> dict:
        return {"dummy": True}
    def close(self):
        pass
    def task_easy(self, input_text: str) -> float:
        return task_easy(input_text)
    def task_medium(self, input_text: str) -> float:
        return task_medium(input_text)
    def task_hard(self, input_text: str) -> float:
        return task_hard(input_text)

app = FastAPI()

class ResetResponse(BaseModel):
    observation: str
class StepRequest(BaseModel):
    action: str
class StepResponse(BaseModel):
    observation: str
    reward: float
    done: bool
    info: dict
class StateResponse(BaseModel):
    state: dict
class TaskResponse(BaseModel):
    tasks: List[str]
class GradeRequest(BaseModel):
    task_id: str
    input_text: str
class GradeResponse(BaseModel):
    score: float

_env = KnowledgeGraphEnv()

@app.get("/ping")
async def ping():
    return {"status": "ok"}

@app.post("/reset", response_model=ResetResponse)
async def reset_endpoint():
    obs = _env.reset()
    return ResetResponse(observation=obs)

@app.post("/step", response_model=StepResponse)
async def step_endpoint(req: StepRequest):
    obs, reward, done, info = _env.step(req.action)
    return StepResponse(observation=obs, reward=reward, done=done, info=info)

@app.get("/state", response_model=StateResponse)
async def state_endpoint():
    return StateResponse(state=_env.state())

@app.get("/tasks", response_model=TaskResponse)
async def tasks_endpoint():
    return TaskResponse(tasks=TASKS)

@app.post("/grade", response_model=GradeResponse)
async def grade_endpoint(req: GradeRequest):
    if req.task_id not in GRADERS:
        raise HTTPException(status_code=404, detail="Task not found")
    score = GRADERS[req.task_id](req.input_text)
    return GradeResponse(score=score)

@app.on_event("shutdown")
def shutdown_event():
    _env.close()
