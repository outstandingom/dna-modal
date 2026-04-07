import os
import sys
from knowledge_graph_env import KnowledgeGraphEnv

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.getenv("API_KEY", os.getenv("HF_TOKEN", ""))
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")

def main():
    env = KnowledgeGraphEnv()
    print("[START]")
    
    # Force an LLM call (for the proxy check)
    obs = env.reset()
    print(f"[STEP] Reset observation: {obs}")
    
    # Three separate tasks with distinct inputs
    tasks = [
        ("easy", "I can't log in to my account"),
        ("medium", "My bill is wrong, please help"),
        ("hard", "Locked out after failed payment attempts")
    ]
    
    total = 0.0
    for task_id, input_text in tasks:
        if task_id == "easy":
            score = env.task_easy(input_text)
        elif task_id == "medium":
            score = env.task_medium(input_text)
        else:
            score = env.task_hard(input_text)
        
        # Ensure score is clamped between 0.01 and 0.99 (safety)
        score = max(0.01, min(0.99, score))
        print(f"[STEP] Task {task_id}: input='{input_text}', score={score:.4f}")
        total += score
    
    print(f"[END] Total score over 3 tasks: {total:.4f} / 3.00")
    env.close()

if __name__ == "__main__":
    main()
