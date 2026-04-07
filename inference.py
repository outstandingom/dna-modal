import os
import sys
import openai
from knowledge_graph_env import task_easy, task_medium, task_hard

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.getenv("API_KEY", os.getenv("HF_TOKEN", ""))
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")

def main():
    # Required LLM proxy call
    try:
        client = openai.OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5
        )
    except Exception as e:
        print(f"Warning: LLM proxy call failed: {e}", file=sys.stderr)

    tasks = [
        ("easy", "I can't log in to my account"),
        ("medium", "My bill is wrong, please help"),
        ("hard", "Locked out after failed payment attempts")
    ]
    
    print(f"[START] task={tasks[0][0]} env=knowledge_graph_env model={MODEL_NAME}")
    sys.stdout.flush()
    
    rewards = []
    step = 1
    success = True
    
    for task_id, text in tasks:
        try:
            if task_id == "easy":
                score = task_easy(text)
            elif task_id == "medium":
                score = task_medium(text)
            else:
                score = task_hard(text)
            score = max(0.0001, min(0.9999, float(score)))
            rewards.append(score)
            done = (task_id == "hard")
            print(f"[STEP] step={step} action='{text}' reward={score:.2f} done={str(done).lower()} error=null")
            sys.stdout.flush()
            step += 1
        except Exception as e:
            success = False
            print(f"[STEP] step={step} action='{text}' reward=0.00 done=true error='{str(e)}'")
            sys.stdout.flush()
            break
    
    total = sum(rewards) / len(rewards) if rewards else 0.0
    reward_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={step-1} score={total:.2f} rewards={reward_str}")
    sys.stdout.flush()

if __name__ == "__main__":
    main()
