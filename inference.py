import os
from knowledge_graph_env import KnowledgeGraphEnv

# Required by validator – use their injected variables if present
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.getenv("API_KEY", os.getenv("HF_TOKEN", ""))
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")

def main():
    env = KnowledgeGraphEnv()
    print("[START]")
    
    # Test inputs for the three tasks
    test_inputs = {
        "easy": "I can't log in to my account",
        "medium": "My bill is wrong, please help",
        "hard": "Locked out after failed payment attempts"
    }
    
    total_score = 0.0
    for task_id, input_text in test_inputs.items():
        if task_id == "easy":
            score = env.task_easy(input_text)
        elif task_id == "medium":
            score = env.task_medium(input_text)
        else:
            score = env.task_hard(input_text)
        
        print(f"[STEP] Task {task_id}: input='{input_text}', score={score:.4f}")
        total_score += score
    
    print(f"[END] Total score over 3 tasks: {total_score:.4f} / 3.00")
    env.close()

if __name__ == "__main__":
    main()
