import os
from knowledge_graph_env import KnowledgeGraphEnv

# Required by pre-submission checklist
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
HF_TOKEN = os.getenv("HF_TOKEN", "")   # No default key

def main():
    env = KnowledgeGraphEnv()
    print("[START]")

    obs = env.reset()
    print(f"[STEP] Observation: {obs}")

    total_reward = 0.0
    step_count = 0
    max_steps = 3

    for step_idx in range(max_steps):
        state = env.state()
        step_name = state["step_name"]
        print(f"\n--- Step {step_idx+1}: {step_name.upper()} ---")

        # Simple heuristic agent
        if step_name == "identify":
            words = obs.lower().split()
            if "login" in words or "account" in words:
                action = "login issue"
            elif "billing" in words or "payment" in words:
                action = "billing"
            elif "slow" in words or "performance" in words:
                action = "slow performance"
            elif "crash" in words:
                action = "crash"
            elif "feature" in words or "request" in words:
                action = "feature request"
            else:
                action = "hardware failure"
        elif step_name == "relate":
            if "login" in obs.lower() or "account" in obs.lower():
                action = "account locked"
            elif "billing" in obs.lower() or "payment" in obs.lower():
                action = "refund"
            elif "slow" in obs.lower():
                action = "crash"
            elif "feature" in obs.lower():
                action = "enhancement"
            else:
                action = "battery issue"
        else:  # answer
            if "login" in obs.lower() or "account" in obs.lower():
                action = "reset password"
            elif "billing" in obs.lower() or "payment" in obs.lower():
                action = "process refund"
            elif "slow" in obs.lower():
                action = "optimize code"
            elif "feature" in obs.lower():
                action = "implement new feature"
            else:
                action = "replace battery"

        obs, reward, done, info = env.step(action)
        print(f"Action: {action}")
        print(f"Reward: {reward:.4f}")
        total_reward += reward
        step_count += 1

        if done:
            break

    print(f"\n[END] Total reward: {total_reward:.4f} after {step_count} steps")
    env.close()

if __name__ == "__main__":
    main()
