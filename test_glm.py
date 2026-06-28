import os
import requests
import json

# ==========================================
# 1. ADD YOUR HUGGINGFACE DETAILS HERE
# ==========================================
# Replace this with your HuggingFace Access Token (Starts with 'hf_...')
# You can get it from: https://huggingface.co/settings/tokens
HF_TOKEN = "hf_your_token_here"

# Replace this with your exact model URL. 
# If it's a Serverless Inference API, it looks like: https://api-inference.huggingface.co/models/THUDM/glm-4-9b-chat
# If you created a dedicated Inference Endpoint, paste that exact URL here.
API_URL = "https://api-inference.huggingface.co/models/OutstandingOm/glm-4-9b-chat-bucket"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

def query_glm(prompt_text):
    print(f"\nSending message to your HuggingFace GLM model...")
    
    # Payload format for standard HuggingFace text generation
    payload = {
        "inputs": prompt_text,
        "parameters": {
            "max_new_tokens": 150,
            "temperature": 0.7,
            "return_full_text": False
        }
    }
    
    response = requests.post(API_URL, headers=headers, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        # Extract the generated text
        if isinstance(result, list) and len(result) > 0 and 'generated_text' in result[0]:
            print(f"\n[GLM Output]: {result[0]['generated_text']}\n")
        else:
            print(f"\n[GLM Output]: {result}\n")
    else:
        print(f"\n[ERROR]: Failed to connect. Status code: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    print("=== HuggingFace GLM Terminal Tester ===")
    user_input = input("Type a message for GLM: ")
    query_glm(user_input)
