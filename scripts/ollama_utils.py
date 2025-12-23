import requests
import json

class OllamaClient:
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url

    def generate(self, model, prompt, stream=False):
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream
        }
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        if stream:
            return response
        else:
            return response.json()

    def chat(self, model, messages, stream=False):
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream
        }
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        if stream:
            return response
        else:
            return response.json()

if __name__ == "__main__":
    # Quick test
    client = OllamaClient()
    try:
        res = client.generate("llama3", "Say hi!")
        print(res.get("response"))
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
