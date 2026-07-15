"""
LLM Service - Wraps Ollama local inference.
Modular so both agents share one connection layer (no duplication).
"""
import requests


class LLMService:
    """Handles all HTTP communication with the local Ollama server."""

    def __init__(self, model: str = "llama3.2", base_url: str = "http://localhost:11434"):
        self.model = model
        self.api_url = f"{base_url}/api/generate"
        self.tags_url = f"{base_url}/api/tags"

    def generate(self, prompt: str, max_tokens: int = 300) -> str:
        """Send prompt to Ollama, return response text. Graceful on all errors."""
        try:
            res = requests.post(
                self.api_url,
                json={"model": self.model, "prompt": prompt,
                      "stream": False, "options": {"num_predict": max_tokens, "temperature": 0.8}},
                timeout=90
            )
            res.raise_for_status()
            return res.json().get("response", "").strip()
        except requests.exceptions.ConnectionError:
            return "[ERROR] Ollama is offline. Run: ollama serve"
        except requests.exceptions.Timeout:
            return "[ERROR] LLM timeout — model may still be loading."
        except Exception as e:
            return f"[ERROR] {e}"

    def is_available(self) -> bool:
        """Quick health-check before starting a debate."""
        try:
            requests.get(self.tags_url, timeout=4)
            return True
        except Exception:
            return False
