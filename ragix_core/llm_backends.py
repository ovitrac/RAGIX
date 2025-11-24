"""
LLM Backend Abstraction for RAGIX

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-24
"""

from typing import List, Dict
import requests


class OllamaLLM:
    """
    Simple wrapper around Ollama's /api/chat endpoint.

    Assumes Ollama is running locally:
        ollama serve

    The model is specified at initialization.
    """

    def __init__(self, model: str, base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def generate(self, system_prompt: str, history: List[Dict[str, str]]) -> str:
        """
        Call Ollama's chat API with:
            - a system prompt
            - a history of messages (user/assistant)

        Returns the assistant's message as plain text.
        """
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }

        r = requests.post(f"{self.base_url}/api/chat", json=payload)
        r.raise_for_status()
        data = r.json()
        return data["message"]["content"].strip()
