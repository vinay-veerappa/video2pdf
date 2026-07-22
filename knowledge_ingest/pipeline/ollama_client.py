"""Ollama client. Retry/backoff logic borrowed from the user's original script."""

import re
import json
import time
import requests
from typing import Optional


class OllamaClient:
    def __init__(self, cfg):
        self.cfg = cfg  # OllamaConfig

    def generate(
        self,
        prompt: str,
        model: str,
        system: str = "",
        temperature: float = 0.2,
        num_ctx: int = 16384,
        num_predict: int = 4096,
        json_mode: bool = False,
    ) -> str:
        url = f"{self.cfg.url}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_ctx": num_ctx,
                "num_predict": num_predict,
            },
        }
        # Ollama supports structured output; force valid JSON when asked.
        if json_mode:
            payload["format"] = "json"

        last_err: Optional[Exception] = None
        for attempt in range(self.cfg.max_retries):
            try:
                start = time.time()
                r = requests.post(url, json=payload, timeout=self.cfg.request_timeout)
                r.raise_for_status()
                elapsed = time.time() - start
                print(f"      ({model} {elapsed:.1f}s)", flush=True)
                return r.json()["response"]
            except Exception as e:
                last_err = e
                if attempt < self.cfg.max_retries - 1:
                    print(f"      retry {attempt+1}/{self.cfg.max_retries}: {str(e)[:80]}")
                    time.sleep(self.cfg.retry_backoff_s)
        raise last_err

    @staticmethod
    def parse_json(raw: str):
        """Strip fences/think-tags and parse. Prefers a top-level OBJECT over an
        inner array (bug fixed: inner 'inferred_fields' array hijacked the match)."""
        s = (raw or "").strip()
        s = re.sub(r'<think>.*?</think>', '', s, flags=re.DOTALL).strip()
        if not s:
            raise ValueError("empty response after stripping think-tags")

        if "```" in s:
            m = re.search(r'```(?:json)?\s*(.*?)```', s, flags=re.DOTALL)
            if m:
                s = m.group(1).strip()
            else:
                s = s.replace("```json", "").replace("```", "").strip()

        s2 = s.lstrip()
        if s2.startswith("{"):
            i, j = s.find("{"), s.rfind("}")
            if 0 <= i < j:
                try:
                    return json.loads(s[i:j + 1])
                except json.JSONDecodeError:
                    pass
        if s2.startswith("["):
            i, j = s.find("["), s.rfind("]")
            if 0 <= i < j:
                try:
                    parsed = json.loads(s[i:j + 1])
                    if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], dict):
                        return parsed[0]
                    return parsed
                except json.JSONDecodeError:
                    pass
        return json.loads(s)
