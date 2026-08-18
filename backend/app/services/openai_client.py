from __future__ import annotations

import json
from time import perf_counter
from typing import Any

try:
    import truststore

    truststore.inject_into_ssl()
except Exception:
    pass

from openai import APIConnectionError, AuthenticationError, OpenAI

from app.core.config import settings


class OpenAIService:
    def __init__(self) -> None:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Add it to backend/.env first.")
        self.client = OpenAI(api_key=settings.openai_api_key, timeout=60.0, max_retries=2)

    @staticmethod
    def _handle_openai_error(exc: Exception) -> None:
        if isinstance(exc, AuthenticationError):
            raise RuntimeError("OpenAI rejected the API key. Update OPENAI_API_KEY in backend/.env and restart the backend.") from exc
        if isinstance(exc, APIConnectionError):
            raise RuntimeError("Could not connect to OpenAI. Check your network connection and local certificate settings.") from exc
        raise exc

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self.embed_texts_with_metadata(texts)["embeddings"]

    def embed_texts_with_metadata(self, texts: list[str]) -> dict[str, Any]:
        start = perf_counter()
        try:
            response = self.client.embeddings.create(
                model=settings.openai_embedding_model,
                input=texts,
            )
        except Exception as exc:
            self._handle_openai_error(exc)
        return {
            "embeddings": [item.embedding for item in response.data],
            "model": getattr(response, "model", settings.openai_embedding_model),
            "duration_ms": round((perf_counter() - start) * 1000),
        }

    def chat(self, messages: list[dict[str, str]], model: str | None = None, temperature: float = 0.1) -> str:
        return self.chat_with_metadata(messages, model=model, temperature=temperature)["content"]

    def chat_with_metadata(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.1,
        response_format: dict[str, str] | None = None,
        max_tokens: int = 700,
    ) -> dict[str, Any]:
        start = perf_counter()
        request_args: dict[str, Any] = {
            "model": model or settings.openai_generation_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            request_args["response_format"] = response_format
        try:
            response = self.client.chat.completions.create(**request_args)
        except Exception as exc:
            self._handle_openai_error(exc)
        usage = response.usage
        return {
            "content": response.choices[0].message.content.strip(),
            "model": response.model,
            "duration_ms": round((perf_counter() - start) * 1000),
            "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
            "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
            "total_tokens": getattr(usage, "total_tokens", None) if usage else None,
        }

    def judge_json(self, prompt: str) -> dict[str, Any]:
        return self.judge_json_with_metadata(prompt)["judge"]

    def judge_json_with_metadata(self, prompt: str) -> dict[str, Any]:
        result = self.chat_with_metadata(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict but calibrated RAG evaluation judge. "
                        "Return valid JSON only. Reward usable, well-grounded answers, but penalise material omissions, "
                        "scope drift, and unsupported claims."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            model=settings.openai_judge_model or settings.openai_generation_model,
            temperature=0.0,
            response_format={"type": "json_object"},
            max_tokens=1500,
        )
        content = result["content"]
        try:
            judge = json.loads(content)
        except json.JSONDecodeError:
            judge = {"parse_error": True, "raw": content}
        return {"judge": judge, **result}
