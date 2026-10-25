"""LLM judge for answer faithfulness.

The judge runs on DeepSeek, a different model family from the GPT-OSS model
that generates the answers. A judge from the generator's own family tends to
prefer its own output, which inflates faithfulness.
"""

from __future__ import annotations

import hashlib
import json
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

SYSTEM_PROMPT = (
    "You are a strict evaluator. You return JSON only. "
    "Judge only what the supplied context supports."
)

USER_PROMPT = """Context passages:
{context}

Question: {question}
Answer: {answer}

Break the answer into atomic factual claims. For each claim, decide whether the
context passages support it. A claim not supported by the context is
unsupported, even if it is true in general. Do not reward fluency.

Return JSON with exactly this shape:
{{"claims": [{{"claim": "...", "supported": true, "evidence": "..."}}]}}
"""


class JudgeError(RuntimeError):
    """The judge returned output that could not be parsed."""


@dataclass(frozen=True)
class JudgeConfig:
    api_key: str
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"
    timeout: int = 60


class DeepSeekJudge:
    def __init__(
        self,
        config: JudgeConfig,
        cache_dir: str | Path | None = None,
        transport: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self.config = config
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._transport = transport or self._post
        self._lock = threading.Lock()

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.config.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def _cache_key(self, question: str, answer: str, context: str) -> str:
        blob = json.dumps([self.config.model, question, answer, context])
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _read_cache(self, key: str) -> dict[str, Any] | None:
        if self.cache_dir is None:
            return None
        path = self.cache_dir / f"{key}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_cache(self, key: str, value: dict[str, Any]) -> None:
        if self.cache_dir is None:
            return
        with self._lock:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            (self.cache_dir / f"{key}.json").write_text(
                json.dumps(value, indent=2) + "\n", encoding="utf-8"
            )

    def _payload(self, question: str, answer: str, context: str) -> dict[str, Any]:
        return {
            "model": self.config.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": USER_PROMPT.format(
                        context=context, question=question, answer=answer
                    ),
                },
            ],
        }

    def _parse(self, raw: dict[str, Any]) -> list[dict[str, Any]]:
        try:
            content = raw["choices"][0]["message"]["content"]
            claims = json.loads(content)["claims"]
            if not isinstance(claims, list):
                raise TypeError("claims is not a list")
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise JudgeError(f"judge returned unparseable output: {raw!r}") from exc
        return claims

    def score_faithfulness(
        self, question: str, answer: str, context: str
    ) -> dict[str, Any]:
        key = self._cache_key(question, answer, context)
        cached = self._read_cache(key)
        if cached is not None:
            return cached

        raw = self._transport(self._payload(question, answer, context))
        claims = self._parse(raw)
        total = len(claims)
        supported = sum(1 for claim in claims if claim.get("supported") is True)
        result = {
            "claims": claims,
            "supported": supported,
            "total": total,
            "score": round(supported / total, 4) if total else 1.0,
            "model": raw.get("model", self.config.model),
        }
        self._write_cache(key, result)
        return result
