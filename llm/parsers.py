from __future__ import annotations

import json

from neo4j_factual_chatbot.schemas.chat import TokenUsage


def get_response_text(response) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]))
        return "\n".join(parts).strip()
    return str(content)


def parse_first_json(text: str) -> dict:
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            candidate = "\n".join(lines[1:-1]).strip()

    start = candidate.find("{")
    if start == -1:
        raise json.JSONDecodeError("No JSON object found", candidate, 0)

    return json.JSONDecoder().raw_decode(candidate[start:])[0]


def get_token_usage(response) -> TokenUsage:
    def to_int(value) -> int:
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    def parse_usage(payload) -> TokenUsage | None:
        if not isinstance(payload, dict):
            return None

        for key in ("usage", "usage_metadata", "token_usage", "token_count"):
            nested = payload.get(key)
            if isinstance(nested, dict):
                usage = parse_usage(nested)
                if usage:
                    return usage

        prompt = payload.get("prompt_tokens", payload.get("input_tokens"))
        completion = payload.get("completion_tokens", payload.get("output_tokens"))
        total = payload.get("total_tokens")
        if prompt is None and completion is None and total is None:
            return None

        prompt_tokens = to_int(prompt)
        completion_tokens = to_int(completion)
        total_tokens = to_int(total if total is not None else prompt_tokens + completion_tokens)
        return TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

    return (
        parse_usage(getattr(response, "usage_metadata", None))
        or parse_usage(getattr(response, "response_metadata", None))
        or TokenUsage()
    )
