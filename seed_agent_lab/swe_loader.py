"""Pure normalization and assistant-span extraction helpers for SWE-smith trajectories."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence


SWE_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a bash command in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "str_replace_editor",
            "description": "View, create, or edit files in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "path": {"type": "string"},
                    "old_str": {"type": "string"},
                    "new_str": {"type": "string"},
                    "view_range": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["command", "path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit",
            "description": "Submit the final patch.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def _flatten_content(content: Any) -> str:
    """Convert string or Anthropic-style content block lists into plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [item.get("text", "") for item in content if isinstance(item, Mapping) and "text" in item]
        return "\n".join(parts)
    return str(content)


def _parse_arguments(raw_args: Any) -> dict[str, Any]:
    """Parse JSON-encoded tool arguments into a dictionary."""
    if isinstance(raw_args, Mapping):
        return dict(raw_args)
    if isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args)
            if isinstance(parsed, Mapping):
                return dict(parsed)
        except json.JSONDecodeError:
            return {"raw": raw_args}
    return {}


def normalize_swe_messages(raw_messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Convert SWE-smith trajectory messages into Qwen3.5 chat_template compatible dicts."""
    normalized: list[dict[str, Any]] = []
    for msg in raw_messages:
        role = msg.get("role", "")
        if role not in ("system", "user", "assistant", "tool"):
            continue
        item: dict[str, Any] = {"role": role, "content": _flatten_content(msg.get("content"))}
        if role == "assistant":
            thought = _flatten_content(msg.get("thought"))
            if thought and thought != item["content"]:
                item["reasoning_content"] = thought
            raw_tcs = msg.get("tool_calls") or []
            if raw_tcs:
                parsed_tcs = []
                for tc in raw_tcs:
                    fn = tc.get("function", {})
                    parsed_tcs.append(
                        {
                            "type": "function",
                            "function": {
                                "name": str(fn.get("name", "")),
                                "arguments": _parse_arguments(fn.get("arguments")),
                            },
                        }
                    )
                item["tool_calls"] = parsed_tcs
        normalized.append(item)
    return normalized


_ASSISTANT_BLOCK_RE = re.compile(r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>", re.DOTALL)


def assistant_token_mask(token_texts: Sequence[str]) -> tuple[bool, ...]:
    """Identify which decoded tokens lie inside assistant response turns."""
    full_text = "".join(token_texts)
    if not full_text:
        return tuple()

    char_mask = [False] * len(full_text)
    for match in _ASSISTANT_BLOCK_RE.finditer(full_text):
        for i in range(match.start(1), match.end(1)):
            char_mask[i] = True

    mask: list[bool] = []
    cursor = 0
    for tok in token_texts:
        end = cursor + len(tok)
        mask.append(any(char_mask[cursor:end]))
        cursor = end
    return tuple(mask)
