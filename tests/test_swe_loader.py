"""Unit tests for SWE-smith message normalization and assistant token masking."""

from seed_agent_lab.swe_loader import assistant_token_mask, normalize_swe_messages


def test_normalize_swe_messages() -> None:
    raw = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": [{"type": "text", "text": "fix bug"}]},
        {
            "role": "assistant",
            "content": "checking",
            "tool_calls": [
                {"type": "function", "function": {"name": "bash", "arguments": '{"command": "ls"}'}}
            ],
        },
        {"role": "tool", "content": [{"type": "text", "text": "app.py"}]},
    ]
    out = normalize_swe_messages(raw)
    assert out[1]["content"] == "fix bug"
    assert out[2]["tool_calls"][0]["function"]["arguments"] == {"command": "ls"}
    assert out[3]["content"] == "app.py"


def test_assistant_token_mask() -> None:
    tokens = ["<|im_start|>", "user\n", "hi", "<|im_end|>\n", "<|im_start|>", "assistant\n", "hello", " world", "<|im_end|>"]
    mask = assistant_token_mask(tokens)
    assert mask == (False, False, False, False, False, False, True, True, False)
