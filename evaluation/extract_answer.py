"""Answer extraction used for all reported TA-Acc scores.

Identical to `extract_answer_legacy` in run_eval_vllm_offline.py except that the rule
"keep only the first word of a text answer longer than five words" is removed, so that
every model is scored with the same extraction (no torch/vLLM dependency).
"""
from __future__ import annotations

import json
import re
from typing import Optional

_ALIASES = {
    "numeric": "number",
    "scalar_int": "number",
    "bool": "boolean",
    "dict": "map",
    "list_str": "set",
    "set_str": "set",
    "edge_set": "set",
}

_PREFIXES = [
    r"^analysis\s*", r"^we need to\s*", r"^we need\s*", r"^based on\s*", r"^the answer is\s*",
    r"^answer:\s*", r"^result:\s*", r"^\**answer\**:\s*", r"^json\s*", r"^set\s*", r"^map\s*",
    r"^text\s*", r"^device:\s*", r"^router:\s*", r"^hostname:\s*",
]


def normalize_answer_type(answer_type: Optional[str]) -> str:
    normalized = str(answer_type or "text").strip().lower()
    return _ALIASES.get(normalized, normalized)


def extract_answer(raw_text: str, answer_type: Optional[str]) -> str:
    """Recover the final answer from a raw model output."""
    atype = normalize_answer_type(answer_type)
    text = (raw_text or "").strip()
    if not text:
        return ""

    # Drop reasoning blocks (<think>...</think>, GPT-OSS "analysis ... assistantfinal").
    if "</think>" in text:
        after = text.split("</think>")[-1].strip()
    elif "assistantfinal" in text:
        after = text.split("assistantfinal")[-1].strip()
        if after.startswith("think"):
            after = after[5:].strip()
    elif "<think>" in text:
        after = text.split("<think>")[0].strip()
    else:
        after = text

    cleaned = after
    for _ in range(3):
        cleaned = re.sub(r"<([^>]+)>", r"\1", cleaned).strip()
    for prefix in _PREFIXES:
        cleaned = re.sub(prefix, "", cleaned, flags=re.IGNORECASE)

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if not lines and cleaned.strip():
        lines = [cleaned.strip()]
    if not lines:
        return ""
    first_line = lines[0]

    if first_line.lower().startswith("we ") and len(first_line) > 50:
        if atype == "number":
            m = re.search(r"\b(-?\d+\.?\d*)\b", first_line)
            if m:
                first_line = m.group(1)
        elif atype == "text":
            m = re.search(r"\bof\s+([A-Za-z0-9_.:/-]+)", first_line)
            if m:
                first_line = m.group(1)

    if atype == "set":
        m = re.search(r"\[[\s\S]*?\]", cleaned)
        if m:
            try:
                return json.dumps(json.loads(m.group(0)), ensure_ascii=False)
            except Exception:
                pass
        return "[]"

    if atype in {"map", "map_str_int", "map_str_str"}:
        m = re.search(r"\{[\s\S]*?\}", cleaned)
        if m:
            try:
                return json.dumps(json.loads(m.group(0)), ensure_ascii=False)
            except Exception:
                pass
        return "{}"

    if atype == "number":
        m = re.search(r"-?\d+\.?\d*", cleaned)
        return m.group(0) if m else "0"

    if atype == "boolean":
        lower = cleaned.lower()
        if "true" in lower or "yes" in lower:
            return "true"
        return "false"

    answer = first_line.strip("\"'")
    if answer.lower() == "login local":
        answer = "local"
    return answer
