"""
Shared helpers for handling HuggingFace text-generation output:
normalising the generated text and robustly parsing the JSON the model returns
(including salvaging output truncated at the token limit).
"""
from __future__ import annotations

import json
import re

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def extract_generated_text(output) -> str:
    """
    Normalise a HuggingFace text-generation pipeline result into the
    assistant's reply string.

    When the pipeline is given chat messages, `generated_text` is the FULL
    conversation as a list of {"role", "content"} dicts (input + reply), not a
    string. We return only the last assistant message's content. Also handles
    the plain-string case (non-chat pipelines) and the extra list nesting that
    batched calls add.
    """
    # Batched calls wrap each result in a single-element list.
    item = output[0] if isinstance(output, list) and output and isinstance(output[0], dict) else output
    generated = item["generated_text"] if isinstance(item, dict) else item

    if isinstance(generated, str):
        return generated
    if isinstance(generated, list):  # list of message dicts
        for msg in reversed(generated):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                return msg.get("content", "")
        if generated and isinstance(generated[-1], dict):
            return generated[-1].get("content", "")
    return ""


def parse_json_block(text: str) -> dict | None:
    """
    Extract the JSON object from a model reply. Handles markdown fences and
    surrounding prose. Uses the FIRST "{" and LAST "}" so the whole object
    (with nested objects) is captured. Returns None when nothing parses.
    """
    fenced = _FENCE_RE.search(text)
    if fenced:
        text = fenced.group(1)

    start = text.find("{")
    if start == -1:
        return None
    body = text[start:]

    # Fast path: the model closed the JSON properly.
    end = body.rfind("}") + 1
    if end:
        try:
            data = json.loads(body[:end])
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

    # Salvage path: output was truncated (hit max_new_tokens mid-JSON). Pull the
    # COMPLETE objects out of each array and drop the half-written trailing one,
    # so a rich chunk still yields the entities/relationships that came through.
    salvaged = {
        "entities": _complete_objects_after(body, "entities"),
        "relationships": _complete_objects_after(body, "relationships"),
    }
    if salvaged["entities"] or salvaged["relationships"]:
        return salvaged
    return None


def _complete_objects_after(s: str, key: str) -> list:
    """From `"key": [ ... `, return the fully-closed {...} objects, stopping at
    the first object that was cut off by truncation."""
    m = re.search(r'"' + key + r'"\s*:\s*\[', s)
    if not m:
        return []
    i, n, objs = m.end(), len(s), []
    while i < n:
        while i < n and s[i] in " \t\r\n,":
            i += 1
        if i >= n or s[i] != "{":
            break  # ']' end of array, or truncated before an object started
        depth, in_str, esc, j, complete = 0, False, False, i, False
        while j < n:
            ch = s[j]
            if in_str:
                esc = (ch == "\\" and not esc)
                if ch == '"' and not esc:
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    j += 1
                    complete = True
                    break
            j += 1
        if not complete:
            break  # this object was truncated — stop here
        try:
            objs.append(json.loads(s[i:j]))
        except json.JSONDecodeError:
            break
        i = j
    return objs
