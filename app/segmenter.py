from __future__ import annotations

import re
from typing import List

from app.schemas import ConversationUnit


def _clean_segments(parts: List[str]) -> List[str]:
    cleaned: List[str] = []
    for part in parts:
        value = re.sub(r"\s+", " ", part).strip()
        if value:
            cleaned.append(value)
    return cleaned


def _segment_by_lines(text: str) -> List[str]:
    return [line.strip() for line in text.split("\n") if line.strip()]


def _segment_by_sentences(text: str) -> List[str]:
    compact = re.sub(r"\s+", " ", text).strip()
    if not compact:
        return []

    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", compact)
    if len(parts) <= 1:
        parts = re.split(r"\s*;\s*", compact)
    return [p.strip() for p in parts if p.strip()]


def segment_conversation(text: str) -> List[ConversationUnit]:
    line_parts = _clean_segments(_segment_by_lines(text))
    if len(line_parts) >= 2:
        final_parts = line_parts
    else:
        final_parts = _clean_segments(_segment_by_sentences(text))

    if len(final_parts) < 2:
        # Keep a single segment as a valid fallback.
        final_parts = _clean_segments([text])

    if not final_parts:
        raise ValueError("Cannot segment empty conversation text.")

    return [
        ConversationUnit(id=f"U{idx}", text=segment)
        for idx, segment in enumerate(final_parts, start=1)
    ]
