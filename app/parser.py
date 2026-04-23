from __future__ import annotations

from pathlib import Path


SUPPORTED_EXTENSIONS = {".txt", ".md"}


def load_conversation_text(path: Path) -> str:
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported input extension: {path.suffix}")
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    text = path.read_text(encoding="utf-8")
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip("\ufeff")
    normalized_lines = [line.rstrip() for line in text.split("\n")]
    normalized = "\n".join(normalized_lines).strip()

    if not normalized:
        raise ValueError("Input transcript is empty.")
    return normalized
