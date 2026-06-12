from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from app.schemas import (
    ConversationUnit,
    ConstraintItem,
    NoteItem,
    QuestionItem,
    RequirementItem,
    SpecOutput,
)
from app.utils import model_dump_compat, model_validate_compat, normalize_text


class ExtractionError(RuntimeError):
    pass


@dataclass
class ExtractionMeta:
    json_parse_ok: bool = False
    pydantic_validation_ok: bool = False
    used_repair: bool = False
    repaired_output: str | None = None
    parse_error: str | None = None
    validation_error: str | None = None
    semantic_warnings: list[str] = field(default_factory=list)


SOURCE_UNIT_DECISION_TYPES = {
    "functional_requirement",
    "non_functional_requirement",
    "constraint",
    "open_question",
    "note",
    "discard",
}

FUTURE_SCOPE_HINTS = (
    "later",
    "future",
    "phase 2",
    "phase two",
    "not now",
    "eventually",
    "someday",
    "maybe",
    "might",
    "optional",
)

VAGUE_HINTS = (
    "clean",
    "modern",
    "calm",
    "clear",
    "professional",
    "reassuring",
    "trustworthy",
    "simple",
    "easy",
    "intuitive",
    "user friendly",
)

HARD_BOUNDARY_HINTS = (
    "not in version one",
    "not in v1",
    "not part of the first release",
    "not needed for launch",
    "not required for launch",
    "not needed for the launch",
    "not required for the launch",
    "not part of launch",
    "not part of the launch",
    "not be part of launch",
    "not be part of the launch",
    "not be part of first release",
    "not be part of the first release",
    "not in this release",
    "outside initial release",
    "initial release only",
    "web only",
    "no app",
    "within 4 weeks",
    "within four weeks",
    "limited budget",
    "budget is limited",
    "only staff",
    "only admin",
    "only three",
    "only two",
    "only pilot",
    "first launch",
    "first release",
    "future semester",
    "next semester",
    "later phase",
    "phase two",
    "deferred to a later release",
    "shall be deferred",
)

STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "for",
    "in",
    "on",
    "with",
    "be",
    "is",
    "are",
    "it",
    "this",
    "that",
    "should",
    "shall",
    "must",
    "can",
    "could",
    "will",
    "would",
    "from",
    "by",
    "as",
    "at",
}

QUALITY_HINTS = (
    "fast",
    "faster",
    "quick",
    "quickly",
    "performance",
    "responsive",
    "responsiveness",
    "simple",
    "intuitive",
    "mobile",
    "mobile first",
    "mobile-first",
    "tablet",
    "tablets",
    "tablet screen",
    "tablet screens",
    "phone",
    "phones",
    "android",
    "iphone",
    "ios",
    "usability",
    "easy to use",
    "user friendly",
    "accessible",
    "accessibility",
    "secure",
    "security",
    "reliable",
    "reliability",
    "availability",
    "uptime",
    "load quickly",
    "load fast",
    "clean",
    "calm",
    "clear",
    "professional",
    "reassuring",
    "trustworthy",
    "modern",
    "style",
    "design",
)

VAGUE_QUALITY_HINTS = (
    "clean",
    "calm",
    "clear",
    "professional",
    "reassuring",
    "trustworthy",
    "modern",
    "easy to use",
    "user friendly",
    "intuitive",
    "simple",
    "style",
    "design should feel",
    "interface should feel",
)

QUALITY_MEASUREMENT_HINTS = (
    "ms",
    "millisecond",
    "seconds",
    "sec",
    "latency",
    "uptime",
    "wcag",
    "%",
    "p95",
    "p99",
    "lcp",
    "cls",
    "tti",
)

CAPABILITY_HINTS = (
    "allow",
    "allows",
    "able to",
    "can ",
    "rsvp",
    "reserve",
    "book",
    "create",
    "set",
    "update",
    "delete",
    "edit",
    "manage",
    "track",
    "record",
    "submit",
    "upload",
    "assign",
    "access",
    "review",
    "view",
    "see",
    "find",
    "search",
    "filter",
    "browse",
    "cancel",
    "mark",
    "block",
    "approve",
    "reschedule",
    "request",
    "notify",
    "replace",
    "connect",
    "leave",
    "send",
    "receive",
    "export",
    "import",
    "login",
    "log in",
    "sign up",
    "register",
    "pay",
    "checkout",
)

FEW_SHOT_CONTAMINATION_HINTS = (
    "u_ex",
    "ex_",
    "array<string>",
    "array string",
    "future or contextual note",
    "no additional notes are present",
    "specific unresolved ambiguity",
    "specific developer question",
    "specific developers asking",
    "is there any ambiguity",
    "about the functionality",
    "additional features would improve",
    "what should we do next",
    "is this a future feature",
    "this project involves integrating",
    "we aim to make",
    "already started implementing",
    "conduct user testing",
    "gather feedback",
    "exact or near-exact source wording",
    "given ..., when ..., then",
    "source sentence describing",
    "summary of the real conversation only",
    "observable result occurs",
    "operating condition",
    "event happens",
    "measurable or observable quality result",
    "<actor>",
    "<action>",
    "<source",
    "<summary",
    "<future",
)


def strip_reasoning_blocks(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"^\s*<think>.*?(?=\{)", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


def _strip_code_fence(text: str) -> str:
    text = strip_reasoning_blocks(text)
    fenced = re.match(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", text, flags=re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    return text.strip()


def _extract_first_json_object(text: str) -> str | None:
    text = strip_reasoning_blocks(text)
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def lightweight_repair_json(raw_text: str) -> str:
    repaired = strip_reasoning_blocks(raw_text)
    repaired = _strip_code_fence(repaired)
    repaired = repaired.replace("\u201c", '"').replace("\u201d", '"')
    repaired = repaired.replace("\u2018", "'").replace("\u2019", "'")
    repaired = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", repaired)

    start_idx = repaired.find("{")
    end_idx = repaired.rfind("}")
    if start_idx >= 0 and end_idx > start_idx:
        repaired = repaired[start_idx : end_idx + 1]

    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)

    brace_diff = repaired.count("{") - repaired.count("}")
    bracket_diff = repaired.count("[") - repaired.count("]")
    if bracket_diff > 0 and brace_diff == 0 and repaired.rstrip().endswith("}"):
        repaired = repaired.rstrip()
        repaired = repaired[:-1] + ("]" * bracket_diff) + "}"
    elif bracket_diff > 0:
        repaired = repaired + ("]" * bracket_diff)
    if brace_diff > 0:
        repaired = repaired + ("}" * brace_diff)
    return repaired.strip()


def _build_candidates(raw_output: str) -> list[str]:
    candidates: list[str] = []
    raw = strip_reasoning_blocks(raw_output)
    if raw:
        candidates.append(raw)
    no_fence = _strip_code_fence(raw_output)
    if no_fence and no_fence not in candidates:
        candidates.append(no_fence)
    extracted = _extract_first_json_object(raw_output)
    if extracted and extracted not in candidates:
        candidates.append(extracted)
    return candidates


def parse_json_object_safe(
    raw_output: str,
    *,
    allow_repair: bool = True,
) -> tuple[dict[str, Any] | None, ExtractionMeta]:
    meta = ExtractionMeta()
    candidates = _build_candidates(raw_output)
    errors: list[str] = []

    for idx, candidate in enumerate(candidates, start=1):
        try:
            parsed = json.loads(candidate)
            if not isinstance(parsed, dict):
                errors.append(f"c{idx}/raw: top-level JSON must be an object")
                continue
            meta.json_parse_ok = True
            return parsed, meta
        except Exception as exc:
            errors.append(f"c{idx}/raw: {exc}")

    if allow_repair:
        for idx, candidate in enumerate(candidates, start=1):
            repaired = lightweight_repair_json(candidate)
            try:
                parsed = json.loads(repaired)
                if not isinstance(parsed, dict):
                    errors.append(f"c{idx}/repair: top-level JSON must be an object")
                    continue
                meta.json_parse_ok = True
                meta.used_repair = True
                meta.repaired_output = repaired
                return parsed, meta
            except Exception as exc:
                errors.append(f"c{idx}/repair: {exc}")

    if not errors:
        errors.append("no JSON candidate found")
    if allow_repair:
        meta.parse_error = "JSON parse failed after one repair pass: " + " | ".join(errors)
    else:
        meta.parse_error = "JSON parse failed with repair disabled: " + " | ".join(errors)
    return None, meta


def _normalize_source_units(value: Any, valid_unit_ids: set[str]) -> list[str]:
    if not isinstance(value, list):
        return []
    normalized: list[str] = []
    for item in value:
        sid = str(item).strip()
        if sid and sid in valid_unit_ids:
            normalized.append(sid)
    return normalized


def _source_alias_map_from_payload(
    payload: dict[str, Any],
    conversation_units: Iterable[ConversationUnit],
) -> dict[str, str]:
    actual_by_text = {normalize_text(unit.text): unit.id for unit in conversation_units}
    aliases: dict[str, str] = {}
    raw_units = payload.get("conversation_units")
    if not isinstance(raw_units, list):
        return aliases
    for raw in raw_units:
        if not isinstance(raw, dict):
            continue
        raw_id = str(raw.get("id", "")).strip()
        raw_text = normalize_text(str(raw.get("text", "")).strip())
        if raw_id and raw_text in actual_by_text:
            aliases[raw_id] = actual_by_text[raw_text]
    return aliases


def _repair_item_source_units(
    raw_item: Any,
    *,
    valid_unit_ids: set[str],
    aliases: dict[str, str],
    conversation_units: Iterable[ConversationUnit],
) -> None:
    if not isinstance(raw_item, dict):
        return
    repaired: list[str] = []
    for source_id in raw_item.get("source_units", []):
        sid = str(source_id).strip()
        if sid in valid_unit_ids:
            repaired.append(sid)
        elif sid in aliases:
            repaired.append(aliases[sid])
    if not repaired:
        text_parts = [str(raw_item.get("text", "")).strip()]
        evidence = raw_item.get("evidence_spans", [])
        if isinstance(evidence, list):
            text_parts.extend(str(item).strip() for item in evidence if str(item).strip())
        repaired = _infer_source_units_from_text(" ".join(text_parts), conversation_units)
    raw_item["source_units"] = list(dict.fromkeys(repaired))


def _repair_payload_source_units(
    payload: dict[str, Any],
    conversation_units: Iterable[ConversationUnit],
) -> None:
    units = list(conversation_units)
    valid_unit_ids = {unit.id for unit in units}
    aliases = _source_alias_map_from_payload(payload, units)
    for list_key in (
        "functional_requirements",
        "non_functional_requirements",
        "constraints",
        "open_questions",
        "follow_up_questions",
        "notes",
    ):
        raw_items = payload.get(list_key)
        if not isinstance(raw_items, list):
            continue
        for raw_item in raw_items:
            _repair_item_source_units(
                raw_item,
                valid_unit_ids=valid_unit_ids,
                aliases=aliases,
                conversation_units=units,
            )


def _contains_few_shot_contamination(value: Any) -> bool:
    if isinstance(value, str):
        normalized = normalize_text(value)
        return any(hint in normalized for hint in FEW_SHOT_CONTAMINATION_HINTS)
    if isinstance(value, list):
        return any(_contains_few_shot_contamination(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_few_shot_contamination(item) for item in value.values())
    return False


def _remove_few_shot_contamination(payload: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    if _contains_few_shot_contamination(payload.get("project_summary", "")):
        payload["project_summary"] = (
            "The conversation describes a software project requiring source-grounded requirements."
        )
        warnings.append("few_shot_contamination_replaced_project_summary")

    for list_key in (
        "functional_requirements",
        "non_functional_requirements",
        "constraints",
        "open_questions",
        "follow_up_questions",
        "notes",
    ):
        raw_items = payload.get(list_key)
        if not isinstance(raw_items, list):
            continue
        kept: list[Any] = []
        removed = 0
        for raw_item in raw_items:
            if _contains_few_shot_contamination(raw_item):
                removed += 1
                continue
            kept.append(raw_item)
        if removed:
            payload[list_key] = kept
            warnings.append(f"few_shot_contamination_removed_{list_key}:{removed}")
    return warnings


def _infer_source_units_from_text(
    text: str,
    conversation_units: Iterable[ConversationUnit],
) -> list[str]:
    text_tokens = _tokens(text)
    if not text_tokens:
        return []

    scored: list[tuple[float, str]] = []
    for unit in conversation_units:
        unit_tokens = _tokens(unit.text)
        if not unit_tokens:
            continue
        overlap = len(text_tokens & unit_tokens) / max(1, len(text_tokens))
        if overlap > 0:
            scored.append((overlap, unit.id))

    if not scored:
        return []

    scored.sort(reverse=True)
    best_score = scored[0][0]
    return [unit_id for score, unit_id in scored if score >= best_score and score >= 0.08]


def _token_variants(token: str) -> set[str]:
    variants = {token}
    if token.endswith("ing") and len(token) > 5:
        stem = token[:-3]
        variants.add(stem)
        if stem.endswith("d"):
            variants.add(stem[:-1])
    if token.endswith("ed") and len(token) > 4:
        variants.add(token[:-2])
    if token.endswith("s") and len(token) > 4:
        variants.add(token[:-1])
    if token.endswith("ly") and len(token) > 4:
        variants.add(token[:-2])
    synonyms = {
        "quick": {"fast", "speed"},
        "quickly": {"fast", "speed", "quick"},
        "load": {"loading", "speed"},
        "loading": {"load", "speed"},
        "mobile": {"phone", "phones"},
        "phones": {"phone", "mobile", "android", "iphone", "ios"},
        "phone": {"phones", "mobile", "android", "iphone", "ios"},
        "android": {"mobile", "phone", "phones", "iphone", "ios"},
        "iphone": {"mobile", "phone", "phones", "android", "ios"},
        "ios": {"mobile", "phone", "phones", "android", "iphone"},
        "payment": {"payments", "pay"},
        "payments": {"payment", "pay"},
    }
    variants.update(synonyms.get(token, set()))
    return {variant for variant in variants if variant and variant not in STOPWORDS}


def _tokens(text: str) -> set[str]:
    parts = normalize_text(text).split()
    tokens: set[str] = set()
    for token in parts:
        if token and token not in STOPWORDS:
            tokens.update(_token_variants(token))
    return tokens


def _is_quality_oriented_text(text: str) -> bool:
    text_lc = normalize_text(text)
    return any(hint in text_lc for hint in QUALITY_HINTS)


def _is_vague_quality_text(text: str) -> bool:
    text_lc = normalize_text(text)
    if not _is_quality_oriented_text(text_lc):
        return False
    has_vague_hint = any(hint in text_lc for hint in VAGUE_QUALITY_HINTS)
    has_measurement_hint = any(hint in text_lc for hint in QUALITY_MEASUREMENT_HINTS)
    has_number = bool(re.search(r"\b\d+(\.\d+)?\b", text_lc))
    return has_vague_hint and not (has_measurement_hint or has_number)


def _is_style_only_quality_text(text: str) -> bool:
    text_lc = normalize_text(text)
    return any(
        hint in text_lc
        for hint in (
            "clean",
            "modern",
            "calm",
            "clear",
            "professional",
            "reassuring",
            "trustworthy",
            "style",
            "design should feel",
            "interface should feel",
            "look modern",
        )
    )


def _should_keep_vague_quality_as_nfr(text: str) -> bool:
    text_lc = normalize_text(text)
    if _is_style_only_quality_text(text_lc):
        return False
    return "first year" in text_lc or "first-year" in text_lc


def _is_privacy_or_security_prohibition_text(text: str) -> bool:
    text_lc = normalize_text(text)
    has_prohibition = any(
        phrase in text_lc
        for phrase in (
            "do not expose",
            "not expose",
            "do not reveal",
            "not reveal",
            "do not display",
            "not display",
            "do not show",
            "not show",
            "must not expose",
            "must not reveal",
            "must not display",
            "must not show",
        )
    )
    has_sensitive_data = any(
        phrase in text_lc
        for phrase in (
            "birth date",
            "birth dates",
            "contact information",
            "phone number",
            "email address",
            "personal information",
            "private information",
            "health notes",
            "student health",
            "children",
            "child",
        )
    )
    return has_prohibition and has_sensitive_data


def _looks_like_capability_statement(text: str) -> bool:
    text_lc = normalize_text(text)
    return any(hint in text_lc for hint in CAPABILITY_HINTS)


def _is_actor_action_capability_text(text: str) -> bool:
    text_lc = normalize_text(text)
    actor_hint = (
        r"(customers?|users?|staff|authorized staff|clinic staff|managers?|"
        r"clinic managers?|property managers?|nurses?|residents?|parents?|teachers?|"
        r"students?|patients?|trainers?|tutors?|officers?|applicants?|clerks?|permit clerks?)"
    )
    action_hint = (
        r"(record|submit|receive|send|assign|update|review|mark|approve|access|"
        r"see|view|track|manage|create|request|complete|upload|filter|leave|notify|replace)"
    )
    return bool(
        re.search(
            rf"\b{actor_hint}\b.+\b(should|must|need|needs)\b.+\b{action_hint}\b",
            text_lc,
        )
        or re.search(rf"\b{actor_hint}\b.+\b(should|must)\s+be\s+able\s+to\b", text_lc)
        or re.search(rf"\b{actor_hint}\b.+\b(want|wants)\s+to\s+{action_hint}\b", text_lc)
    )


def _is_system_action_capability_text(text: str) -> bool:
    text_lc = normalize_text(text)
    action_hint = (
        r"(notify|allow|let|replace|upload|filter|leave|show|display|create|"
        r"update|submit|record|track|connect|send|receive|export|import)"
    )
    return bool(
        re.search(
            rf"\b(system|website|site|app|application|platform|portal)\b.+\b(should|must|shall|needs? to)\b.+\b{action_hint}\b",
            text_lc,
        )
    )


def _looks_like_constraint_text(text: str) -> bool:
    text_lc = normalize_text(text)
    constraint_hints = (
        "not part of the first release",
        "not in version one",
        "not in v1",
        "not required at launch",
        "not needed for launch",
        "not required for launch",
        "not part of launch",
        "not part of the launch",
        "not in this release",
        "not in scope",
        "not decided",
        "undecided",
        "still being discussed",
        "outside initial release",
        "outside this project",
        "initial release only",
        "first launch",
        "first release",
        "future semester",
        "next semester",
        "later phase",
        "phase two",
        "web only",
        "no app",
        "within",
        "deadline",
        "budget",
        "only staff",
        "only admin",
        "only three",
        "only two",
        "only pilot",
        "must launch",
        "not to buy",
        "prefer not to buy",
        "separate document management product",
    )
    return any(hint in text_lc for hint in constraint_hints) or _is_deadline_constraint_text(text_lc)


def _looks_like_future_scope_text(text: str) -> bool:
    text_lc = normalize_text(text)
    return any(hint in text_lc for hint in FUTURE_SCOPE_HINTS)


def _is_uncertainty_text(text: str) -> bool:
    text_lc = normalize_text(text)
    return bool(
        any(
            hint in text_lc
            for hint in (
                "not sure whether",
                "not sure if",
                "not sure",
                "unsure whether",
                "unsure if",
                "undecided",
                "not decided",
                "still deciding",
            )
        )
        or re.search(r"\b(whether|if)\b.+\b(or|versus|vs)\b", text_lc)
    )


def _is_deadline_constraint_text(text: str) -> bool:
    text_lc = normalize_text(text)
    return bool(
        "deadline" in text_lc
        or re.search(r"\b(first\s+(?:\w+\s+){0,3}version|version one|v1|launch|release)\b.+\b(ready|done|complete|completed|available)\b.+\b(before|by)\b", text_lc)
        or re.search(r"\bpilot\b.+\b(ready|done|complete|completed|available)\b.+\b(before|by)\b", text_lc)
        or re.search(r"\bneed\s+(a\s+)?first\s+(?:\w+\s+){0,3}version\b.+\b(before|by)\b", text_lc)
        or re.search(r"\b(ready|done|complete|completed|available)\b.+\b(before|by)\b.+\b(launch|release|fair|event|deadline)\b", text_lc)
    )


def _is_hard_constraint_text(text: str) -> bool:
    return _looks_like_constraint_text(text) and not (
        _is_quality_oriented_text(text) and not _is_deadline_constraint_text(text)
    )


def _is_project_intro_text(text: str) -> bool:
    text_lc = normalize_text(text)
    product_nouns = (
        "app",
        "application",
        "dashboard",
        "portal",
        "platform",
        "site",
        "system",
        "tool",
        "website",
    )
    has_need_product = bool(
        re.search(r"\b(needs?|wants?)\b", text_lc)
        and any(re.search(rf"\b{noun}\b", text_lc) for noun in product_nouns)
    )
    if not has_need_product:
        return False
    has_embedded_capability = bool(
        re.search(r"\bwhere\b.+\b(can|should|must|able to)\b", text_lc)
        or re.search(
            r"\b(?:need|needs|want|wants)\s+an?\s+"
            r"(?:web app|web tool|app|application|dashboard|portal|system|tool|platform|website|site)\s+"
            r"to\s+\w+",
            text_lc,
        )
        or "should be able to" in text_lc
        or "must be able to" in text_lc
    )
    return not has_embedded_capability


def _project_intro_capability_text(text: str) -> str | None:
    cleaned = text.strip().rstrip(".")
    normalized = normalize_text(cleaned)
    product_action_match = re.search(
        r"\b(?:need|needs|want|wants)\s+an?\s+"
        r"(?P<product>web app|web tool|app|application|system|tool|platform|website|site)\s+"
        r"to\s+(?P<action>.+)$",
        cleaned,
        flags=re.IGNORECASE,
    )
    if product_action_match:
        product = product_action_match.group("product").lower()
        action = product_action_match.group("action").strip().rstrip(".")
        return f"The system shall provide a {product} to {action}."
    if "online booking system" in normalized:
        target = re.sub(r"^.*?for\s+", "", cleaned, flags=re.IGNORECASE).strip()
        if target and target != cleaned:
            return f"The system shall provide online booking for {target}."
        return "The system shall provide online booking."
    if "scheduling app" in normalized:
        tail = re.sub(r"^.*?scheduling app\s*(for\s+)?", "", cleaned, flags=re.IGNORECASE).strip()
        if tail:
            return f"The system shall provide scheduling for {tail}."
        return "The system shall provide scheduling."
    return None


def _rewrite_text_alignment_score(rewrite_text: str, candidate_text: str) -> float:
    rewrite_tokens = _tokens(rewrite_text)
    candidate_tokens = _tokens(candidate_text)
    if not rewrite_tokens or not candidate_tokens:
        return 0.0
    overlap = len(rewrite_tokens & candidate_tokens)
    denom = max(1, min(len(rewrite_tokens), len(candidate_tokens)))
    return overlap / denom


def coerce_rewrite_type_for_quality(text: str, rewrite_type: str) -> tuple[str, str | None]:
    if rewrite_type == "functional_requirement":
        if _is_quality_oriented_text(text) and not _looks_like_capability_statement(text):
            if _is_vague_quality_text(text) and not _should_keep_vague_quality_as_nfr(text):
                return "open_question", f"Please clarify measurable quality expectations for: {text}"
            return "non_functional_requirement", None
    if rewrite_type == "non_functional_requirement":
        if _looks_like_capability_statement(text) and not _is_quality_oriented_text(text):
            return "functional_requirement", None
        if _is_vague_quality_text(text) and not _should_keep_vague_quality_as_nfr(text):
            return "open_question", f"Please clarify measurable quality expectations for: {text}"
    return rewrite_type, None


def _dedupe_question_items(items: list[QuestionItem]) -> list[QuestionItem]:
    seen: set[tuple[str, tuple[str, ...]]] = set()
    vague_quality_sources: set[tuple[str, ...]] = set()
    deduped: list[QuestionItem] = []
    for item in items:
        normalized = normalize_text(item.text)
        sources_key = tuple(sorted(set(item.source_units)))
        is_vague_quality_question = (
            normalized.startswith("please clarify measurable quality expectations")
            or normalized.startswith("please confirm non functional requirement details")
            or normalized.startswith("please confirm non-functional requirement details")
            or normalized.startswith("please provide measurable acceptance criteria")
        )
        if is_vague_quality_question and sources_key in vague_quality_sources:
            continue
        if is_vague_quality_question:
            vague_quality_sources.add(sources_key)
        key = (normalized, sources_key)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _dedupe_note_items(items: list[NoteItem]) -> list[NoteItem]:
    seen: set[tuple[str, tuple[str, ...]]] = set()
    deduped: list[NoteItem] = []
    for item in items:
        key = (normalize_text(item.text), tuple(sorted(set(item.source_units))))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _append_question_once(
    items: list[QuestionItem],
    *,
    text: str,
    source_units: list[str],
) -> None:
    candidate = QuestionItem(text=text.strip(), source_units=source_units)
    candidate_key = (normalize_text(candidate.text), tuple(sorted(set(candidate.source_units))))
    for item in items:
        key = (normalize_text(item.text), tuple(sorted(set(item.source_units))))
        if key == candidate_key:
            return
    items.append(candidate)


def _future_scope_feature_from_text(text: str) -> str:
    feature = text.strip().rstrip(".")
    exclusion_then_future_match = re.match(
        r"(?P<feature>.+?)\s+is\s+not\s+(needed|required)\s+for\s+(the\s+)?first\s+release,\s+"
        r"but\s+(we\s+)?(may|might|could|can)\s+add\s+(it\s+)?later$",
        feature,
        flags=re.IGNORECASE,
    )
    if exclusion_then_future_match:
        return exclusion_then_future_match.group("feature").strip(" .")
    feature = re.sub(
        r"\s*,?\s*but\s+(it\s+is\s+|it\s+should\s+)?not\s+"
        r"((needed|required)\s+for\s+(launch|the\s+launch|first\s+release|the\s+first\s+release)|"
        r"(be\s+)?(part\s+of|in)\s+(launch|the\s+launch|this\s+release|the\s+first\s+release))$",
        "",
        feature,
        flags=re.IGNORECASE,
    )
    feature = re.sub(
        r"\s+would\s+be\s+(nice|useful)\s+later$",
        "",
        feature,
        flags=re.IGNORECASE,
    )
    feature = re.sub(
        r"\s+would\s+be\s+(nice|useful)$",
        "",
        feature,
        flags=re.IGNORECASE,
    )
    feature = re.sub(
        r"^(we|the team|team)\s+(may|might|can|could|will)\s+",
        "",
        feature,
        flags=re.IGNORECASE,
    )
    feature = re.sub(
        r"^(maybe\s+)?(we\s+)?(will\s+)?",
        "",
        feature,
        flags=re.IGNORECASE,
    )
    feature = re.sub(
        r"\s+(in\s+)?(a\s+)?(later phase|future semester|next semester|phase two|phase 2|later release|future release|later)$",
        "",
        feature,
        flags=re.IGNORECASE,
    )
    feature = re.sub(
        r"\s+(can|could|may|might|will)?\s*(be\s+)?(added|introduced|included|supported)$",
        "",
        feature,
        flags=re.IGNORECASE,
    )
    feature = re.sub(r"^(add|introduce|include|support)\s+", "", feature, flags=re.IGNORECASE)
    return feature.strip(" .") or "the future-scope feature"


def _future_scope_is_explicitly_deferred(text: str) -> bool:
    text_lc = normalize_text(text)
    return any(
        hint in text_lc
        for hint in (
            "should not block first launch",
            "should not block the first launch",
            "not block first launch",
            "not block the first launch",
            "can wait until phase two",
            "can wait until phase 2",
            "wait until phase two",
            "wait until phase 2",
            "not needed for launch",
            "not required for launch",
            "not needed for the launch",
            "not required for the launch",
            "not part of the first release",
            "not in the first release",
            "not needed for first release",
            "not required for first release",
            "not needed for the first release",
            "not required for the first release",
            "not in this release",
            "not part of launch",
            "not part of the launch",
            "not be part of launch",
            "not be part of the launch",
            "not be part of first release",
            "not be part of the first release",
        )
    )


def _add_future_scope_questions(
    *,
    text: str,
    source_units: list[str],
    open_questions: list[QuestionItem],
    follow_up_questions: list[QuestionItem],
) -> None:
    feature = _future_scope_feature_from_text(text)
    if _future_scope_is_explicitly_deferred(text):
        return
    _append_question_once(
        open_questions,
        text=f"Should {feature} be included in the first release or deferred?",
        source_units=source_units,
    )
    normalized_feature = normalize_text(feature)
    if any(word in normalized_feature for word in ("payment", "billing", "paid")):
        follow_up = f"Which payment providers or methods should be supported if {feature} is added later?"
    elif any(word in normalized_feature for word in ("id card", "integration", "external")):
        follow_up = f"What external systems are required for future {feature}?"
    elif any(word in normalized_feature for word in ("group", "session")):
        follow_up = f"How should {feature} differ from one-on-one sessions if added later?"
    else:
        follow_up = f"What implementation details are needed if {feature} is added later?"
    _append_question_once(
        follow_up_questions,
        text=follow_up,
        source_units=source_units,
    )


def _add_uncertainty_questions(
    *,
    text: str,
    source_units: list[str],
    open_questions: list[QuestionItem],
    follow_up_questions: list[QuestionItem],
) -> None:
    normalized = normalize_text(text)
    choice_match = re.search(
        r"\bwhether\s+(?P<actor>.+?)\s+should\s+(?P<option_a>.+?)\s+or\s+(?P<option_b>.+)$",
        text.strip().rstrip("."),
        flags=re.IGNORECASE,
    )
    if choice_match:
        actor = choice_match.group("actor").strip()
        option_a = choice_match.group("option_a").strip()
        option_b = choice_match.group("option_b").strip()
        actor = re.sub(
            r"^(we\s+are\s+|we're\s+|we\s+remain\s+|still\s+|not\s+sure\s+|unsure\s+|deciding\s+|"
            r"we\s+are\s+still\s+deciding\s+|we\s+are\s+not\s+sure\s+)",
            "",
            actor,
            flags=re.IGNORECASE,
        ).strip()
        open_question = f"Should {actor} {option_a} or {option_b}?"
        follow_up = "What decision criteria should be used to resolve this requirement option?"
    elif "account" in normalized and ("one time" in normalized or "one-time" in normalized or "link" in normalized):
        open_question = "Should users create accounts or use one-time links?"
        follow_up = "What authentication, privacy, and account recovery requirements apply to the selected access method?"
    else:
        open_question = f"Which option should be selected for this unresolved decision: {text.rstrip('.')}?"
        follow_up = "What decision criteria should be used to resolve this requirement option?"
    _append_question_once(open_questions, text=open_question, source_units=source_units)
    _append_question_once(follow_up_questions, text=follow_up, source_units=source_units)


def _add_vague_quality_questions(
    *,
    text: str,
    source_units: list[str],
    open_questions: list[QuestionItem],
    follow_up_questions: list[QuestionItem],
) -> None:
    _append_question_once(
        open_questions,
        text=f"Please clarify measurable quality expectations for: {text.rstrip('.')}?",
        source_units=source_units,
    )
    normalized = normalize_text(text)
    if any(word in normalized for word in ("clean", "modern", "calm", "professional", "visual", "look")):
        follow_up = "Could you provide reference websites or examples for the requested visual style?"
    elif any(word in normalized for word in ("easy", "simple", "usability")):
        follow_up = "What primary user flow should be used to judge whether the interface is easy enough?"
    elif "reliable" in normalized or "reliability" in normalized:
        follow_up = "Do you have uptime or error-rate targets for this reliability expectation?"
    elif "secure" in normalized or "security" in normalized:
        follow_up = "Which security or compliance controls are required for the protected data?"
    elif any(word in normalized for word in ("fast", "quick", "reliable", "secure")):
        follow_up = "What measurable target should be used to verify this quality expectation?"
    else:
        follow_up = "What observable criteria should be used to verify this quality expectation?"
    _append_question_once(
        follow_up_questions,
        text=follow_up,
        source_units=source_units,
    )


def _source_text_for_ids(source_units: list[str], unit_map: dict[str, str]) -> str:
    return " ".join(unit_map.get(source_id, "") for source_id in source_units).strip()


def _sources_or_item_are_future_scope(
    *,
    item_text: str,
    source_units: list[str],
    unit_map: dict[str, str],
) -> bool:
    source_text = _source_text_for_ids(source_units, unit_map)
    return _looks_like_future_scope_text(item_text) or _looks_like_future_scope_text(source_text)


def _semantic_source_text_for_item(
    item: RequirementItem | ConstraintItem,
    source_units: list[str],
    unit_map: dict[str, str],
) -> str:
    evidence_text = " ".join(span.strip() for span in item.evidence_spans if span.strip())
    if evidence_text:
        return evidence_text
    return _source_text_for_ids(source_units, unit_map)


def _dedupe_constraint_items(items: list[ConstraintItem]) -> list[ConstraintItem]:
    seen: set[tuple[str, tuple[str, ...]]] = set()
    deduped: list[ConstraintItem] = []
    for item in items:
        key = (normalize_text(item.text), tuple(sorted(set(item.source_units))))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _dedupe_requirement_items(items: list[RequirementItem]) -> list[RequirementItem]:
    seen: set[tuple[str, tuple[str, ...]]] = set()
    deduped: list[RequirementItem] = []
    for item in items:
        key = (normalize_text(item.text), tuple(sorted(set(item.source_units))))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


OBSOLETE_QUESTION_PREFIXES_WHEN_SOURCE_COVERED = (
    "please clarify whether this is a hard project constraint",
    "please confirm requirement scope and intent",
    "please confirm non functional requirement details",
    "please confirm non-functional requirement details",
)


def _remove_obsolete_questions_for_covered_sources(
    open_questions: list[QuestionItem],
    requirements: list[RequirementItem],
    constraints: list[ConstraintItem],
) -> list[QuestionItem]:
    covered_items = [*requirements, *constraints]
    covered_sources = {source_id for item in covered_items for source_id in item.source_units}
    filtered: list[QuestionItem] = []
    for question in open_questions:
        normalized_question = normalize_text(question.text)
        question_tokens = _tokens(question.text)
        related_covered_items = [
            item
            for item in covered_items
            if set(question.source_units).intersection(set(item.source_units))
        ]
        overlaps_covered_item = any(
            question_tokens
            and (
                len(question_tokens & _tokens(item.text))
                / max(1, min(len(question_tokens), len(_tokens(item.text))))
            )
            >= 0.4
            for item in related_covered_items
        )
        related_explicit_exclusion = any(
            isinstance(item, ConstraintItem)
            and "excluded from" in normalize_text(item.text)
            for item in related_covered_items
        )
        asks_to_redecide_excluded_scope = (
            related_explicit_exclusion
            and normalized_question.startswith("should ")
            and "included in the first release" in normalized_question
            and "deferred" in normalized_question
        )
        if asks_to_redecide_excluded_scope:
            continue
        if (
            question.source_units
            and all(source_id in covered_sources for source_id in question.source_units)
            and overlaps_covered_item
            and any(
                normalized_question.startswith(prefix)
                for prefix in OBSOLETE_QUESTION_PREFIXES_WHEN_SOURCE_COVERED
            )
        ):
            continue
        filtered.append(question)
    return filtered


def _remove_obsolete_notes_for_covered_sources(
    notes: list[NoteItem],
    requirements: list[RequirementItem],
    constraints: list[ConstraintItem],
) -> list[NoteItem]:
    covered_sources = {
        source_id
        for item in [*requirements, *constraints]
        for source_id in item.source_units
    }
    obsolete_prefixes = (
        "context source unit not converted into a requirement",
        "project context source unit not converted into a requirement",
        "future scope source unit not converted into a first release requirement",
        "future-scope source unit not converted into a first-release requirement",
        "future scope boundary candidate",
        "future-scope boundary candidate",
        "constraint like candidate requirement",
        "constraint-like candidate requirement",
        "constraint like quality expectation",
        "constraint-like quality expectation",
    )
    filtered: list[NoteItem] = []
    for note in notes:
        normalized = normalize_text(note.text)
        if (
            note.source_units
            and all(source_id in covered_sources for source_id in note.source_units)
            and any(normalized.startswith(prefix) for prefix in obsolete_prefixes)
        ):
            continue
        filtered.append(note)
    return filtered


def _reindex_requirement_items(items: list[RequirementItem], prefix: str) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
        payload = model_dump_compat(item)
        payload["id"] = f"{prefix}{index}"
        payloads.append(payload)
    return payloads


def _reindex_constraint_items(items: list[ConstraintItem]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
        payload = model_dump_compat(item)
        payload["id"] = f"CON{index}"
        payloads.append(payload)
    return payloads


def _repair_trace_item_sources(
    items: list[QuestionItem] | list[NoteItem],
    conversation_units: Iterable[ConversationUnit],
) -> list[QuestionItem] | list[NoteItem]:
    units = list(conversation_units)
    unit_map = {unit.id: unit.text for unit in units}
    repaired_items: list[QuestionItem | NoteItem] = []
    for item in items:
        valid_sources = [sid for sid in item.source_units if sid in unit_map]
        text_tokens = _tokens(item.text)
        source_tokens = _tokens(" ".join(unit_map[sid] for sid in valid_sources))
        overlap = (
            len(text_tokens & source_tokens) / len(text_tokens)
            if text_tokens and source_tokens
            else 0.0
        )
        if not valid_sources:
            inferred = _infer_source_units_from_text(item.text, units)
            if inferred:
                payload = model_dump_compat(item)
                payload["source_units"] = inferred
                repaired_items.append(model_validate_compat(type(item), payload))
                continue
        repaired_items.append(item)
    return repaired_items


def _is_formal_requirement_text(text: str) -> bool:
    return bool(
        re.match(
            r"^(the\s+)?(system|website|site|app|application)\s+(shall|should|must)\b",
            str(text).strip(),
            flags=re.IGNORECASE,
        )
    )


def _fallback_requirement_text_from_source(text: str, category: str) -> str:
    cleaned = text.strip().rstrip(".")
    if not cleaned:
        return "The system shall satisfy the stated requirement."

    actor_pattern = (
        r"(?P<actor>customers?|users?|staff|authorized staff|authorized office staff|"
        r"clinic staff|admins?|visitors?|clients?|developers?|members?|officers?|"
        r"students?|patients?|trainers?|tutors?|reception staff|nurses?|residents?|"
        r"parents?|teachers?|managers?|clinic managers?|property managers?|"
        r"applicants?|clerks?|permit clerks?|directors?)"
    )
    system_action_match = re.search(
        r"^(the\s+)?(system|website|site|app|application|platform)\s+"
        r"(?:should|must|shall)\s+(?P<action>.+)$",
        cleaned,
        flags=re.IGNORECASE,
    )
    if system_action_match:
        action = system_action_match.group("action").strip().rstrip(".")
        return f"The system shall {action}."
    conditional_system_action_match = re.search(
        r"^when\s+(?P<condition>.+?),\s*(the\s+)?(system|website|site|app|application|platform)\s+"
        r"(?:should|must|shall)\s+(?P<action>.+)$",
        cleaned,
        flags=re.IGNORECASE,
    )
    if conditional_system_action_match:
        condition = conditional_system_action_match.group("condition").strip(" .")
        action = conditional_system_action_match.group("action").strip().rstrip(".")
        return f"The system shall {action} when {condition}."
    admin_panel_match = re.search(
        rf"\b{actor_pattern}\s+(?:need|needs|should have|must have)\s+"
        r"an?\s+admin\s+(?P<surface>panel|dashboard|page)\s+to\s+(?P<action>.+)$",
        cleaned,
        flags=re.IGNORECASE,
    )
    if admin_panel_match:
        actor = admin_panel_match.group("actor").lower()
        surface = admin_panel_match.group("surface").lower()
        action = admin_panel_match.group("action").strip().rstrip(".")
        return f"The system shall provide an admin {surface} for {actor} to {action}."
    dashboard_match = re.search(
        rf"\b{actor_pattern}\s+(?:need|needs|should have|must have)\s+"
        r"an?\s+(?P<surface>dashboard|panel|page|queue)\s+"
        r"(?:to|where\s+they\s+can)\s+(?P<action>.+)$",
        cleaned,
        flags=re.IGNORECASE,
    )
    if dashboard_match:
        actor = dashboard_match.group("actor").lower()
        surface = dashboard_match.group("surface").lower()
        action = dashboard_match.group("action").strip().rstrip(".")
        return f"The system shall provide a {surface} for {actor} to {action}."
    ability_match = re.search(
        rf"\b{actor_pattern}\s+(?:should|must|shall)\s+be\s+able\s+to\s+(?P<action>.+)$",
        cleaned,
        flags=re.IGNORECASE,
    )
    if not ability_match:
        ability_match = re.search(
            rf"\b{actor_pattern}\s+(?:also\s+)?(?:should|must|shall|need|needs|want|wants)\s+to\s+(?P<action>.+)$",
            cleaned,
            flags=re.IGNORECASE,
        )
    if not ability_match:
        ability_match = re.search(
            rf"\b{actor_pattern}\s+(?:also\s+)?(?:should|must|shall|need|needs|want|wants)\s+"
            r"(?P<action>(?!be\b|to\b).+)$",
            cleaned,
            flags=re.IGNORECASE,
        )
    if not ability_match:
        ability_match = re.search(
            rf"\bwhere\s+{actor_pattern}\s+can\s+(?P<action>.+)$",
            cleaned,
            flags=re.IGNORECASE,
        )
    if not ability_match:
        ability_match = re.search(
            rf"\b{actor_pattern}\s+can\s+(?P<action>.+)$",
            cleaned,
            flags=re.IGNORECASE,
        )
    if ability_match:
        actor = ability_match.group("actor").lower()
        action = ability_match.group("action").strip().rstrip(".")
        return f"The system shall allow {actor} to {action}."

    if category == "non_functional_requirement":
        cleaned_lc = normalize_text(cleaned)
        if _is_privacy_or_security_prohibition_text(cleaned_lc):
            data_match = re.search(
                r"(?:expose|reveal|display|show)\s+(?P<data>.+?)\s+to\s+(?P<audience>.+)$",
                cleaned,
                flags=re.IGNORECASE,
            )
            if data_match:
                protected_data = data_match.group("data").strip(" .")
                audience = data_match.group("audience").strip(" .")
                return f"The system shall not expose {protected_data} to {audience}."
            return "The system shall protect sensitive personal information from unauthorized disclosure."
        if "tablet" in cleaned_lc:
            if "dashboard" in cleaned_lc:
                return "The staff dashboard shall support tablet screen usage."
            return "The system shall support tablet screen usage."
        if any(token in cleaned_lc for token in ("phone", "phones", "mobile", "android", "iphone", "ios")):
            if "load" in cleaned_lc or "quick" in cleaned_lc or "fast" in cleaned_lc:
                return "The system shall load quickly on mobile."
            if "android" in cleaned_lc and ("iphone" in cleaned_lc or "ios" in cleaned_lc):
                return "The system should support both Android and iPhone devices reliably."
            return "The system shall support mobile access."
        if "secure" in cleaned_lc or "security" in cleaned_lc:
            if "personal information" in cleaned_lc:
                return "The system shall be secure because it includes personal information."
            return "The system shall be secure."
        if "reliable" in cleaned_lc or "reliability" in cleaned_lc:
            if "exam" in cleaned_lc:
                return "The system shall be reliable around exam periods."
            return "The system shall be reliable."
        if "load" in cleaned_lc and ("fast" in cleaned_lc or "quick" in cleaned_lc):
            if "evening" in cleaned_lc or "rush" in cleaned_lc:
                return "The system shall load fast during evening rush hours."
            return "The system shall load quickly."
        if "easy to use" in cleaned_lc and "first" in cleaned_lc and "student" in cleaned_lc:
            return "The system shall be easy to use for first-year students."
        parts = re.split(r"\bso\b", cleaned, maxsplit=1, flags=re.IGNORECASE)
        quality_clause = parts[-1].strip(" ,") if parts else cleaned
        quality_clause = re.sub(
            r"^(the\s+)?(site|website|app|application|system)\s+(should|must|shall)\s+",
            "the system shall ",
            quality_clause,
            flags=re.IGNORECASE,
        )
        if re.match(r"^the system shall\b", quality_clause, flags=re.IGNORECASE):
            return quality_clause[:1].upper() + quality_clause[1:].rstrip(".") + "."
        return f"The system shall satisfy this quality requirement: {cleaned}."

    csv_match = re.search(r"\bbasic\s+csv\s+export\s+of\s+(?P<object>.+?)\s+is\s+enough", cleaned, flags=re.IGNORECASE)
    if csv_match:
        return f"The system shall provide a basic CSV export of {csv_match.group('object').strip(' .')}."
    actor_match = re.match(
        r"(?P<actor>.+?)\s+(?:should|must|shall|need(?:s)?\s+to|should\s+be\s+able\s+to|must\s+be\s+able\s+to)\s+(?P<action>.+)$",
        cleaned,
        flags=re.IGNORECASE,
    )
    if actor_match:
        actor = actor_match.group("actor").strip(" .")
        action = actor_match.group("action").strip(" .")
        return f"{actor[:1].upper() + actor[1:]} shall {action}."
    return f"The system shall provide: {cleaned}."


def _fallback_constraint_text_from_source(text: str) -> str:
    cleaned = text.strip().rstrip(".")
    normalized = normalize_text(cleaned)
    no_first_release_match = re.search(
        r"(?:no,\s*)?(?:the\s+)?first\s+release\s+does\s+not\s+need\s+(?P<feature>.+?)(?:\.\s*we\s+may\s+add\s+them\s+later)?$",
        cleaned,
        flags=re.IGNORECASE,
    )
    if no_first_release_match:
        feature = no_first_release_match.group("feature").strip(" .")
        return f"{feature[:1].upper() + feature[1:]} shall be deferred beyond the first release."
    avoid_purchase_match = re.search(
        r"(?:prefer\s+)?not\s+to\s+buy\s+(?P<product>.+?)(?:\s+unless\s+(?P<condition>.+))?$",
        cleaned,
        flags=re.IGNORECASE,
    )
    if avoid_purchase_match:
        product = avoid_purchase_match.group("product").strip(" .")
        condition = (avoid_purchase_match.group("condition") or "necessary").strip(" .")
        return f"The project shall avoid requiring {product} unless {condition}."
    if _is_deadline_constraint_text(cleaned):
        ready_match = re.search(
            r"(first\s+(?:\w+\s+){0,3}version|version one|v1|launch|release|pilot).+?\b(ready|done|complete|completed|available)\b\s+(?P<deadline>before|by)\s+(?P<target>.+)$",
            cleaned,
            flags=re.IGNORECASE,
        )
        if not ready_match:
            ready_match = re.search(
                r"(first\s+(?:\w+\s+){0,3}version).+?\b(?P<deadline>before|by)\s+(?P<target>.+)$",
                cleaned,
                flags=re.IGNORECASE,
            )
        if ready_match:
            subject = ready_match.group(1).strip()
            target = ready_match.group("target").strip(" .")
            if subject.lower() == "v1":
                subject = "version one"
            return f"The {subject} shall be ready {ready_match.group('deadline').lower()} {target}."
        return f"{cleaned[:1].upper() + cleaned[1:]}."
    wait_until_match = re.match(
        r"(?P<feature>.+?)\s+(?:can|could|may|might|should)?\s*wait\s+until\s+"
        r"(?P<when>phase two|phase 2|later phase|later release|future release|later)$",
        cleaned,
        flags=re.IGNORECASE,
    )
    if wait_until_match:
        feature = wait_until_match.group("feature").strip(" .")
        if feature:
            return f"{feature[:1].upper() + feature[1:]} shall be deferred to a later release."
    if any(
        hint in normalized
        for hint in (
            "excluded from launch",
            "excluded from the launch",
            "excluded from first release",
            "excluded from the first release",
            "not needed for launch",
            "not required for launch",
            "not needed for the launch",
            "not required for the launch",
            "not needed for first release",
            "not required for first release",
            "not needed for the first release",
            "not required for the first release",
            "not in this release",
            "not part of launch",
            "not part of the launch",
            "not be part of launch",
            "not be part of the launch",
            "not be part of first release",
            "not be part of the first release",
        )
    ):
        feature = _future_scope_feature_from_text(cleaned)
        scope = "first-release scope" if "release" in normalized else "launch scope"
        if re.search(r"\bbut\b.+\b(may|might|could|can)\s+add\b.+\blater\b", cleaned, flags=re.IGNORECASE):
            return (
                f"{feature[:1].upper() + feature[1:]} shall be excluded from the {scope} "
                "and may be considered for a future release."
            )
        return f"{feature[:1].upper() + feature[1:]} shall be excluded from the {scope}."
    deferred_match = re.match(
        r"(?P<feature>.+?)\s+(?:can|could|may|might|is planned to|will)?\s*"
        r"(?:be\s+)?(?:added|considered|planned|supported|included)?\s*"
        r"(?:in\s+)?(?:a\s+)?(?P<when>later phase|future semester|phase two|phase 2|later release|future release|later)$",
        cleaned,
        flags=re.IGNORECASE,
    )
    if deferred_match:
        feature = deferred_match.group("feature").strip(" .")
        feature = re.sub(
            r"^(we|the team|team)\s+(may|might|can|could|will)\s+(add|consider|include|support)\s+",
            "",
            feature,
            flags=re.IGNORECASE,
        ).strip(" .")
        if feature:
            return f"{feature[:1].upper() + feature[1:]} shall be deferred to a later release."
    if "not" in normalized and ("first release" in normalized or "phase two" in normalized or "launch" in normalized):
        return f"{cleaned[:1].upper() + cleaned[1:]}."
    return f"{cleaned[:1].upper() + cleaned[1:]} shall be treated as a project constraint."


def _normalize_requirement_from_source(
    item: RequirementItem,
    *,
    source_units: list[str],
    unit_map: dict[str, str],
    category: str,
) -> RequirementItem:
    if len(source_units) != 1 or source_units[0] not in unit_map:
        return item

    source_text = _semantic_source_text_for_item(item, source_units, unit_map)
    normalized_text: str | None = None
    if category == "functional_requirement":
        project_intro_capability = _project_intro_capability_text(source_text)
        source_capability_like = _looks_like_capability_statement(source_text) or any(
            phrase in normalize_text(source_text)
            for phrase in ("need ", "needs ", "should be able to", "must be able to", "where ")
        )
        if project_intro_capability:
            normalized_text = project_intro_capability
        elif source_capability_like and (
            _rewrite_text_alignment_score(item.text, source_text) < 0.5
            or not _is_formal_requirement_text(item.text)
        ):
            normalized_text = _fallback_requirement_text_from_source(
                source_text,
                "functional_requirement",
            )
    elif category == "non_functional_requirement":
        source_quality_tokens = normalize_text(source_text)
        item_quality_tokens = normalize_text(item.text)
        source_has_performance = any(
            hint in source_quality_tokens for hint in ("load", "quick", "quickly", "fast")
        )
        item_has_performance = any(
            hint in item_quality_tokens for hint in ("load", "quick", "quickly", "fast")
        )
        if _is_quality_oriented_text(source_text) and (
            not _is_quality_oriented_text(item.text)
            or (source_has_performance and not item_has_performance)
            or not _is_formal_requirement_text(item.text)
            or _rewrite_text_alignment_score(item.text, source_text) < 0.7
        ):
            normalized_text = _fallback_requirement_text_from_source(
                source_text,
                "non_functional_requirement",
            )

    if not normalized_text or normalize_text(normalized_text) == normalize_text(item.text):
        return item

    payload = model_dump_compat(item)
    payload["text"] = normalized_text
    payload["source_units"] = source_units
    payload["evidence_spans"] = [source_text]
    payload["acceptance_criteria"] = []
    payload["quality_checks"] = {}
    return model_validate_compat(RequirementItem, payload)


def _normalize_constraint_from_source(
    item: ConstraintItem,
    *,
    source_units: list[str],
    unit_map: dict[str, str],
) -> ConstraintItem:
    if len(source_units) != 1 or source_units[0] not in unit_map:
        return item
    source_text = _semantic_source_text_for_item(item, source_units, unit_map)
    if not (_looks_like_future_scope_text(source_text) or _is_deadline_constraint_text(source_text)):
        return item
    normalized_text = _fallback_constraint_text_from_source(source_text)
    if normalize_text(normalized_text) == normalize_text(item.text):
        return item
    payload = model_dump_compat(item)
    payload["text"] = normalized_text
    payload["source_units"] = source_units
    payload["evidence_spans"] = [source_text]
    payload["acceptance_criteria"] = []
    payload["quality_checks"] = {}
    return model_validate_compat(ConstraintItem, payload)


def _append_missing_source_unit_coverage(
    *,
    conversation_units: Iterable[ConversationUnit],
    verified_fr: list[RequirementItem],
    verified_nfr: list[RequirementItem],
    verified_constraints: list[ConstraintItem],
    open_questions: list[QuestionItem],
    follow_up_questions: list[QuestionItem],
    notes: list[NoteItem],
    warnings: list[str],
) -> None:
    covered_requirement_sources = {
        source_id
        for item in [*verified_fr, *verified_nfr, *verified_constraints]
        for source_id in item.source_units
    }
    covered_requirement_sources.update(
        source_id for item in open_questions for source_id in item.source_units
    )
    note_sources = {source_id for item in notes for source_id in item.source_units}

    for unit in conversation_units:
        if unit.id in covered_requirement_sources:
            continue

        text = unit.text.strip()
        if not text:
            continue
        text_lc = normalize_text(text)
        if _is_uncertainty_text(text):
            _add_uncertainty_questions(
                text=text,
                source_units=[unit.id],
                open_questions=open_questions,
                follow_up_questions=follow_up_questions,
            )
            warnings.append(f"source_units:{unit.id} coverage_fallback_recorded_uncertainty_question")
            continue
        if _is_deadline_constraint_text(text):
            verified_constraints.append(
                ConstraintItem(
                    id="CON_AUTO",
                    text=_fallback_constraint_text_from_source(text),
                    source_units=[unit.id],
                    evidence_spans=[text],
                )
            )
            warnings.append(f"source_units:{unit.id} coverage_fallback_added_deadline_constraint")
            continue
        if _is_project_intro_text(text):
            project_intro_capability = _project_intro_capability_text(text)
            if project_intro_capability:
                verified_fr.append(
                    RequirementItem(
                        id="FR_AUTO",
                        text=project_intro_capability,
                        source_units=[unit.id],
                        evidence_spans=[text],
                    )
                )
                warnings.append(f"source_units:{unit.id} coverage_fallback_added_project_intro_functional_requirement")
                continue
            warnings.append(f"source_units:{unit.id} coverage_fallback_recorded_project_context_note")
            continue
        quality_like = _is_quality_oriented_text(text)
        capability_like = _is_actor_action_capability_text(text) or _looks_like_capability_statement(text) or any(
            phrase in text_lc
            for phrase in (
                "need ",
                "needs ",
                "should be able to",
                "must be able to",
                "where ",
            )
        )
        future_scope_like = _looks_like_future_scope_text(text)
        constraint_like = _looks_like_constraint_text(text)
        vague_quality = _is_vague_quality_text(text)

        if _is_uncertainty_text(text):
            open_questions.append(
                QuestionItem(
                    text=text.rstrip("?") + "?",
                    source_units=[unit.id],
                )
            )
            warnings.append(f"source_units:{unit.id} coverage_fallback_recorded_open_question")
            continue

        if future_scope_like and not constraint_like:
            verified_constraints.append(
                ConstraintItem(
                    id="CON_AUTO",
                    text=_fallback_constraint_text_from_source(text),
                    source_units=[unit.id],
                    evidence_spans=[text],
                )
            )
            if unit.id not in note_sources:
                notes.append(
                    NoteItem(
                        text=f"Future-scope source unit not converted into a requirement: {text.rstrip('.')}.",
                        source_units=[unit.id],
                    )
                )
            _add_future_scope_questions(
                text=text,
                source_units=[unit.id],
                open_questions=open_questions,
                follow_up_questions=follow_up_questions,
            )
            warnings.append(f"source_units:{unit.id} coverage_fallback_recorded_future_scope_note")
            continue

        if vague_quality and not _should_keep_vague_quality_as_nfr(text):
            _add_vague_quality_questions(
                text=text,
                source_units=[unit.id],
                open_questions=open_questions,
                follow_up_questions=follow_up_questions,
            )
            warnings.append(f"source_units:{unit.id} coverage_fallback_recorded_open_question")
            continue

        if constraint_like and _is_hard_constraint_text(text):
            verified_constraints.append(
                ConstraintItem(
                    id="CON_AUTO",
                    text=_fallback_constraint_text_from_source(text),
                    source_units=[unit.id],
                    evidence_spans=[text],
                )
            )
            if future_scope_like:
                _add_future_scope_questions(
                    text=text,
                    source_units=[unit.id],
                    open_questions=open_questions,
                    follow_up_questions=follow_up_questions,
                )
            warnings.append(f"source_units:{unit.id} coverage_fallback_added_constraint")
            continue

        if quality_like and not capability_like:
            verified_nfr.append(
                RequirementItem(
                    id="NFR_AUTO",
                    text=_fallback_requirement_text_from_source(
                        text,
                        "non_functional_requirement",
                    ),
                    source_units=[unit.id],
                    evidence_spans=[text],
                )
            )
            warnings.append(f"source_units:{unit.id} coverage_fallback_added_non_functional_requirement")
            continue

        if capability_like:
            verified_fr.append(
                RequirementItem(
                    id="FR_AUTO",
                    text=_fallback_requirement_text_from_source(
                        text,
                        "functional_requirement",
                    ),
                    source_units=[unit.id],
                    evidence_spans=[text],
                )
            )
            warnings.append(f"source_units:{unit.id} coverage_fallback_added_functional_requirement")


def semantic_verify(
    spec_output: SpecOutput,
    conversation_units: Iterable[ConversationUnit],
) -> tuple[SpecOutput, list[str]]:
    unit_map = {unit.id: unit.text for unit in conversation_units}
    warnings: list[str] = []

    verified_fr: list[RequirementItem] = []
    verified_nfr: list[RequirementItem] = []
    verified_constraints: list[ConstraintItem] = []
    open_questions: list[QuestionItem] = list(spec_output.open_questions)
    follow_up_questions: list[QuestionItem] = list(spec_output.follow_up_questions)
    notes: list[NoteItem] = list(spec_output.notes)
    candidate_fr: list[RequirementItem] = []
    candidate_nfr: list[RequirementItem] = []

    for item in spec_output.functional_requirements:
        adjusted_type, clarification = coerce_rewrite_type_for_quality(
            item.text,
            "functional_requirement",
        )
        if adjusted_type == "non_functional_requirement":
            candidate_nfr.append(item)
            warnings.append(f"raw_functional_requirements:{item.id} reclassified_as_non_functional")
        elif adjusted_type == "open_question":
            open_questions.append(
                QuestionItem(
                    text=(clarification or f"Please clarify requirement details for: {item.text}").rstrip("?") + "?",
                    source_units=item.source_units,
                )
            )
            warnings.append(f"raw_functional_requirements:{item.id} reclassified_as_open_question")
        else:
            candidate_fr.append(item)

    for item in spec_output.non_functional_requirements:
        item_source_text = " ".join(unit_map.get(sid, "") for sid in item.source_units)
        if _is_privacy_or_security_prohibition_text(item.text) or _is_privacy_or_security_prohibition_text(item_source_text):
            candidate_nfr.append(item)
            continue
        adjusted_type, clarification = coerce_rewrite_type_for_quality(
            item.text,
            "non_functional_requirement",
        )
        if adjusted_type == "functional_requirement":
            candidate_fr.append(item)
            warnings.append(f"raw_non_functional_requirements:{item.id} reclassified_as_functional")
        elif adjusted_type == "open_question":
            open_questions.append(
                QuestionItem(
                    text=(clarification or f"Please clarify non-functional requirement details for: {item.text}").rstrip("?") + "?",
                    source_units=item.source_units,
                )
            )
            warnings.append(f"raw_non_functional_requirements:{item.id} reclassified_as_open_question")
        else:
            candidate_nfr.append(item)

    def _overlap_score(text: str, source_ids: list[str]) -> float:
        source_text = " ".join(unit_map[sid] for sid in source_ids if sid in unit_map)
        requirement_tokens = _tokens(text)
        source_tokens = _tokens(source_text)
        if not requirement_tokens:
            return 0.0
        return len(requirement_tokens & source_tokens) / len(requirement_tokens)

    def _evaluate_item(
        text: str,
        source_units: list[str],
        category: str,
        evidence_spans: list[str] | None = None,
    ) -> tuple[str, list[str], float, str]:
        reasons: list[str] = []
        valid_sources = [sid for sid in source_units if sid in unit_map]
        if not source_units:
            reasons.append("missing_source_units")
        if source_units and not valid_sources:
            reasons.append("source_units_not_found")

        overlap = _overlap_score(text, valid_sources)
        requirement_tokens = _tokens(text)
        overlap_threshold = 0.2 if category == "constraints" else 0.12
        if requirement_tokens and len(requirement_tokens) >= 4 and overlap < overlap_threshold:
            reasons.append("low_semantic_overlap")

        evidence_reasons = {
            "missing_source_units",
            "source_units_not_found",
            "low_semantic_overlap",
        }
        if any(reason in reasons for reason in evidence_reasons):
            inferred_sources = _infer_source_units_from_text(text, conversation_units)
            inferred_overlap = _overlap_score(text, inferred_sources)
            min_inferred_overlap = 0.2 if category == "constraints" else 0.12
            if (
                inferred_sources
                and inferred_overlap >= min_inferred_overlap
                and inferred_overlap >= (overlap + 0.08)
            ):
                valid_sources = inferred_sources
                overlap = inferred_overlap
                reasons = [r for r in reasons if r not in evidence_reasons]
                reasons.append("source_units_repaired_by_inference")

        source_text = " ".join(unit_map[sid] for sid in valid_sources)
        semantic_source_text = " ".join(span.strip() for span in (evidence_spans or []) if span.strip()) or source_text
        text_lc = text.lower()
        source_lc = semantic_source_text.lower()
        has_future_scope = any(h in text_lc or h in source_lc for h in FUTURE_SCOPE_HINTS)
        has_vague_wording = any(h in text_lc for h in VAGUE_HINTS)
        has_hard_boundary = any(h in text_lc or h in source_lc for h in HARD_BOUNDARY_HINTS)
        has_project_intro = _is_project_intro_text(text) or _is_project_intro_text(semantic_source_text)
        project_intro_capability = _project_intro_capability_text(semantic_source_text)
        has_uncertainty = _is_uncertainty_text(text) or _is_uncertainty_text(semantic_source_text)
        has_deadline_constraint = _is_deadline_constraint_text(text) or _is_deadline_constraint_text(semantic_source_text)

        action = "keep"
        if category == "constraints":
            reasons_requiring_downgrade = [
                reason for reason in reasons if reason != "source_units_repaired_by_inference"
            ]
            has_constraint_signal = (
                has_hard_boundary
                or _looks_like_constraint_text(text)
                or _looks_like_constraint_text(semantic_source_text)
            )
            if has_future_scope and not has_hard_boundary:
                action = "downgrade_note"
                reasons.append("future_scope_without_explicit_boundary")
            elif any(reason in reasons for reason in evidence_reasons):
                action = "downgrade_question"
            elif reasons_requiring_downgrade and not has_hard_boundary:
                action = "downgrade_question"
            elif not has_constraint_signal:
                action = "downgrade_question"
                reasons.append("not_hard_constraint")
            elif has_vague_wording and not has_hard_boundary:
                action = "downgrade_question"
                reasons.append("vague_constraint")
        else:
            reasons_requiring_downgrade = [
                reason for reason in reasons if reason != "source_units_repaired_by_inference"
            ]
            if has_uncertainty:
                action = "downgrade_question"
                reasons.append("unresolved_requirement_option")
            elif has_deadline_constraint:
                action = "downgrade_note"
                reasons.append("deadline_constraint_not_requirement")
            elif has_project_intro and not (
                category == "functional_requirements" and project_intro_capability
            ):
                action = "downgrade_note"
                reasons.append("project_context_not_requirement")
            elif has_future_scope:
                action = "downgrade_note"
                reasons.append("future_scope")
            elif reasons_requiring_downgrade:
                action = "downgrade_question"
            elif has_vague_wording and category == "functional_requirements":
                action = "downgrade_question"
                reasons.append("vague_functional_requirement")
            elif has_vague_wording and category == "non_functional_requirements":
                reasons.append("vague_non_functional_requirement")

        warning = ""
        if reasons:
            warning = f"downgraded/flagged ({', '.join(dict.fromkeys(reasons))})"
        return action, valid_sources or source_units, overlap, warning

    for item in candidate_fr:
        action, sources, overlap, warning = _evaluate_item(
            item.text, item.source_units, "functional_requirements", item.evidence_spans
        )
        if action == "keep":
            item_payload = model_dump_compat(item)
            item_payload["source_units"] = sources
            verified_fr.append(
                _normalize_requirement_from_source(
                    model_validate_compat(RequirementItem, item_payload),
                    source_units=sources,
                    unit_map=unit_map,
                    category="functional_requirement",
                )
            )
            if warning:
                warnings.append(f"raw_functional_requirements:{item.id} {warning}, overlap={overlap:.2f}")
            continue

        if warning:
            warnings.append(f"raw_functional_requirements:{item.id} {warning}, overlap={overlap:.2f}")

        if action == "downgrade_note":
            source_text = _source_text_for_ids(sources, unit_map)
            if "project_context_not_requirement" in warning:
                continue
            if _is_deadline_constraint_text(source_text or item.text):
                note_text = f"Constraint-like candidate requirement: {item.text}"
            else:
                note_text = f"Future-scope candidate requirement: {item.text}"
            notes.append(
                NoteItem(
                    text=note_text,
                    source_units=sources,
                )
            )
            if _sources_or_item_are_future_scope(
                item_text=item.text,
                source_units=sources,
                unit_map=unit_map,
            ):
                _add_future_scope_questions(
                    text=_source_text_for_ids(sources, unit_map) or item.text,
                    source_units=sources,
                    open_questions=open_questions,
                    follow_up_questions=follow_up_questions,
                )
        else:
            source_text = _source_text_for_ids(sources, unit_map) or item.text
            if _is_uncertainty_text(source_text) or source_text.strip().endswith("?"):
                _add_uncertainty_questions(
                    text=source_text,
                    source_units=sources,
                    open_questions=open_questions,
                    follow_up_questions=follow_up_questions,
                )
            else:
                open_questions.append(
                    QuestionItem(
                        text=f"Please confirm requirement scope and intent: {item.text}",
                        source_units=sources,
                    )
                )

    for item in candidate_nfr:
        action, sources, overlap, warning = _evaluate_item(
            item.text, item.source_units, "non_functional_requirements", item.evidence_spans
        )
        if action == "keep":
            item_payload = model_dump_compat(item)
            item_payload["source_units"] = sources
            verified_nfr.append(
                _normalize_requirement_from_source(
                    model_validate_compat(RequirementItem, item_payload),
                    source_units=sources,
                    unit_map=unit_map,
                    category="non_functional_requirement",
                )
            )
            if warning:
                warnings.append(
                    f"raw_non_functional_requirements:{item.id} {warning}, overlap={overlap:.2f}"
                )
                open_questions.append(
                    QuestionItem(
                        text=f"Please provide measurable acceptance criteria for: {item.text}",
                        source_units=sources,
                    )
                )
                _add_vague_quality_questions(
                    text=" ".join(unit_map.get(source_id, "") for source_id in sources) or item.text,
                    source_units=sources,
                    open_questions=open_questions,
                    follow_up_questions=follow_up_questions,
                )
            continue

        if warning:
            warnings.append(
                f"raw_non_functional_requirements:{item.id} {warning}, overlap={overlap:.2f}"
            )

        if action == "downgrade_note":
            source_text = _source_text_for_ids(sources, unit_map)
            if "project_context_not_requirement" in warning:
                continue
            if _is_deadline_constraint_text(source_text or item.text):
                note_text = f"Constraint-like quality expectation: {item.text}"
            else:
                note_text = f"Future-scope quality expectation: {item.text}"
            notes.append(
                NoteItem(
                    text=note_text,
                    source_units=sources,
                )
            )
            if _sources_or_item_are_future_scope(
                item_text=item.text,
                source_units=sources,
                unit_map=unit_map,
            ):
                _add_future_scope_questions(
                    text=_source_text_for_ids(sources, unit_map) or item.text,
                    source_units=sources,
                    open_questions=open_questions,
                    follow_up_questions=follow_up_questions,
                )
        else:
            source_text = _source_text_for_ids(sources, unit_map) or item.text
            if _is_uncertainty_text(source_text):
                _add_uncertainty_questions(
                    text=source_text,
                    source_units=sources,
                    open_questions=open_questions,
                    follow_up_questions=follow_up_questions,
                )
            else:
                open_questions.append(
                    QuestionItem(
                        text=f"Please confirm non-functional requirement details: {item.text}",
                        source_units=sources,
                    )
                )

    for item in spec_output.constraints:
        action, sources, overlap, warning = _evaluate_item(
            item.text, item.source_units, "constraints", item.evidence_spans
        )
        if action == "keep":
            source_text = _source_text_for_ids(sources, unit_map) or item.text
            if _is_uncertainty_text(source_text):
                open_questions.append(
                    QuestionItem(text=source_text.rstrip("?") + "?", source_units=sources)
                )
                warnings.append(f"raw_constraints:{item.id} reclassified_as_open_question")
                continue
            item_payload = model_dump_compat(item)
            item_payload["source_units"] = sources
            verified_constraints.append(
                _normalize_constraint_from_source(
                    model_validate_compat(ConstraintItem, item_payload),
                    source_units=sources,
                    unit_map=unit_map,
                )
            )
            if warning:
                warnings.append(f"raw_constraints:{item.id} {warning}, overlap={overlap:.2f}")
                if "vague_constraint" in warning:
                    open_questions.append(
                        QuestionItem(
                            text=f"Please specify concrete boundary details for: {item.text}",
                            source_units=sources,
                        )
                    )
            continue

        if warning:
            warnings.append(f"raw_constraints:{item.id} {warning}, overlap={overlap:.2f}")

        if action == "downgrade_note":
            notes.append(
                NoteItem(
                    text=f"Future-scope boundary candidate: {item.text}",
                    source_units=sources,
                )
            )
            if _sources_or_item_are_future_scope(
                item_text=item.text,
                source_units=sources,
                unit_map=unit_map,
            ):
                _add_future_scope_questions(
                    text=_source_text_for_ids(sources, unit_map) or item.text,
                    source_units=sources,
                    open_questions=open_questions,
                    follow_up_questions=follow_up_questions,
                )
        else:
            open_questions.append(
                QuestionItem(
                    text=f"Please clarify whether this is a hard project constraint: {item.text}",
                    source_units=sources,
                )
            )

    _append_missing_source_unit_coverage(
        conversation_units=conversation_units,
        verified_fr=verified_fr,
        verified_nfr=verified_nfr,
        verified_constraints=verified_constraints,
        open_questions=open_questions,
        follow_up_questions=follow_up_questions,
        notes=notes,
        warnings=warnings,
    )

    kept_fr: list[RequirementItem] = []
    for item in verified_fr:
        source_text = " ".join(unit_map.get(sid, "") for sid in item.source_units)
        privacy_text = f"{item.text} {source_text}"
        privacy_lc = normalize_text(privacy_text)
        if _is_privacy_or_security_prohibition_text(privacy_text) or (
            "not store" in privacy_lc
            and any(term in privacy_lc for term in ("national id", "resident registration", "payment card", "card details"))
        ):
            verified_nfr.append(item)
            warnings.append(f"functional_requirements:{item.id} reclassified_as_non_functional_privacy")
        else:
            kept_fr.append(item)
    verified_fr = kept_fr
    kept_constraints: list[ConstraintItem] = []
    for item in verified_constraints:
        source_text = _source_text_for_ids(item.source_units, unit_map) or item.text
        if item.text.strip().endswith("?") or source_text.strip().endswith("?"):
            open_questions.append(
                QuestionItem(text=source_text.rstrip("?") + "?", source_units=item.source_units)
            )
            warnings.append(f"constraints:{item.id} reclassified_as_open_question")
        else:
            kept_constraints.append(item)
    verified_constraints = kept_constraints

    covered_sources = {
        source_id
        for item in [*verified_fr, *verified_nfr, *verified_constraints]
        for source_id in item.source_units
    }
    answered_questions: set[tuple[str, ...]] = set()
    for question in open_questions:
        if len(question.source_units) != 1:
            continue
        source_id = question.source_units[0]
        source_text = unit_map.get(source_id, "")
        if not source_text.strip().endswith("?"):
            continue
        match = re.match(r"U(?P<num>\d+)$", source_id)
        if not match:
            continue
        current = int(match.group("num"))
        if any(f"U{idx}" in covered_sources for idx in range(current + 1, current + 3)):
            answered_questions.add(tuple(question.source_units))
    if answered_questions:
        open_questions = [
            question
            for question in open_questions
            if tuple(question.source_units) not in answered_questions
        ]
        warnings.append(f"open_questions:answered_by_following_source_units:{len(answered_questions)}")

    payload = model_dump_compat(spec_output)
    verified_fr = _dedupe_requirement_items(verified_fr)
    verified_nfr = _dedupe_requirement_items(verified_nfr)
    verified_constraints = _dedupe_constraint_items(verified_constraints)
    open_questions = _remove_obsolete_questions_for_covered_sources(
        open_questions,
        [*verified_fr, *verified_nfr],
        verified_constraints,
    )
    notes = _remove_obsolete_notes_for_covered_sources(
        notes,
        [*verified_fr, *verified_nfr],
        verified_constraints,
    )
    payload["functional_requirements"] = _reindex_requirement_items(
        verified_fr,
        "FR",
    )
    payload["non_functional_requirements"] = _reindex_requirement_items(
        verified_nfr,
        "NFR",
    )
    payload["constraints"] = _reindex_constraint_items(
        verified_constraints
    )
    open_questions = _repair_trace_item_sources(
        _dedupe_question_items(open_questions),
        conversation_units,
    )
    follow_up_questions = _repair_trace_item_sources(
        _dedupe_question_items(follow_up_questions),
        conversation_units,
    )
    notes = _repair_trace_item_sources(
        _dedupe_note_items(notes),
        conversation_units,
    )
    payload["open_questions"] = [
        model_dump_compat(item) for item in open_questions
    ]
    payload["follow_up_questions"] = [
        model_dump_compat(item) for item in follow_up_questions
    ]
    payload["notes"] = [model_dump_compat(item) for item in notes]
    combined_warnings = sorted(set(list(spec_output.verification_warnings) + warnings))
    payload["verification_warnings"] = combined_warnings
    verified_spec = model_validate_compat(SpecOutput, payload)
    return verified_spec, combined_warnings


def _source_unit_decision_items(payload: dict[str, Any]) -> list[Any] | None:
    for key in ("source_unit_decisions", "unit_decisions", "decisions"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
    return None


def _source_unit_id_from_decision(raw: dict[str, Any]) -> str:
    return str(
        raw.get("source_unit")
        or raw.get("source_unit_id")
        or raw.get("source_id")
        or raw.get("id")
        or ""
    ).strip()


def _iter_atomic_source_unit_decisions(payload: dict[str, Any]) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for raw in _source_unit_decision_items(payload) or []:
        if not isinstance(raw, dict):
            continue
        source_id = _source_unit_id_from_decision(raw)
        atomic_items = raw.get("atomic_decisions")
        if not isinstance(atomic_items, list):
            flattened.append(raw)
            continue
        if not atomic_items:
            continue
        for atomic in atomic_items:
            if not isinstance(atomic, dict):
                continue
            merged = dict(atomic)
            merged["source_unit"] = source_id
            for inherited_key in ("open_question", "follow_up_question", "note"):
                if not str(merged.get(inherited_key, "")).strip() and raw.get(inherited_key):
                    merged[inherited_key] = raw.get(inherited_key)
            flattened.append(merged)
    return flattened


def _atomic_decision_text(raw: dict[str, Any], source_text: str) -> str:
    for key in ("claim", "text", "requirement", "requirement_text"):
        value = str(raw.get(key, "")).strip()
        if value and not _contains_few_shot_contamination(value):
            return value
    return source_text


def _has_atomic_decision_text(raw: dict[str, Any]) -> bool:
    return any(
        str(raw.get(key, "")).strip() and not _contains_few_shot_contamination(str(raw.get(key, "")))
        for key in ("claim", "text", "requirement", "requirement_text")
    )


def _clean_atomic_clause(clause: str, previous_actor: str | None = None) -> str:
    cleaned = re.sub(r"\s+", " ", clause).strip(" ,.;")
    cleaned = re.sub(r"^(but|although|however)\s+", "", cleaned, flags=re.IGNORECASE).strip()
    if previous_actor and re.match(r"^(they|them)\b", cleaned, flags=re.IGNORECASE):
        cleaned = re.sub(r"^(they|them)\b", previous_actor, cleaned, count=1, flags=re.IGNORECASE)
    if previous_actor and re.match(r"^also\s+want(s)?\s+to\b", cleaned, flags=re.IGNORECASE):
        cleaned = f"{previous_actor} {cleaned}"
    return cleaned


def _leading_actor_phrase(text: str) -> str | None:
    match = re.match(
        r"(?P<actor>[A-Z]?[a-z]+(?:\s+[a-z]+){0,3})\s+"
        r"(?:should|must|shall|need|needs|want|wants|can|may|might)\b",
        text.strip(),
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    actor = match.group("actor").strip()
    if normalize_text(actor) in {"the system", "system", "it"}:
        return None
    return actor


def _split_atomic_decision_clauses(text: str) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text.strip())
    if not cleaned:
        return []
    if re.search(
        r"\b(may|might|could)\s+later\b.+\bbut\s+(that\s+)?(integration|feature|work)\b.+\bnot\s+block\b.+\b(first\s+)?launch\b",
        cleaned,
        flags=re.IGNORECASE,
    ):
        return [cleaned]
    if re.search(
        r"\b(may|might|could)\s+(be\s+)?(added|included|supported)\s+later\b.+\bbut\b.+\bnot\b.+\b(part\s+of|in|needed|required)\b.+\blaunch\b",
        cleaned,
        flags=re.IGNORECASE,
    ):
        return [cleaned]
    if re.search(
        r"\bnot\s+(needed|required)\s+for\s+(the\s+)?first\s+release\b.+\bbut\b.+\b(may|might|could|can)\s+add\b.+\blater\b",
        cleaned,
        flags=re.IGNORECASE,
    ):
        return [cleaned]

    multi_action_match = re.match(
        r"(?P<actor>staff|parents|users|customers|managers|teachers|students|residents|applicants)\s+"
        r"(?:should|must|shall)\s+be\s+able\s+to\s+"
        r"(?P<first>[^,]+),\s+(?P<second>[^,]+),\s+and\s+(?P<third>.+)$",
        cleaned,
        flags=re.IGNORECASE,
    )
    if multi_action_match:
        actor = multi_action_match.group("actor")
        return [
            f"{actor} should be able to {multi_action_match.group('first').strip()}",
            f"{actor} should be able to {multi_action_match.group('second').strip()}",
            f"{actor} should be able to {multi_action_match.group('third').strip()}",
        ]

    protected = cleaned
    protected = re.sub(r"\s*,?\s+but\s+", " || but ", protected, flags=re.IGNORECASE)
    protected = re.sub(r"\s*,?\s+although\s+", " || although ", protected, flags=re.IGNORECASE)
    protected = re.sub(r"\s*,?\s+however\s+", " || however ", protected, flags=re.IGNORECASE)
    protected = re.sub(r"\s*,?\s+and\s+they\s+also\s+", " || they also ", protected, flags=re.IGNORECASE)
    protected = re.sub(
        r"\s*,?\s+and\s+the\s+city\s+would\s+prefer\s+",
        " || the city would prefer ",
        protected,
        flags=re.IGNORECASE,
    )

    raw_parts = [part for part in protected.split(" || ") if part.strip()]
    if len(raw_parts) <= 1:
        return [cleaned]

    clauses: list[str] = []
    previous_actor: str | None = None
    for part in raw_parts:
        clause = _clean_atomic_clause(part, previous_actor)
        if not clause:
            continue
        previous_actor = _leading_actor_phrase(clause) or previous_actor
        clauses.append(clause)
    return clauses or [cleaned]


def _normalize_source_unit_decision(decision: str, source_text: str) -> str:
    text = normalize_text(str(decision or ""))
    source_lc = normalize_text(source_text)
    if _is_privacy_or_security_prohibition_text(source_lc):
        return "non_functional_requirement"
    if _is_project_intro_text(source_lc):
        return "note"
    if _is_uncertainty_text(source_lc):
        return "open_question"
    if _is_deadline_constraint_text(source_lc):
        return "constraint"
    if _looks_like_future_scope_text(source_lc):
        return "constraint"
    if _looks_like_constraint_text(source_lc) and _is_hard_constraint_text(source_lc):
        return "constraint"
    if _is_actor_action_capability_text(source_lc) or _is_system_action_capability_text(source_lc):
        return "functional_requirement"
    if _is_quality_oriented_text(source_lc) and not _looks_like_future_scope_text(source_lc):
        return "non_functional_requirement"
    if not text:
        if _looks_like_future_scope_text(source_lc) or _looks_like_constraint_text(source_lc):
            return "constraint"
        if _is_quality_oriented_text(source_lc):
            return "non_functional_requirement"
        if _looks_like_capability_statement(source_lc):
            return "functional_requirement"
        return "note"
    if any(word in text for word in ("discard", "ignore", "none")):
        return "discard"
    if any(word in text for word in ("future", "scope", "constraint", "boundary", "defer")):
        return "constraint"
    if any(word in text for word in ("question", "ambiguity", "clarify", "unclear")):
        return "open_question"
    if any(word in text for word in ("note", "context", "summary")):
        return "note"
    if any(word in text for word in ("non functional", "non_functional", "nfr", "quality", "performance", "usability", "security", "reliability", "mobile")):
        return "non_functional_requirement"
    if any(word in text for word in ("functional", "requirement", "capability", "feature", "fr")):
        if _is_quality_oriented_text(source_lc) and not _looks_like_capability_statement(source_lc):
            return "non_functional_requirement"
        return "functional_requirement"
    return "note"


def _question_from_decision(
    raw: dict[str, Any],
    *,
    source_text: str,
    source_units: list[str],
    fallback_prefix: str,
) -> QuestionItem:
    text = str(
        raw.get("open_question")
        or raw.get("question")
        or raw.get("clarification_question")
        or ""
    ).strip()
    if not text:
        text = f"{fallback_prefix}: {source_text.rstrip('.')}?"
    if not text.endswith("?"):
        text = text.rstrip(".") + "?"
    return QuestionItem(text=text, source_units=source_units)


def _build_spec_from_source_unit_decisions(
    payload: dict[str, Any],
    conversation_units: Iterable[ConversationUnit],
) -> SpecOutput:
    units = list(conversation_units)
    unit_map = {unit.id: unit.text for unit in units}
    project_summary = str(payload.get("project_summary", "")).strip()
    if not project_summary:
        project_summary = "The conversation describes a software project with source-grounded requirements."

    functional_requirements: list[RequirementItem] = []
    non_functional_requirements: list[RequirementItem] = []
    constraints: list[ConstraintItem] = []
    open_questions: list[QuestionItem] = []
    follow_up_questions: list[QuestionItem] = []
    notes: list[NoteItem] = []
    warnings: list[str] = ["source_unit_decision_schema_used"]
    handled_decisions: set[tuple[str, str, str]] = set()
    raw_items: list[dict[str, Any]] = []
    for raw in _iter_atomic_source_unit_decisions(payload):
        if not isinstance(raw, dict):
            continue
        source_id = _source_unit_id_from_decision(raw)
        source_text = unit_map.get(source_id, "").strip()
        if source_text and not _has_atomic_decision_text(raw):
            clauses = _split_atomic_decision_clauses(source_text)
            if len(clauses) > 1:
                warnings.append(f"source_unit_decisions:{source_id} split_blank_claim_into_{len(clauses)}_atomic_claims")
            for clause in clauses:
                split_raw = dict(raw)
                split_raw["claim"] = clause
                raw_items.append(split_raw)
            continue
        raw_items.append(raw)

    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        source_id = _source_unit_id_from_decision(raw)
        if source_id not in unit_map:
            if source_id:
                warnings.append(f"source_unit_decisions:{source_id} ignored_unknown_source_unit")
            continue
        source_text = unit_map[source_id].strip()
        decision_text = _atomic_decision_text(raw, source_text).strip()
        evidence_text = decision_text if normalize_text(decision_text) != normalize_text(source_text) else source_text
        source_units = [source_id]
        decision = _normalize_source_unit_decision(str(raw.get("decision", "")), decision_text)
        decision_key = (source_id, decision, normalize_text(decision_text))
        if decision_key in handled_decisions:
            continue
        handled_decisions.add(decision_key)

        if _is_uncertainty_text(decision_text):
            _add_uncertainty_questions(
                text=decision_text,
                source_units=source_units,
                open_questions=open_questions,
                follow_up_questions=follow_up_questions,
            )
            continue

        if _is_deadline_constraint_text(decision_text):
            constraints.append(
                ConstraintItem(
                    id="CON_AUTO",
                    text=_fallback_constraint_text_from_source(decision_text),
                    source_units=source_units,
                    evidence_spans=[evidence_text],
                )
            )
            continue

        if _looks_like_future_scope_text(decision_text):
            constraints.append(
                ConstraintItem(
                    id="CON_AUTO",
                    text=_fallback_constraint_text_from_source(decision_text),
                    source_units=source_units,
                    evidence_spans=[evidence_text],
                )
            )
            notes.append(
                NoteItem(
                    text=f"Future-scope source unit not converted into a first-release requirement: {decision_text.rstrip('.')}.",
                    source_units=source_units,
                )
            )
            _add_future_scope_questions(
                text=decision_text,
                source_units=source_units,
                open_questions=open_questions,
                follow_up_questions=follow_up_questions,
            )
            continue

        if decision == "discard":
            notes.append(
                NoteItem(
                    text=f"Source unit discarded by model decision: {decision_text.rstrip('.')}.",
                    source_units=source_units,
                )
            )
            continue

        if decision == "open_question":
            open_questions.append(
                _question_from_decision(
                    raw,
                    source_text=decision_text,
                    source_units=source_units,
                    fallback_prefix="Please clarify requirement details for",
                )
            )
            follow_up = str(raw.get("follow_up_question") or raw.get("followup_question") or "").strip()
            if follow_up:
                if not follow_up.endswith("?"):
                    follow_up = follow_up.rstrip(".") + "?"
                follow_up_questions.append(QuestionItem(text=follow_up, source_units=source_units))
            continue

        if decision == "note" and not (
            _is_quality_oriented_text(decision_text)
            or _looks_like_capability_statement(decision_text)
            or _looks_like_constraint_text(decision_text)
            or _project_intro_capability_text(decision_text)
        ):
            note = str(raw.get("note") or raw.get("claim") or "").strip()
            notes.append(
                NoteItem(
                    text=note or f"Context source unit not converted into a requirement: {decision_text.rstrip('.')}.",
                    source_units=source_units,
                )
            )
            continue

        if decision == "constraint" or _is_hard_constraint_text(decision_text):
            constraints.append(
                ConstraintItem(
                    id="CON_AUTO",
                    text=_fallback_constraint_text_from_source(decision_text),
                    source_units=source_units,
                    evidence_spans=[evidence_text],
                )
            )
            continue

        if decision == "non_functional_requirement" or (
            _is_quality_oriented_text(decision_text) and not _looks_like_capability_statement(decision_text)
        ):
            if _is_vague_quality_text(decision_text):
                _add_vague_quality_questions(
                    text=decision_text,
                    source_units=source_units,
                    open_questions=open_questions,
                    follow_up_questions=follow_up_questions,
                )
                if not _should_keep_vague_quality_as_nfr(decision_text):
                    continue
            non_functional_requirements.append(
                RequirementItem(
                    id="NFR_AUTO",
                    text=_fallback_requirement_text_from_source(
                        decision_text,
                        "non_functional_requirement",
                    ),
                    source_units=source_units,
                    evidence_spans=[evidence_text],
                )
            )
            continue

        capability_text = _project_intro_capability_text(decision_text)
        functional_requirements.append(
            RequirementItem(
                id="FR_AUTO",
                text=capability_text
                or _fallback_requirement_text_from_source(decision_text, "functional_requirement"),
                source_units=source_units,
                evidence_spans=[evidence_text],
            )
        )

    return SpecOutput(
        project_summary=project_summary,
        functional_requirements=functional_requirements,
        non_functional_requirements=non_functional_requirements,
        constraints=constraints,
        open_questions=open_questions,
        follow_up_questions=follow_up_questions,
        notes=notes,
        conversation_units=units,
        verification_warnings=warnings,
    )


# Single-shot output helpers.
def extract_spec_output_safe(
    raw_output: str,
    conversation_units: Iterable[ConversationUnit],
    *,
    allow_repair: bool = True,
) -> tuple[SpecOutput | None, ExtractionMeta]:
    payload, meta = parse_json_object_safe(raw_output, allow_repair=allow_repair)
    if payload is None:
        return None, meta
    try:
        if isinstance(payload.get("output"), dict):
            wrapper_payload = payload
            payload = dict(payload["output"])
            if "conversation_units" not in payload:
                payload["conversation_units"] = wrapper_payload.get("conversation_units", [])

        if _source_unit_decision_items(payload) is not None:
            spec = _build_spec_from_source_unit_decisions(payload, conversation_units)
            meta.pydantic_validation_ok = True
            return spec, meta

        required_fields = (
            "project_summary",
            "functional_requirements",
            "non_functional_requirements",
            "constraints",
            "open_questions",
            "follow_up_questions",
            "notes",
        )
        missing = [key for key in required_fields if key not in payload]
        if missing:
            raise ExtractionError(f"Missing required top-level fields: {', '.join(missing)}")
        for list_key in (
            "functional_requirements",
            "non_functional_requirements",
            "constraints",
            "open_questions",
            "follow_up_questions",
            "notes",
        ):
            if not isinstance(payload.get(list_key), list):
                raise ExtractionError(f"Field `{list_key}` must be a list.")
        if not isinstance(payload.get("project_summary"), str) or not str(
            payload.get("project_summary", "")
        ).strip():
            raise ExtractionError("Field `project_summary` must be a non-empty string.")

        normalized = {
            "project_summary": str(payload.get("project_summary", "")).strip()
            or "The conversation describes a software project requiring clarification.",
            "functional_requirements": payload.get("functional_requirements", []),
            "non_functional_requirements": payload.get("non_functional_requirements", []),
            "constraints": payload.get("constraints", []),
            "open_questions": payload.get("open_questions", []),
            "follow_up_questions": payload.get("follow_up_questions", []),
            "notes": payload.get("notes", []),
            "conversation_units": [model_dump_compat(u) for u in conversation_units],
            "verification_warnings": payload.get("verification_warnings", []),
        }
        _repair_payload_source_units(normalized, conversation_units)
        contamination_warnings = _remove_few_shot_contamination(normalized)
        if contamination_warnings:
            existing_warnings = normalized.get("verification_warnings", [])
            if not isinstance(existing_warnings, list):
                existing_warnings = []
            normalized["verification_warnings"] = [
                *[str(item) for item in existing_warnings],
                *contamination_warnings,
            ]
        spec = model_validate_compat(SpecOutput, normalized)
        meta.pydantic_validation_ok = True
        return spec, meta
    except Exception as exc:
        meta.validation_error = f"Schema validation failed: {exc}"
        return None, meta


def extract_spec_output(
    raw_output: str,
    conversation_units: Iterable[ConversationUnit],
) -> tuple[SpecOutput, ExtractionMeta]:
    spec, meta = extract_spec_output_safe(raw_output, conversation_units)
    if spec is None:
        if meta.validation_error:
            raise ExtractionError(meta.validation_error)
        if meta.parse_error:
            raise ExtractionError(meta.parse_error)
        raise ExtractionError("Unknown extraction failure.")
    return spec, meta
