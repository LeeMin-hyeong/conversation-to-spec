from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable

from app.quality import acceptance_criteria_are_weak, default_acceptance_criteria, infer_quality_checks
from app.schemas import ConstraintItem, ConversationUnit, QuestionItem, RequirementItem, SpecOutput
from app.utils import model_dump_compat, model_validate_compat, normalize_text


LOW_CONFIDENCE_THRESHOLD = 0.5
MAX_LLM_DIAGNOSES = 6

ISSUE_TYPES = {
    "actor_resolution",
    "privacy_prohibition",
    "answered_question",
    "deferred_scope",
    "weak_acceptance_criteria",
    "unsupported",
    "ok",
}

REQUIREMENT_TYPES = {
    "create",
    "approve_cancel",
    "browse_reserve",
    "export_csv",
    "view_list",
    "privacy_prohibition",
    "deferred_scope",
    "other",
}


@dataclass
class PostprocessResult:
    spec: SpecOutput
    warnings: list[str]
    num_llm_calls: int = 0
    changed: bool = False


@dataclass
class ItemDiagnosis:
    issue_type: str = "ok"
    actor: str = ""
    requirement_type: str = "other"
    should_repair: bool = False


def _unit_map(conversation_units: Iterable[ConversationUnit]) -> dict[str, str]:
    return {unit.id: unit.text for unit in conversation_units}


def _source_text(item: RequirementItem | ConstraintItem, unit_texts: dict[str, str]) -> str:
    texts = [unit_texts.get(source_id, "") for source_id in item.source_units]
    return " ".join(text.strip() for text in texts if text.strip())


def _evidence_text(item: RequirementItem | ConstraintItem, unit_texts: dict[str, str]) -> str:
    evidence = " ".join(str(span).strip() for span in item.evidence_spans if str(span).strip())
    return evidence or _source_text(item, unit_texts)


def _is_low_confidence(item: RequirementItem | ConstraintItem) -> bool:
    confidence = item.verification.confidence
    return confidence is not None and confidence < LOW_CONFIDENCE_THRESHOLD


def _has_warning(item: RequirementItem | ConstraintItem, warning: str) -> bool:
    return warning in set(item.verification.warnings)


def _needs_diagnosis(item: RequirementItem | ConstraintItem) -> bool:
    if _is_low_confidence(item):
        return True
    if _has_warning(item, "weak_acceptance_criteria"):
        return True
    if not item.quality_checks.has_clear_actor:
        return True
    return False


def _parse_llm_json(raw_output: str) -> dict[str, Any] | None:
    text = re.sub(r"<think>.*?</think>", "", str(raw_output), flags=re.DOTALL).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(text[start : end + 1])
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _coerce_diagnosis(payload: dict[str, Any] | None) -> ItemDiagnosis:
    if not payload:
        return ItemDiagnosis()
    issue_type = str(payload.get("issue_type", "ok")).strip().lower()
    requirement_type = str(payload.get("requirement_type", "other")).strip().lower()
    if issue_type not in ISSUE_TYPES:
        issue_type = "ok"
    if requirement_type not in REQUIREMENT_TYPES:
        requirement_type = "other"
    actor = re.sub(r"\s+", " ", str(payload.get("actor", "")).strip())
    should_repair = bool(payload.get("should_repair", issue_type != "ok"))
    return ItemDiagnosis(
        issue_type=issue_type,
        actor=actor,
        requirement_type=requirement_type,
        should_repair=should_repair,
    )


def _diagnosis_prompt(
    *,
    category: str,
    item: RequirementItem | ConstraintItem,
    evidence_text: str,
    previous_source_text: str,
    next_source_text: str,
) -> str:
    schema = {
        "issue_type": (
            "actor_resolution | privacy_prohibition | answered_question | deferred_scope | "
            "weak_acceptance_criteria | unsupported | ok"
        ),
        "actor": "one actor noun phrase if a pronoun should be resolved, else empty string",
        "requirement_type": (
            "create | approve_cancel | browse_reserve | export_csv | view_list | "
            "privacy_prohibition | deferred_scope | other"
        ),
        "should_repair": True,
    }
    return (
        "You diagnose one requirement item. Do not rewrite the full specification.\n"
        "Choose labels only. Prefer conservative labels supported by the evidence.\n"
        "Return JSON only. Do not include markdown, comments, or reasoning.\n\n"
        f"Category: {category}\n"
        f"Requirement: {item.text}\n"
        f"Acceptance criteria: {json.dumps(item.acceptance_criteria, ensure_ascii=False)}\n"
        f"MiniCheck confidence: {item.verification.confidence}\n"
        f"Warnings: {json.dumps(item.verification.warnings, ensure_ascii=False)}\n"
        f"Previous source text: {previous_source_text}\n"
        f"Evidence/source text: {evidence_text}\n"
        f"Next source text: {next_source_text}\n\n"
        f"Required JSON schema:\n{json.dumps(schema, ensure_ascii=False, indent=2)}\n"
        "Output JSON only.\n/no_think"
    )


def _llm_diagnose(
    *,
    runner: Any | None,
    generation_config: dict[str, Any],
    category: str,
    item: RequirementItem | ConstraintItem,
    evidence_text: str,
    previous_source_text: str,
    next_source_text: str,
) -> tuple[ItemDiagnosis, int]:
    if runner is None:
        return ItemDiagnosis(), 0
    prompt = _diagnosis_prompt(
        category=category,
        item=item,
        evidence_text=evidence_text,
        previous_source_text=previous_source_text,
        next_source_text=next_source_text,
    )
    config = dict(generation_config or {})
    config["max_new_tokens"] = min(int(config.get("max_new_tokens", 900)), 220)
    config["temperature"] = 0.0
    try:
        raw_output = runner.generate(prompt, config)
    except Exception:
        return ItemDiagnosis(), 0
    return _coerce_diagnosis(_parse_llm_json(raw_output)), 1


def _fallback_requirement_type(text: str, evidence_text: str) -> str:
    normalized = normalize_text(f"{text} {evidence_text}")
    if _is_privacy_prohibition(normalized):
        return "privacy_prohibition"
    if _is_deferred_scope(normalized):
        return "deferred_scope"
    if "csv" in normalized and "export" in normalized:
        return "export_csv"
    if any(word in normalized for word in ("approve", "reject", "confirm", "cancel")):
        return "approve_cancel"
    if any(word in normalized for word in ("browse", "reserve", "book", "sign up", "signup")):
        return "browse_reserve"
    if any(word in normalized for word in ("daily", "list", "roster", "see", "view")):
        return "view_list"
    if any(word in normalized for word in ("create", "add", "register")):
        return "create"
    return "other"


def _is_privacy_prohibition(normalized_text: str) -> bool:
    has_negative = any(
        phrase in normalized_text
        for phrase in ("not collect", "not store", "must not collect", "must not store", "should not collect", "should not store", "do not collect", "do not store")
    )
    has_sensitive = any(
        term in normalized_text
        for term in (
            "national id",
            "resident registration",
            "credit card",
            "payment card",
            "card number",
            "card details",
            "medical history",
            "personal information",
            "sensitive",
        )
    )
    return has_negative and has_sensitive


def _is_deferred_scope(normalized_text: str) -> bool:
    return (
        any(phrase in normalized_text for phrase in ("first release does not need", "not need", "not needed", "not included", "deferred"))
        and any(word in normalized_text for word in ("later", "future", "release", "launch"))
    )


def _extract_prohibited_object(evidence_text: str) -> str:
    text = evidence_text.strip().rstrip(".")
    match = re.search(
        r"\b(?:must|should|shall|do)\s+not\s+(?:collect|store|retain|save)\s+(?P<object>.+)$",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group("object").strip().rstrip(".")
    match = re.search(
        r"\bnot\s+(?:collect|store|retain|save)\s+(?P<object>.+)$",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group("object").strip().rstrip(".")
    return text


def _normalize_privacy_text(evidence_text: str) -> str:
    obj = _extract_prohibited_object(evidence_text)
    obj = re.sub(r"^(we|the system)\s+", "", obj, flags=re.IGNORECASE).strip()
    return f"The system shall not collect or store {obj}."


def _extract_deferred_feature(evidence_text: str, item_text: str) -> str:
    combined = f"{evidence_text} {item_text}"
    patterns = [
        r"does not need (?P<feature>.+?)(?:\.|,|;|$)",
        r"do not need (?P<feature>.+?)(?:\.|,|;|$)",
        r"not include (?P<feature>.+?)(?:\.|,|;|$)",
        r"(?P<feature>.+?) shall be deferred",
    ]
    for pattern in patterns:
        match = re.search(pattern, combined, flags=re.IGNORECASE)
        if match:
            feature = match.group("feature").strip()
            feature = re.sub(r"^(the|a|an)\s+", "", feature, flags=re.IGNORECASE)
            return feature
    return item_text.strip().rstrip(".")


def _normalize_deferred_constraint(evidence_text: str, item_text: str) -> str:
    feature = _extract_deferred_feature(evidence_text, item_text)
    return f"{feature[:1].upper() + feature[1:]} shall be deferred beyond the first release."


def _source_number(source_id: str) -> int | None:
    match = re.match(r"U(?P<num>\d+)$", source_id)
    if not match:
        return None
    return int(match.group("num"))


def _neighbor_texts(item: RequirementItem | ConstraintItem, unit_texts: dict[str, str]) -> tuple[str, str]:
    nums = [_source_number(source_id) for source_id in item.source_units]
    nums = [num for num in nums if num is not None]
    if not nums:
        return "", ""
    prev_text = unit_texts.get(f"U{min(nums) - 1}", "")
    next_text = unit_texts.get(f"U{max(nums) + 1}", "")
    return prev_text, next_text


def _actor_from_question(text: str) -> str:
    patterns = [
        r"what should (?P<actor>[\w\s-]+?) be able to",
        r"what should (?P<actor>[\w\s-]+?) do",
        r"what can (?P<actor>[\w\s-]+?) do",
        r"how should (?P<actor>[\w\s-]+?) ",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return re.sub(r"\s+", " ", match.group("actor").strip())
    return ""


def _actor_from_context(text: str) -> str:
    actor = _actor_from_question(text)
    if actor:
        return actor
    match = re.match(
        r"^(?P<actor>[A-Z][A-Za-z\s-]{1,40}?)\s+(?:also\s+)?(?:need|needs|should|shall|must|can)\b",
        text.strip(),
    )
    if match:
        return re.sub(r"\s+", " ", match.group("actor").strip())
    return ""


def _resolve_pronoun_text(text: str, actor: str) -> str:
    if not actor:
        return text
    resolved = re.sub(r"^(they|them|their)\b", actor, text.strip(), flags=re.IGNORECASE)
    resolved = re.sub(
        r"^(?P<prefix>after [^,]+,\s*)they shall\b",
        rf"\g<prefix>{actor} shall",
        resolved,
        flags=re.IGNORECASE,
    )
    resolved = re.sub(r"\bthey shall\b", f"{actor} shall", resolved, flags=re.IGNORECASE)
    resolved = re.sub(r"\bthey should\b", f"{actor} shall", resolved, flags=re.IGNORECASE)
    resolved = re.sub(r"\bthey need to\b", f"{actor} shall", resolved, flags=re.IGNORECASE)
    return resolved[:1].upper() + resolved[1:] if resolved else resolved


def _usable_actor(actor: str) -> str:
    actor = re.sub(r"\s+", " ", str(actor or "").strip())
    normalized = normalize_text(actor)
    if not actor or normalized in {"none", "n/a", "na", "unknown", "they", "them", "their", "it"}:
        return ""
    return actor


def _specific_acceptance_criteria(
    text: str,
    requirement_type: str,
    category: str,
) -> list[str]:
    normalized = normalize_text(text)
    if requirement_type == "create":
        return [
            "Given an authorized user enters all required fields, When they save the new record, Then the system shall create the record with the submitted values."
        ]
    if requirement_type == "approve_cancel":
        return [
            "Given a pending reservation or signup exists, When an authorized user confirms, approves, rejects, or cancels it, Then the system shall update the record status accordingly."
        ]
    if requirement_type == "browse_reserve":
        return [
            "Given open slots are available, When a user browses the list and selects one open slot, Then the system shall save the reservation for the selected slot."
        ]
    if requirement_type == "export_csv":
        return [
            "Given confirmed or approved records exist, When an authorized user exports the list, Then the system shall provide a CSV file containing those records."
        ]
    if requirement_type == "view_list":
        return [
            "Given confirmed or approved records exist, When an authorized user opens the relevant list, Then the system shall display the records grouped by the stated context."
        ]
    if requirement_type == "privacy_prohibition":
        return [
            "Given user data is submitted, When the system validates or stores the data, Then the prohibited sensitive data shall not be collected or stored."
        ]
    if requirement_type == "deferred_scope":
        return [
            "Given the first release scope is reviewed, When release contents are checked, Then the deferred capability shall be excluded from the first release."
        ]
    if "mobile" in normalized or "phone" in normalized:
        return default_acceptance_criteria(text, "non_functional_requirement")
    return default_acceptance_criteria(text, "constraint" if category == "constraints" else "functional_requirement")


def _refresh_quality(item_payload: dict[str, Any]) -> dict[str, Any]:
    item_payload["quality_checks"] = model_dump_compat(
        infer_quality_checks(
            requirement_text=str(item_payload.get("text", "")),
            source_units=item_payload.get("source_units", []),
            evidence_spans=item_payload.get("evidence_spans", []),
            acceptance_criteria=item_payload.get("acceptance_criteria", []),
        )
    )
    return item_payload


def _apply_item_edit(
    *,
    category: str,
    raw_item: dict[str, Any],
    item: RequirementItem | ConstraintItem,
    diagnosis: ItemDiagnosis,
    evidence_text: str,
    previous_source_text: str,
) -> tuple[str, dict[str, Any], str | None]:
    normalized = normalize_text(f"{item.text} {evidence_text}")
    requirement_type = diagnosis.requirement_type
    if requirement_type == "other":
        requirement_type = _fallback_requirement_type(item.text, evidence_text)

    issue_type = diagnosis.issue_type
    if issue_type == "ok":
        if requirement_type == "privacy_prohibition":
            issue_type = "privacy_prohibition"
        elif requirement_type == "deferred_scope":
            issue_type = "deferred_scope"
        elif _has_warning(item, "weak_acceptance_criteria"):
            issue_type = "weak_acceptance_criteria"

    new_category = category
    new_item = dict(raw_item)
    warning: str | None = None

    if issue_type == "privacy_prohibition" or _is_privacy_prohibition(normalized):
        new_item["text"] = _normalize_privacy_text(evidence_text)
        new_item["acceptance_criteria"] = _specific_acceptance_criteria(
            new_item["text"], "privacy_prohibition", "non_functional_requirements"
        )
        new_category = "non_functional_requirements"
        warning = f"{category}:{item.id} confidence_postprocess_privacy_prohibition"
    elif issue_type == "deferred_scope" or _is_deferred_scope(normalized):
        new_item["text"] = _normalize_deferred_constraint(evidence_text, item.text)
        new_item["acceptance_criteria"] = _specific_acceptance_criteria(
            new_item["text"], "deferred_scope", "constraints"
        )
        new_category = "constraints"
        warning = f"{category}:{item.id} confidence_postprocess_deferred_scope"
    else:
        actor = _usable_actor(diagnosis.actor) or _actor_from_context(previous_source_text)
        if issue_type == "actor_resolution" or (
            actor and not item.quality_checks.has_clear_actor
        ):
            resolved = _resolve_pronoun_text(str(new_item.get("text", "")), actor)
            if resolved != new_item.get("text"):
                new_item["text"] = resolved
                warning = f"{category}:{item.id} confidence_postprocess_actor_resolution"
        if (
            _has_warning(item, "weak_acceptance_criteria")
            or acceptance_criteria_are_weak(str(new_item.get("text", "")), new_item.get("acceptance_criteria", []))
            or issue_type == "weak_acceptance_criteria"
            or _is_low_confidence(item)
        ):
            new_item["acceptance_criteria"] = _specific_acceptance_criteria(
                str(new_item.get("text", "")), requirement_type, category
            )
            warning = warning or f"{category}:{item.id} confidence_postprocess_acceptance_criteria"

    return new_category, _refresh_quality(new_item), warning


def _dedupe_answered_questions(
    *,
    open_questions: list[dict[str, Any]],
    follow_up_questions: list[dict[str, Any]],
    covered_sources: set[str],
    unit_texts: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    answered_sources: set[str] = set()
    for question in open_questions:
        sources = [str(source_id) for source_id in question.get("source_units", [])]
        if len(sources) != 1:
            continue
        source_id = sources[0]
        source_text = unit_texts.get(source_id, "")
        question_text = str(question.get("text", ""))
        normalized_question = normalize_text(question_text)
        normalized_source = normalize_text(source_text)
        if (
            source_id in covered_sources
            and ("included in the first release" in normalized_question or "included in first release" in normalized_question)
            and ("deferred" in normalized_question or "later" in normalized_source or "does not need" in normalized_source)
        ):
            answered_sources.add(source_id)
            continue
        if not source_text.strip().endswith("?"):
            continue
        number = _source_number(source_id)
        if number is None:
            continue
        if any(f"U{idx}" in covered_sources for idx in range(number + 1, number + 4)):
            answered_sources.add(source_id)

    if not answered_sources:
        return open_questions, follow_up_questions, 0

    def keep(question: dict[str, Any]) -> bool:
        sources = {str(source_id) for source_id in question.get("source_units", [])}
        return not bool(sources & answered_sources)

    return (
        [question for question in open_questions if keep(question)],
        [question for question in follow_up_questions if keep(question)],
        len(answered_sources),
    )


def _reindex_items(items: list[dict[str, Any]], prefix: str) -> list[dict[str, Any]]:
    reindexed: list[dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
        updated = dict(item)
        updated["id"] = f"{prefix}{index}"
        reindexed.append(updated)
    return reindexed


def confidence_aware_postprocess(
    spec: SpecOutput,
    conversation_units: Iterable[ConversationUnit],
    *,
    runner: Any | None = None,
    generation_config: dict[str, Any] | None = None,
) -> PostprocessResult:
    unit_texts = _unit_map(conversation_units)
    payload = model_dump_compat(spec)
    warnings: list[str] = []
    changed = False
    num_llm_calls = 0
    diagnoses_used = 0

    new_sections: dict[str, list[dict[str, Any]]] = {
        "functional_requirements": [],
        "non_functional_requirements": [],
        "constraints": [],
    }

    for category, item_cls in (
        ("functional_requirements", RequirementItem),
        ("non_functional_requirements", RequirementItem),
        ("constraints", ConstraintItem),
    ):
        for raw_item in payload.get(category, []):
            item = model_validate_compat(item_cls, raw_item)
            if not _needs_diagnosis(item):
                new_sections[category].append(raw_item)
                continue

            evidence = _evidence_text(item, unit_texts)
            previous_text, next_text = _neighbor_texts(item, unit_texts)
            if diagnoses_used < MAX_LLM_DIAGNOSES:
                diagnosis, calls = _llm_diagnose(
                    runner=runner,
                    generation_config=generation_config or {},
                    category=category,
                    item=item,
                    evidence_text=evidence,
                    previous_source_text=previous_text,
                    next_source_text=next_text,
                )
                num_llm_calls += calls
                diagnoses_used += calls
            else:
                diagnosis = ItemDiagnosis()

            new_category, edited_item, warning = _apply_item_edit(
                category=category,
                raw_item=raw_item,
                item=item,
                diagnosis=diagnosis,
                evidence_text=evidence,
                previous_source_text=previous_text,
            )
            if edited_item != raw_item or new_category != category:
                changed = True
            if warning:
                warnings.append(warning)
            new_sections[new_category].append(edited_item)

    payload["functional_requirements"] = _reindex_items(new_sections["functional_requirements"], "FR")
    payload["non_functional_requirements"] = _reindex_items(new_sections["non_functional_requirements"], "NFR")
    payload["constraints"] = _reindex_items(new_sections["constraints"], "CON")

    covered_sources = {
        str(source_id)
        for section in ("functional_requirements", "non_functional_requirements", "constraints")
        for item in payload.get(section, [])
        for source_id in item.get("source_units", [])
    }
    open_questions, follow_ups, removed = _dedupe_answered_questions(
        open_questions=list(payload.get("open_questions", [])),
        follow_up_questions=list(payload.get("follow_up_questions", [])),
        covered_sources=covered_sources,
        unit_texts=unit_texts,
    )
    if removed:
        payload["open_questions"] = open_questions
        payload["follow_up_questions"] = follow_ups
        warnings.append(f"open_questions:confidence_postprocess_removed_answered:{removed}")
        changed = True

    if warnings:
        existing_warnings = [
            str(warning)
            for warning in payload.get("verification_warnings", [])
            if not str(warning).startswith(
                (
                    "functional_requirements:",
                    "non_functional_requirements:",
                    "constraints:",
                )
            )
        ]
        payload["verification_warnings"] = sorted(
            set([*existing_warnings, *warnings])
        )

    return PostprocessResult(
        spec=model_validate_compat(SpecOutput, payload),
        warnings=sorted(set(warnings)),
        num_llm_calls=num_llm_calls,
        changed=changed,
    )
