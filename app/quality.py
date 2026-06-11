from __future__ import annotations

import re
from typing import Any, Iterable

from app.schemas import (
    ConstraintItem,
    ConversationUnit,
    EnrichedRequirementItem,
    RequirementItem,
    RequirementQualityChecks,
    SpecOutput,
)
from app.utils import model_dump_compat, model_validate_compat, normalize_text


AMBIGUITY_RISKS = {"low", "medium", "high"}

VAGUE_TESTABILITY_WORDS = (
    "fast",
    "quick",
    "quickly",
    "easy",
    "simple",
    "secure",
    "reliable",
)

MEASURABLE_CONTEXT_HINTS = (
    "within",
    "under",
    "less than",
    "more than",
    "at least",
    "no more than",
    "seconds",
    "second",
    "ms",
    "millisecond",
    "%",
    "percent",
    "p95",
    "p99",
    "uptime",
    "sla",
    "wcag",
    "error rate",
    "response time",
    "load time",
)

ACTOR_HINTS = (
    "customer",
    "customers",
    "user",
    "users",
    "visitor",
    "visitors",
    "staff",
    "admin",
    "admins",
    "administrator",
    "administrators",
    "member",
    "members",
    "officer",
    "officers",
    "student",
    "students",
    "patient",
    "patients",
    "trainer",
    "trainers",
    "tutor",
    "tutors",
    "reception",
    "system",
    "website",
    "site",
    "app",
    "application",
)

ACTION_HINTS = (
    "allow",
    "provide",
    "view",
    "see",
    "reserve",
    "book",
    "choose",
    "select",
    "update",
    "create",
    "manage",
    "approve",
    "reschedule",
    "send",
    "receive",
    "browse",
    "cancel",
    "mark",
    "block",
    "protect",
    "support",
)

GENERIC_ACCEPTANCE_HINTS = (
    "requirement is in scope",
    "related workflow is performed",
    "support this capability",
    "permission to edit get",
    "store the updated a ",
    "given x",
    "when y",
    "then z",
    "minimum standards",
    "appropriate section",
    "additional features",
)


def _unit_map(conversation_units: Iterable[ConversationUnit]) -> dict[str, str]:
    return {unit.id: unit.text for unit in conversation_units}


def _clean_string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    cleaned: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            cleaned.append(text)
    return cleaned


def _valid_source_units(source_units: Iterable[str], unit_text_by_id: dict[str, str]) -> list[str]:
    return [sid for sid in source_units if sid in unit_text_by_id]


def _default_evidence_spans(
    source_units: Iterable[str],
    conversation_units: Iterable[ConversationUnit],
) -> list[str]:
    text_by_id = _unit_map(conversation_units)
    spans: list[str] = []
    for source_id in source_units:
        text = text_by_id.get(source_id, "").strip()
        if text:
            spans.append(text)
    return spans


def _evidence_supported_by_sources(
    evidence_spans: Iterable[str],
    source_units: Iterable[str],
    conversation_units: Iterable[ConversationUnit],
) -> bool:
    unit_text = " ".join(
        unit.text for unit in conversation_units if unit.id in set(source_units)
    )
    normalized_source = normalize_text(unit_text)
    for span in evidence_spans:
        normalized_span = normalize_text(str(span))
        if normalized_span and normalized_span in normalized_source:
            return True
    return False


def _criterion_body(requirement_text: str) -> str:
    body = requirement_text.strip().rstrip(".")
    body = re.sub(
        r"^(the\s+)?(system|website|site|app|application)\s+(shall|should|must)\s+",
        "",
        body,
        flags=re.IGNORECASE,
    ).strip()
    if not body:
        body = "satisfies the stated requirement"
    return body[0].lower() + body[1:] if len(body) > 1 else body.lower()


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in normalize_text(text).split()
        if token
        and token
        not in {
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
            "shall",
            "should",
            "must",
            "system",
            "website",
            "site",
            "app",
            "application",
            "given",
            "when",
            "then",
        }
    }


def _criterion_relevance(requirement_text: str, criterion: str) -> float:
    requirement_tokens = _tokens(requirement_text)
    criterion_tokens = _tokens(criterion)
    if not requirement_tokens or not criterion_tokens:
        return 0.0
    return len(requirement_tokens & criterion_tokens) / max(1, len(requirement_tokens))


def _is_weak_acceptance_criterion(requirement_text: str, criterion: str) -> bool:
    normalized = normalize_text(criterion)
    if not normalized:
        return True
    if any(hint in normalized for hint in GENERIC_ACCEPTANCE_HINTS):
        return True
    has_gwt = all(word in normalized for word in ("given", "when", "then"))
    if not has_gwt:
        return True
    return _criterion_relevance(requirement_text, criterion) < 0.15


def acceptance_criteria_are_weak(
    requirement_text: str,
    acceptance_criteria: Iterable[str],
) -> bool:
    criteria = _clean_string_list(list(acceptance_criteria))
    if not criteria:
        return True
    return any(_is_weak_acceptance_criterion(requirement_text, criterion) for criterion in criteria)


def _clean_action(action: str) -> str:
    action = action.strip().rstrip(".")
    action = re.sub(r"\s+", " ", action)
    return action


def _singular_actor(actor: str) -> str:
    actor = _clean_action(actor)
    normalized = normalize_text(actor)
    if normalized.endswith("staff") or normalized in {"staff", "admin"}:
        return actor
    if normalized.endswith("ies"):
        return actor[:-3] + "y"
    if normalized.endswith("s") and not normalized.endswith("ss"):
        return actor[:-1]
    return actor


def _indefinite_actor(actor: str) -> str:
    actor = _singular_actor(actor)
    normalized = normalize_text(actor)
    if normalized.startswith(("authorized ", "staff", "clinic staff", "office staff")) or normalized.endswith("staff"):
        return actor
    article = "an" if actor[:1].lower() in {"a", "e", "i", "o", "u"} else "a"
    return f"{article} {actor}"


def _definite_actor(actor: str) -> str:
    actor = _singular_actor(actor)
    normalized = normalize_text(actor)
    if normalized.startswith("the "):
        return actor
    return f"the {actor}"


def _be_verb_for_subject(subject: str) -> str:
    normalized = normalize_text(subject)
    if normalized.endswith("staff"):
        return "are"
    first_word = normalized.split()[0] if normalized else ""
    if first_word.endswith("s") and not first_word.endswith("ss"):
        return "are"
    return "is"


def _default_functional_acceptance(requirement_text: str) -> str | None:
    text = requirement_text.strip().rstrip(".")
    normalized = normalize_text(text)

    allow_match = re.search(
        r"allow (?P<actor>[\w\s-]+?) to (?P<action>.+)$",
        text,
        flags=re.IGNORECASE,
    )
    if allow_match:
        actor = allow_match.group("actor").strip()
        action = _clean_action(allow_match.group("action"))
        action_normalized = normalize_text(action)
        actor_subject = _indefinite_actor(actor)
        actor_object = _definite_actor(actor)
        if action_normalized.startswith(("see ", "view ")):
            obj = re.sub(r"^(see|view)\s+", "", action, flags=re.IGNORECASE).strip()
            return (
                f"Given {actor_subject} opens the relevant page, "
                f"When the page loads, Then the system shall display {obj}."
            )
        if "reserve" in action_normalized or "book" in action_normalized:
            return (
                f"Given {actor_subject} selects an available time slot, "
                "When they submit the reservation request, Then the system shall save the reservation with the selected time slot."
            )
        if any(word in action_normalized for word in ("email", "sms", "notification", "notifications", "updates")):
            if "sms" in action_normalized and "email" in action_normalized:
                update_object = "SMS or email update"
            elif "sms" in action_normalized:
                update_object = "SMS update"
            elif "email" in action_normalized:
                update_object = "email update"
            else:
                update_object = "notification update"
            return (
                f"Given {actor_subject} has a relevant request or record with a changed status, "
                f"When the status update is saved, Then the system shall send the configured {update_object} to {actor_object}."
            )
        if any(word in action_normalized for word in ("update", "manage", "edit")):
            obj = re.sub(r"^(update|manage|edit)\s+", "", action, flags=re.IGNORECASE).strip()
            return (
                f"Given {actor_subject} has permission to edit {obj}, "
                f"When they save changes, Then the system shall store the updated {obj}."
            )
        return (
            f"Given {actor_subject} starts the supported workflow, "
            f"When they complete the required input, Then the system shall allow them to {action}."
        )

    provide_dashboard = re.search(
        r"provide a (?P<surface>dashboard|panel|page) for (?P<actor>[\w\s-]+?) to (?P<action>.+)$",
        text,
        flags=re.IGNORECASE,
    )
    if provide_dashboard:
        surface = _clean_action(provide_dashboard.group("surface"))
        actor = _clean_action(provide_dashboard.group("actor"))
        action = _clean_action(provide_dashboard.group("action"))
        return (
            f"Given {actor} open the {surface}, When they {action}, "
            "Then the system shall save the requested operational changes."
        )

    provide_product = re.search(
        r"provide an? (?P<product>web app|web tool|app|application|system|tool|platform|website|site) to (?P<action>.+)$",
        text,
        flags=re.IGNORECASE,
    )
    if provide_product:
        product = _clean_action(provide_product.group("product"))
        action = _clean_action(provide_product.group("action"))
        return (
            f"Given an authorized user opens the {product}, When they need to {action}, "
            f"Then the system shall provide the supported workflow to {action}."
        )

    staff_update = re.search(
        r"(staff|admins?|administrators?) .*(update|manage|edit) (?P<object>.+)$",
        text,
        flags=re.IGNORECASE,
    )
    if staff_update:
        actor = staff_update.group(1).lower()
        obj = _clean_action(staff_update.group("object"))
        return (
            f"Given {actor} access the admin area, When they save changes to {obj}, "
            f"Then the system shall store the updated {obj}."
        )

    send_match = re.search(
        r"shall (send|email) (?P<object>.+?) when (?P<condition>.+)$",
        text,
        flags=re.IGNORECASE,
    )
    if send_match:
        obj = _clean_action(send_match.group("object"))
        condition = _clean_action(send_match.group("condition"))
        return (
            f"Given {condition}, When the triggering condition is detected, "
            f"Then the system shall send {obj} to the intended recipient."
        )

    send_match = re.search(
        r"shall (send|email) (?P<object>.+?)(?: to (?P<actor>.+?))?(?: by (?P<channel>.+))?$",
        text,
        flags=re.IGNORECASE,
    )
    if send_match:
        obj = _clean_action(send_match.group("object"))
        actor = _clean_action(send_match.group("actor") or "the intended recipient")
        channel = _clean_action(send_match.group("channel") or "the configured channel")
        return (
            f"Given {actor} {_be_verb_for_subject(actor)} eligible for a notification, When the triggering event occurs, "
            f"Then the system shall send {obj} by {channel}."
        )

    if "load" in normalized and ("mobile" in normalized or "phone" in normalized):
        return (
            "Given a mobile user opens the relevant page, When page-load performance is tested, "
            "Then the measured mobile load time shall be reported against a defined target."
        )
    if "mobile access" in normalized or (
        ("mobile" in normalized or "phone" in normalized)
        and any(word in normalized for word in ("readable", "complete", "screen", "access"))
    ):
        return (
            "Given a user opens the workflow on a mobile screen, When they complete the primary task, "
            "Then the workflow shall remain readable and usable on the mobile screen."
        )
    if "tablet" in normalized and "dashboard" in normalized:
        return (
            "Given staff open the dashboard on a tablet screen, When they use the main dashboard workflow, "
            "Then the dashboard shall remain usable on the tablet screen."
        )
    if (
        ("not show" in normalized or "not reveal" in normalized or "not display" in normalized)
        and any(word in normalized for word in ("address", "phone number", "email", "personal information", "apartment number"))
    ):
        return (
            "Given one user accesses their own record, When records for other users contain private contact information, "
            "Then the system shall not display another user's private contact information."
        )
    if "ready before" in normalized or "ready by" in normalized:
        return (
            "Given the release plan is reviewed, When the launch milestone is checked, "
            "Then the first version shall be marked ready before the stated deadline."
        )

    return None


def default_acceptance_criteria(requirement_text: str, requirement_type: str = "") -> list[str]:
    specific = _default_functional_acceptance(requirement_text)
    if specific:
        return [specific]

    body = _criterion_body(requirement_text)
    if requirement_type == "constraint":
        return [
            "Given the first release scope is reviewed, "
            f"When scope decisions are checked, Then the constraint is satisfied: {requirement_text.strip().rstrip('.')}."
        ]
    return [
        "Given the requirement is in scope, "
        f"When the related workflow is performed, Then the system shall {body}."
    ]


def has_vague_testability_word(text: str) -> bool:
    normalized = normalize_text(text)
    return any(re.search(rf"\b{re.escape(word)}\b", normalized) for word in VAGUE_TESTABILITY_WORDS)


def has_measurable_context(text: str) -> bool:
    normalized = normalize_text(text)
    if any(hint in normalized for hint in MEASURABLE_CONTEXT_HINTS):
        return True
    return bool(re.search(r"\b\d+(\.\d+)?\b", normalized))


def infer_has_clear_actor(text: str) -> bool:
    normalized = normalize_text(text)
    return any(re.search(rf"\b{re.escape(hint)}\b", normalized) for hint in ACTOR_HINTS)


def infer_is_atomic(text: str) -> bool:
    normalized = normalize_text(text)
    wrapper_actions = {"allow", "provide", "support"}
    action_count = sum(
        1
        for action in ACTION_HINTS
        if action not in wrapper_actions
        and re.search(rf"\b{re.escape(action)}\b", normalized)
    )
    if " and " in normalized and action_count > 1:
        return False
    return True


def infer_is_testable(requirement_text: str, acceptance_criteria: Iterable[str]) -> bool:
    criteria_list = list(acceptance_criteria)
    combined = " ".join([requirement_text, *criteria_list])
    if has_vague_testability_word(combined) and not has_measurable_context(combined):
        return False
    return bool(str(requirement_text).strip() and criteria_list)


def infer_ambiguity_risk(
    requirement_text: str,
    evidence_spans: Iterable[str],
    acceptance_criteria: Iterable[str],
) -> str:
    evidence_list = list(evidence_spans)
    criteria_list = list(acceptance_criteria)
    combined = " ".join([requirement_text, *criteria_list])
    if not evidence_list:
        return "high"
    if not criteria_list:
        return "medium"
    if has_vague_testability_word(combined) and not has_measurable_context(combined):
        return "medium"
    return "low"


def infer_quality_checks(
    *,
    requirement_text: str,
    source_units: Iterable[str],
    evidence_spans: Iterable[str],
    acceptance_criteria: Iterable[str],
) -> RequirementQualityChecks:
    evidence_list = list(evidence_spans)
    criteria_list = list(acceptance_criteria)
    return RequirementQualityChecks(
        is_atomic=infer_is_atomic(requirement_text),
        is_testable=infer_is_testable(requirement_text, criteria_list),
        has_clear_actor=infer_has_clear_actor(requirement_text),
        has_traceable_evidence=bool(list(source_units) and evidence_list),
        ambiguity_risk=infer_ambiguity_risk(requirement_text, evidence_list, criteria_list),
    )


def coerce_quality_checks(
    raw_checks: Any,
    *,
    requirement_text: str,
    source_units: Iterable[str],
    evidence_spans: Iterable[str],
    acceptance_criteria: Iterable[str],
) -> RequirementQualityChecks:
    inferred = infer_quality_checks(
        requirement_text=requirement_text,
        source_units=source_units,
        evidence_spans=evidence_spans,
        acceptance_criteria=acceptance_criteria,
    )
    if not isinstance(raw_checks, dict):
        return inferred

    risk = str(raw_checks.get("ambiguity_risk", inferred.ambiguity_risk)).strip().lower()
    if risk not in AMBIGUITY_RISKS:
        risk = inferred.ambiguity_risk

    def _as_bool(key: str, default: bool) -> bool:
        value = raw_checks.get(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "yes", "1"}:
                return True
            if lowered in {"false", "no", "0"}:
                return False
        return bool(default)

    has_traceable_evidence = _as_bool(
        "has_traceable_evidence",
        inferred.has_traceable_evidence,
    )
    if has_traceable_evidence and not list(evidence_spans):
        has_traceable_evidence = False
        risk = "high"

    return RequirementQualityChecks(
        is_atomic=_as_bool("is_atomic", inferred.is_atomic),
        is_testable=_as_bool("is_testable", inferred.is_testable),
        has_clear_actor=_as_bool("has_clear_actor", inferred.has_clear_actor),
        has_traceable_evidence=has_traceable_evidence,
        ambiguity_risk=risk,
    )


def build_enriched_item_fallback(
    item: Any,
    conversation_units: Iterable[ConversationUnit],
) -> EnrichedRequirementItem:
    unit_text_by_id = _unit_map(conversation_units)
    source_units = _valid_source_units(
        [str(source_id).strip() for source_id in getattr(item, "source_units", []) if str(source_id).strip()],
        unit_text_by_id,
    )
    evidence_spans = _default_evidence_spans(source_units, conversation_units)
    acceptance_criteria = default_acceptance_criteria(
        str(getattr(item, "text", "")).strip(),
        str(getattr(item, "type", "")).strip(),
    )
    quality_checks = infer_quality_checks(
        requirement_text=str(getattr(item, "text", "")).strip(),
        source_units=source_units,
        evidence_spans=evidence_spans,
        acceptance_criteria=acceptance_criteria,
    )
    return EnrichedRequirementItem(
        id=str(getattr(item, "id", "")).strip(),
        type=str(getattr(item, "type", "")).strip(),
        text=str(getattr(item, "text", "")).strip(),
        source_units=source_units,
        evidence_spans=evidence_spans,
        acceptance_criteria=acceptance_criteria,
        quality_checks=quality_checks,
    )


def apply_quality_defaults_to_item(
    item: RequirementItem | ConstraintItem,
    conversation_units: Iterable[ConversationUnit],
    requirement_type: str = "",
) -> RequirementItem | ConstraintItem:
    evidence_spans = _clean_string_list(getattr(item, "evidence_spans", []))
    if not evidence_spans or not _evidence_supported_by_sources(
        evidence_spans,
        item.source_units,
        conversation_units,
    ):
        evidence_spans = _default_evidence_spans(item.source_units, conversation_units)
    acceptance_criteria = _clean_string_list(getattr(item, "acceptance_criteria", []))
    if not acceptance_criteria or acceptance_criteria_are_weak(item.text, acceptance_criteria):
        acceptance_criteria = default_acceptance_criteria(item.text, requirement_type)

    quality_checks = getattr(item, "quality_checks", RequirementQualityChecks())
    default_quality = RequirementQualityChecks()
    if model_dump_compat(quality_checks) == model_dump_compat(default_quality):
        quality_checks = infer_quality_checks(
            requirement_text=item.text,
            source_units=item.source_units,
            evidence_spans=evidence_spans,
            acceptance_criteria=acceptance_criteria,
        )
    elif quality_checks.has_traceable_evidence and not evidence_spans:
        quality_checks = RequirementQualityChecks(
            is_atomic=quality_checks.is_atomic,
            is_testable=quality_checks.is_testable,
            has_clear_actor=quality_checks.has_clear_actor,
            has_traceable_evidence=False,
            ambiguity_risk="high",
        )

    payload = model_dump_compat(item)
    payload["evidence_spans"] = evidence_spans
    payload["acceptance_criteria"] = acceptance_criteria
    payload["quality_checks"] = model_dump_compat(quality_checks)
    return model_validate_compat(type(item), payload)


def ensure_spec_quality_defaults(
    spec: SpecOutput,
    conversation_units: Iterable[ConversationUnit],
) -> SpecOutput:
    payload = model_dump_compat(spec)
    payload["functional_requirements"] = [
        model_dump_compat(apply_quality_defaults_to_item(item, conversation_units, "functional_requirement"))
        for item in spec.functional_requirements
    ]
    payload["non_functional_requirements"] = [
        model_dump_compat(apply_quality_defaults_to_item(item, conversation_units, "non_functional_requirement"))
        for item in spec.non_functional_requirements
    ]
    payload["constraints"] = [
        model_dump_compat(apply_quality_defaults_to_item(item, conversation_units, "constraint"))
        for item in spec.constraints
    ]
    return model_validate_compat(SpecOutput, payload)


def validate_requirement_quality_items(
    items: Iterable[RequirementItem | ConstraintItem],
    conversation_units: Iterable[ConversationUnit],
    *,
    category_label: str,
) -> list[str]:
    unit_text_by_id = _unit_map(conversation_units)
    warnings: list[str] = []
    for item in items:
        item_label = f"{category_label}:{item.id}"
        missing_sources = [sid for sid in item.source_units if sid not in unit_text_by_id]
        if missing_sources:
            warnings.append(
                f"{item_label} references unknown source_units: {', '.join(missing_sources)}"
            )

        evidence_spans = _clean_string_list(getattr(item, "evidence_spans", []))
        acceptance_criteria = _clean_string_list(getattr(item, "acceptance_criteria", []))
        quality_checks = getattr(item, "quality_checks", RequirementQualityChecks())

        if quality_checks.has_traceable_evidence and not evidence_spans:
            warnings.append(f"{item_label} is marked traceable but has no evidence spans.")
        if not evidence_spans:
            warnings.append(f"{item_label} has no evidence spans.")
        if not acceptance_criteria:
            warnings.append(f"{item_label} has no acceptance criteria.")
        for index, criterion in enumerate(getattr(item, "acceptance_criteria", []), start=1):
            if not str(criterion).strip():
                warnings.append(f"{item_label} has an empty acceptance criterion at index {index}.")

        risk = str(getattr(quality_checks, "ambiguity_risk", "")).strip().lower()
        if risk not in AMBIGUITY_RISKS:
            warnings.append(f"{item_label} has invalid ambiguity_risk: {risk or '<empty>'}.")

        combined_text = " ".join([item.text, *acceptance_criteria])
        if (
            quality_checks.is_testable
            and has_vague_testability_word(combined_text)
            and not has_measurable_context(combined_text)
        ):
            warnings.append(
                f"{item_label} is marked testable but uses vague quality wording without measurable context."
            )
    return sorted(set(warnings))


def validate_spec_quality(
    spec: SpecOutput,
    conversation_units: Iterable[ConversationUnit],
) -> list[str]:
    warnings: list[str] = []
    warnings.extend(
        validate_requirement_quality_items(
            spec.functional_requirements,
            conversation_units,
            category_label="functional_requirements",
        )
    )
    warnings.extend(
        validate_requirement_quality_items(
            spec.non_functional_requirements,
            conversation_units,
            category_label="non_functional_requirements",
        )
    )
    warnings.extend(
        validate_requirement_quality_items(
            spec.constraints,
            conversation_units,
            category_label="constraints",
        )
    )
    return sorted(set(warnings))
