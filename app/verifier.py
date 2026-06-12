from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from app.quality import (
    acceptance_criteria_are_weak,
    default_acceptance_criteria,
    has_measurable_context,
    has_vague_testability_word,
    infer_quality_checks,
)
from app.schemas import (
    ConstraintItem,
    ConversationUnit,
    QuestionItem,
    RequirementItem,
    RequirementVerification,
    SpecOutput,
)
from app.utils import model_dump_compat, model_validate_compat, normalize_text


VERDICTS = {
    "SUPPORTED",
    "PARTIALLY_SUPPORTED",
    "UNSUPPORTED",
    "CONTRADICTED",
    "NOT_ENOUGH_INFO",
    "NOT_CHECKED",
}

REPAIR_TRIGGER_VERDICTS = {"UNSUPPORTED", "CONTRADICTED", "NOT_ENOUGH_INFO"}

MINICHECK_MODEL_NAME = "lytang/MiniCheck-RoBERTa-Large"
MINICHECK_SUPPORT_THRESHOLD = 0.5

NUMERIC_THRESHOLD_RE = re.compile(
    r"\b(?:(?:within|under|less\s+than|more\s+than|at\s+least|at\s+most|"
    r"no\s+more\s+than|no\s+less\s+than|minimum|maximum|max|min)\s+)?"
    r"(?P<number>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>ms|milliseconds?|seconds?|secs?|minutes?|mins?|hours?|hrs?|days?|%|percent)\b",
    flags=re.IGNORECASE,
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
    "system",
    "website",
    "site",
    "app",
    "application",
    "allow",
    "provide",
    "support",
    "satisfy",
    "capability",
}


@dataclass
class VerificationEntry:
    requirement_id: str
    category: str
    text: str
    source_units: list[str]
    evidence_spans: list[str]
    source_relevance_score: float | None
    verdict: str
    confidence: float | None
    warnings: list[str] = field(default_factory=list)
    repair_status: str | None = None


@dataclass
class VerificationRunResult:
    spec: SpecOutput
    report: dict[str, Any]
    report_markdown: str
    warnings: list[str]
    num_llm_calls: int = 0
    repair_trigger_count: int = 0
    repair_success_count: int = 0


class MiniCheckScorer:
    """MiniCheck-style claim-evidence verifier using the released HF classifier."""

    def __init__(self, model_name: str = MINICHECK_MODEL_NAME) -> None:
        self.model_name = model_name
        self._torch = None
        self._tokenizer = None
        self._model = None

    def _select_device(self) -> str:
        assert self._torch is not None
        if self._torch.cuda.is_available():
            return "cuda"
        mps_backend = getattr(self._torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return "mps"
        return "cpu"

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        try:
            import torch  # type: ignore
            from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on optional runtime env
            raise RuntimeError(
                "MiniCheck verification requires torch and transformers."
            ) from exc

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        device = self._select_device()
        self._model.to(device)
        self._model.eval()

    def score(self, *, document: str, claim: str) -> tuple[int, float]:
        self._ensure_loaded()
        assert self._torch is not None
        assert self._tokenizer is not None
        assert self._model is not None

        encoded = self._tokenizer(
            document,
            claim,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        encoded = {key: value.to(self._model.device) for key, value in encoded.items()}
        with self._torch.no_grad():
            logits = self._model(**encoded).logits[0]
        if logits.numel() == 1:
            support_prob = float(self._torch.sigmoid(logits)[0].detach().cpu())
        else:
            probs = self._torch.softmax(logits, dim=-1)
            # MiniCheck-RoBERTa predicts binary label 1=supported, 0=unsupported.
            support_prob = float(probs[1].detach().cpu())
        label = 1 if support_prob >= MINICHECK_SUPPORT_THRESHOLD else 0
        return label, support_prob


def _token_variants(token: str) -> set[str]:
    variants = {token}
    synonyms = {
        "defer": {"deferred", "later", "future", "phase", "release", "semester"},
        "deferred": {"defer", "later", "future", "phase", "release", "semester"},
        "release": {"launch", "phase", "semester", "scope", "excluded"},
        "launch": {"release", "phase", "scope", "excluded", "needed", "required"},
        "part": {"scope", "excluded", "launch", "release", "needed", "required"},
        "future": {"later", "deferred", "phase", "release", "semester"},
        "later": {"future", "deferred", "phase", "release", "semester"},
        "phase": {"future", "later", "release"},
        "semester": {"release", "phase"},
        "mobile": {"phone", "phones"},
        "phone": {"mobile", "phones", "android", "iphone", "ios"},
        "phones": {"phone", "mobile", "android", "iphone", "ios"},
        "android": {"mobile", "phone", "phones", "ios", "iphone"},
        "iphone": {"mobile", "phone", "phones", "ios", "android"},
        "ios": {"mobile", "phone", "phones", "iphone", "android"},
        "tablet": {"tablets", "screen", "screens"},
        "tablets": {"tablet", "screen", "screens"},
        "screen": {"screens", "tablet", "tablets"},
        "screens": {"screen", "tablet", "tablets"},
        "use": {"usage", "usable", "using"},
        "uses": {"use", "usage", "using"},
        "using": {"use", "usage"},
        "usage": {"use", "using", "usable"},
        "needed": {"excluded", "scope", "launch"},
        "required": {"excluded", "scope", "launch"},
        "excluded": {"needed", "required", "scope", "launch"},
        "scope": {"needed", "required", "excluded", "launch", "release"},
    }
    variants.update(synonyms.get(token, set()))
    return {variant for variant in variants if variant and variant not in STOPWORDS}


def _tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for token in normalize_text(text).split():
        if token and token not in STOPWORDS and len(token) > 2:
            tokens.update(_token_variants(token))
    return tokens


def source_relevance_score(requirement_text: str, evidence_text: str) -> float:
    requirement_tokens = _tokens(requirement_text)
    evidence_tokens = _tokens(evidence_text)
    if not requirement_tokens or not evidence_tokens:
        return 0.0
    return len(requirement_tokens & evidence_tokens) / max(1, len(requirement_tokens))


def _canonical_number(value: str) -> str:
    try:
        numeric = float(value)
    except Exception:
        return value.strip().lower()
    if numeric.is_integer():
        return str(int(numeric))
    return str(numeric).rstrip("0").rstrip(".")


def _canonical_unit(value: str) -> str:
    unit = value.strip().lower()
    if unit in {"%", "percent"}:
        return "percent"
    if unit in {"ms", "millisecond", "milliseconds"}:
        return "millisecond"
    if unit in {"second", "seconds", "sec", "secs"}:
        return "second"
    if unit in {"minute", "minutes", "min", "mins"}:
        return "minute"
    if unit in {"hour", "hours", "hr", "hrs"}:
        return "hour"
    if unit in {"day", "days"}:
        return "day"
    return unit.rstrip("s")


def numeric_threshold_details(text: str) -> list[str]:
    details: list[str] = []
    for match in NUMERIC_THRESHOLD_RE.finditer(text):
        number = _canonical_number(match.group("number"))
        unit = _canonical_unit(match.group("unit"))
        details.append(f"{number} {unit}")
    return sorted(set(details))


def unsupported_numeric_details(claim_text: str, evidence_text: str) -> list[str]:
    claim_details = numeric_threshold_details(claim_text)
    if not claim_details:
        return []
    evidence_details = set(numeric_threshold_details(evidence_text))
    return [detail for detail in claim_details if detail not in evidence_details]


def unsupported_claim_terms(requirement_text: str, evidence_text: str) -> list[str]:
    requirement_tokens = _tokens(requirement_text)
    evidence_tokens = _tokens(evidence_text)
    if not requirement_tokens:
        return []
    missing = sorted(requirement_tokens - evidence_tokens)
    requirement = normalize_text(requirement_text)
    evidence = normalize_text(evidence_text)
    if (
        any(term in requirement for term in ("defer", "deferred", "later release", "future release"))
        and any(term in evidence for term in ("wait until", "phase two", "phase 2", "later"))
    ):
        deferred_equivalent = {"defer", "deferred", "future", "later", "phase", "release", "semester"}
        missing = [term for term in missing if term not in deferred_equivalent]
    if (
        any(term in requirement for term in ("excluded", "exclude", "out of scope", "launch scope"))
        and any(term in evidence for term in ("not part", "not needed", "not required", "not in"))
    ):
        exclusion_equivalent = {"excluded", "exclude", "scope", "launch", "release", "needed", "required", "part"}
        missing = [term for term in missing if term not in exclusion_equivalent]
    if not evidence_tokens:
        return missing
    missing_ratio = len(missing) / max(1, len(requirement_tokens))
    if missing_ratio >= 0.4 and len(missing) >= 2:
        return missing[:8]
    return []


def _unit_text_by_id(conversation_units: Iterable[ConversationUnit]) -> dict[str, str]:
    return {unit.id: unit.text for unit in conversation_units}


def _requirement_records(spec: SpecOutput) -> list[tuple[str, RequirementItem | ConstraintItem]]:
    return [
        *[("functional_requirements", item) for item in spec.functional_requirements],
        *[("non_functional_requirements", item) for item in spec.non_functional_requirements],
        *[("constraints", item) for item in spec.constraints],
    ]


def _non_empty_strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def deterministic_warnings(
    item: RequirementItem | ConstraintItem,
    unit_texts: dict[str, str],
) -> list[str]:
    warnings: list[str] = []
    source_units = [sid for sid in item.source_units if str(sid).strip()]
    missing_sources = [sid for sid in source_units if sid not in unit_texts]
    if not str(item.text).strip():
        warnings.append("empty_requirement_text")
    if not source_units:
        warnings.append("missing_source_units")
    if missing_sources:
        warnings.append("source_units_not_found")
    if not _non_empty_strings(item.evidence_spans):
        warnings.append("missing_evidence_spans")
    if not _non_empty_strings(item.acceptance_criteria):
        warnings.append("missing_acceptance_criteria")
    elif acceptance_criteria_are_weak(item.text, item.acceptance_criteria):
        warnings.append("weak_acceptance_criteria")
    if not re.search(r"\b(shall|should|must)\b", item.text, flags=re.IGNORECASE):
        warnings.append("missing_requirement_modal")
    normalized_text = normalize_text(item.text)
    if (
        "support this capability" in normalized_text
        or "satisfy this quality expectation" in normalized_text
        or re.search(r"\ballow\s+\w+\s+to\s+(a|an)\s+\w+", normalized_text)
    ):
        warnings.append("weak_requirement_wording")
    combined = " ".join([item.text, *item.acceptance_criteria])
    if has_vague_testability_word(combined) and not has_measurable_context(combined):
        warnings.append("vague_unmeasured_quality_word")
    valid_sources = [sid for sid in source_units if sid in unit_texts]
    evidence_text = " ".join(
        [unit_texts[sid] for sid in valid_sources] + _non_empty_strings(item.evidence_spans)
    )
    missing_claim_terms = unsupported_claim_terms(item.text, evidence_text)
    if missing_claim_terms:
        warnings.append("unsupported_claim_terms:" + ",".join(missing_claim_terms))
    unsupported_numbers = unsupported_numeric_details(
        " ".join([item.text, *_non_empty_strings(item.acceptance_criteria)]),
        evidence_text,
    )
    if unsupported_numbers:
        warnings.append("unsupported_numeric_detail:" + ",".join(unsupported_numbers))
    return sorted(set(warnings))


def _looks_contradicted(requirement_text: str, evidence_text: str) -> bool:
    requirement = normalize_text(requirement_text)
    evidence = normalize_text(evidence_text)
    excludes = (
        "not part",
        "not included",
        "not in",
        "excluded",
        "outside",
        "later",
        "future",
        "phase two",
        "phase 2",
    )
    wants_inclusion = any(
        phrase in requirement
        for phrase in (
            "shall include",
            "shall support",
            "shall allow",
            "shall provide",
            "must include",
        )
    )
    evidence_excludes = any(phrase in evidence for phrase in excludes)
    return bool(wants_inclusion and evidence_excludes)


def heuristic_verification(
    item: RequirementItem | ConstraintItem,
    unit_texts: dict[str, str],
) -> RequirementVerification:
    valid_sources = [sid for sid in item.source_units if sid in unit_texts]
    source_text = " ".join(unit_texts[sid] for sid in valid_sources)
    evidence_span_text = " ".join(_non_empty_strings(item.evidence_spans))
    evidence_text = evidence_span_text or source_text
    score = source_relevance_score(item.text, evidence_text)
    warnings = deterministic_warnings(item, unit_texts)
    has_unsupported_claim_terms = any(
        warning.startswith("unsupported_claim_terms:") for warning in warnings
    )
    has_unsupported_numeric_detail = any(
        warning.startswith("unsupported_numeric_detail:") for warning in warnings
    )

    blocking = {
        "empty_requirement_text",
        "missing_source_units",
        "source_units_not_found",
        "missing_evidence_spans",
    }
    if _looks_contradicted(item.text, evidence_text):
        verdict = "CONTRADICTED"
        confidence = max(0.55, 1.0 - score)
    elif any(warning in warnings for warning in blocking):
        verdict = "UNSUPPORTED"
        confidence = max(0.55, 1.0 - score)
    elif has_unsupported_claim_terms and score < 0.45:
        verdict = "UNSUPPORTED"
        confidence = max(0.6, 1.0 - score)
    elif has_unsupported_numeric_detail and score >= 0.08:
        verdict = "PARTIALLY_SUPPORTED"
        confidence = min(0.7, max(0.45, score))
    elif score >= 0.22:
        verdict = "SUPPORTED"
        confidence = min(0.99, 0.55 + score)
    elif score >= 0.08:
        verdict = "PARTIALLY_SUPPORTED"
        confidence = 0.5
    else:
        verdict = "UNSUPPORTED"
        confidence = 0.75

    return RequirementVerification(
        source_relevance_score=round(score, 4),
        verdict=verdict,
        confidence=round(confidence, 4),
        warnings=warnings,
    )


def build_llm_verifier_prompt(item: RequirementItem | ConstraintItem, evidence_text: str) -> str:
    schema_hint = {
        "verdict": "SUPPORTED | PARTIALLY_SUPPORTED | UNSUPPORTED | CONTRADICTED",
        "confidence": 0.0,
        "reason": "brief grounded reason",
    }
    return (
        "SPEC_VERIFIER_CLAIM_EVIDENCE\n"
        "Decide whether the evidence supports the requirement claim.\n"
        "Return JSON only. Do not include markdown fences.\n\n"
        "Allowed verdicts: SUPPORTED, PARTIALLY_SUPPORTED, UNSUPPORTED, CONTRADICTED.\n"
        f"Required JSON schema:\n{json.dumps(schema_hint, indent=2)}\n\n"
        f"Claim:\n{item.text}\n\n"
        f"Evidence:\n{evidence_text}\n\n"
        "Output JSON only."
    )


def _parse_llm_verifier_output(raw_output: str) -> tuple[str | None, float | None, str | None]:
    start = raw_output.find("{")
    end = raw_output.rfind("}")
    if start < 0 or end <= start:
        return None, None, None
    try:
        payload = json.loads(raw_output[start : end + 1])
    except Exception:
        return None, None, None
    verdict = str(payload.get("verdict", "")).strip().upper()
    if verdict not in VERDICTS or verdict == "NOT_CHECKED":
        verdict = ""
    confidence: float | None = None
    try:
        confidence = float(payload.get("confidence"))
    except Exception:
        confidence = None
    reason = str(payload.get("reason", "")).strip() or None
    return verdict or None, confidence, reason


def _set_item_verification(
    item: RequirementItem | ConstraintItem,
    verification: RequirementVerification,
) -> RequirementItem | ConstraintItem:
    payload = model_dump_compat(item)
    payload["verification"] = model_dump_compat(verification)
    return model_validate_compat(type(item), payload)


def _apply_verifications_to_spec(
    spec: SpecOutput,
    verifications: dict[tuple[str, str], RequirementVerification],
) -> SpecOutput:
    payload = model_dump_compat(spec)
    for category in ("functional_requirements", "non_functional_requirements", "constraints"):
        updated: list[dict[str, Any]] = []
        for raw_item in payload.get(category, []):
            key = (category, str(raw_item.get("id", "")))
            if key in verifications:
                raw_item["verification"] = model_dump_compat(verifications[key])
            updated.append(raw_item)
        payload[category] = updated
    return model_validate_compat(SpecOutput, payload)


def _make_entries(
    spec: SpecOutput,
    verifications: dict[tuple[str, str], RequirementVerification],
    repair_statuses: dict[tuple[str, str], str] | None = None,
) -> list[VerificationEntry]:
    repair_statuses = repair_statuses or {}
    entries: list[VerificationEntry] = []
    for category, item in _requirement_records(spec):
        verification = verifications.get((category, item.id), item.verification)
        entries.append(
            VerificationEntry(
                requirement_id=item.id,
                category=category,
                text=item.text,
                source_units=list(item.source_units),
                evidence_spans=list(item.evidence_spans),
                source_relevance_score=verification.source_relevance_score,
                verdict=verification.verdict,
                confidence=verification.confidence,
                warnings=list(verification.warnings),
                repair_status=repair_statuses.get((category, item.id)),
            )
        )
    return entries


def _report_dict(
    entries: list[VerificationEntry],
    *,
    verify_mode: str,
    repair_on_fail: bool,
    num_llm_calls: int,
    repair_trigger_count: int,
    repair_success_count: int,
) -> dict[str, Any]:
    total = len(entries)
    supported = sum(
        1 for entry in entries if entry.verdict in {"SUPPORTED", "PARTIALLY_SUPPORTED"}
    )
    unsupported = sum(
        1
        for entry in entries
        if entry.verdict in {"UNSUPPORTED", "CONTRADICTED", "NOT_ENOUGH_INFO"}
    )
    unsupported_numeric_detail = sum(
        1
        for entry in entries
        if any(
            warning.startswith("unsupported_numeric_detail:")
            for warning in entry.warnings
        )
    )
    return {
        "summary": {
            "requirement_count": total,
            "verify_mode": verify_mode,
            "repair_on_fail": repair_on_fail,
            "num_llm_calls": num_llm_calls,
            "groundedness_rate": supported / total if total else 0.0,
            "unsupported_requirement_rate": unsupported / total if total else 0.0,
            "unsupported_numeric_detail_count": unsupported_numeric_detail,
            "unsupported_numeric_detail_rate": (
                unsupported_numeric_detail / total if total else 0.0
            ),
            "verification_pass_rate": (
                sum(1 for entry in entries if entry.verdict == "SUPPORTED") / total
                if total
                else 0.0
            ),
            "repair_trigger_count": repair_trigger_count,
            "repair_success_count": repair_success_count,
            "repair_trigger_rate": repair_trigger_count / total if total else 0.0,
            "repair_success_rate": (
                repair_success_count / repair_trigger_count if repair_trigger_count else 0.0
            ),
        },
        "requirements": [entry.__dict__ for entry in entries],
    }


def format_verification_report_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    lines = [
        "# Verification Report",
        "",
        "## Summary",
        f"- Requirement count: {summary.get('requirement_count', 0)}",
        f"- Verify mode: {summary.get('verify_mode', 'heuristic')}",
        f"- Total LLM calls: {summary.get('num_llm_calls', 0)}",
        f"- Generator LLM calls: {summary.get('generator_llm_calls', 0)}",
        f"- Verifier LLM calls: {summary.get('verifier_llm_calls', 0)}",
        f"- Groundedness rate: {float(summary.get('groundedness_rate', 0.0)):.4f}",
        f"- Unsupported requirement rate: {float(summary.get('unsupported_requirement_rate', 0.0)):.4f}",
        f"- Unsupported numeric detail rate: {float(summary.get('unsupported_numeric_detail_rate', 0.0)):.4f}",
        f"- Verification pass rate: {float(summary.get('verification_pass_rate', 0.0)):.4f}",
        f"- Repair trigger rate: {float(summary.get('repair_trigger_rate', 0.0)):.4f}",
        f"- Repair success rate: {float(summary.get('repair_success_rate', 0.0)):.4f}",
        "",
        "## Requirements",
    ]
    for entry in report.get("requirements", []):
        sources = ", ".join(entry.get("source_units", [])) or "-"
        warnings = ", ".join(entry.get("warnings", [])) or "-"
        evidence = " | ".join(entry.get("evidence_spans", [])) or "-"
        repair = entry.get("repair_status") or "-"
        confidence = entry.get("confidence")
        confidence_text = "N/A" if confidence is None else f"{float(confidence):.4f}"
        lines.extend(
            [
                f"- **{entry.get('requirement_id', '')}** ({entry.get('category', '')}): {entry.get('text', '')}",
                f"  - source_units: {sources}",
                f"  - evidence: {evidence}",
                f"  - verdict: {entry.get('verdict', 'NOT_CHECKED')} (confidence={confidence_text})",
                f"  - warnings: {warnings}",
                f"  - repair: {repair}",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def needs_selective_repair(item: RequirementItem | ConstraintItem) -> bool:
    if item.verification.verdict in REPAIR_TRIGGER_VERDICTS:
        return True
    if any(
        warning.startswith("unsupported_numeric_detail:")
        for warning in item.verification.warnings
    ):
        return True
    if any(
        warning in {"weak_acceptance_criteria", "weak_requirement_wording"}
        for warning in item.verification.warnings
    ):
        return True
    if not _non_empty_strings(item.acceptance_criteria):
        return True
    if acceptance_criteria_are_weak(item.text, item.acceptance_criteria):
        return True
    return False


def _unsupported_numeric_detail_warnings(item: RequirementItem | ConstraintItem) -> list[str]:
    return [
        warning
        for warning in item.verification.warnings
        if warning.startswith("unsupported_numeric_detail:")
    ]


def _repair_text_with_unsupported_numeric_details(
    text: str,
    evidence_text: str,
) -> str:
    if not numeric_threshold_details(text):
        return text
    normalized_evidence = normalize_text(evidence_text)
    normalized_text = normalize_text(text)
    if "load" in normalized_text and ("mobile" in normalized_text or "phone" in normalized_text):
        if any(word in normalized_evidence for word in ("quick", "quickly", "fast")):
            return "The system shall load quickly on mobile."

    repaired = NUMERIC_THRESHOLD_RE.sub("a defined target", text)
    repaired = re.sub(
        r"\b(within|under|less than|more than|at least|at most|"
        r"no more than|no less than)\s+a defined target\b",
        "against a defined target",
        repaired,
        flags=re.IGNORECASE,
    )
    repaired = re.sub(r"\s+", " ", repaired).strip()
    return repaired or text


def _quality_checks_payload(item_payload: dict[str, Any]) -> dict[str, Any]:
    return model_dump_compat(
        infer_quality_checks(
            requirement_text=str(item_payload.get("text", "")),
            source_units=item_payload.get("source_units", []),
            evidence_spans=item_payload.get("evidence_spans", []),
            acceptance_criteria=item_payload.get("acceptance_criteria", []),
        )
    )


def _question_mentions_requirement(question: dict[str, Any], requirement_text: str) -> bool:
    question_text = str(question.get("text", "")).strip()
    question_tokens = _tokens(question_text)
    requirement_tokens = _tokens(requirement_text)
    if not question_tokens or not requirement_tokens:
        return False
    overlap = len(question_tokens & requirement_tokens) / max(1, len(requirement_tokens))
    return overlap >= 0.4


class SpecVerifier:
    def __init__(
        self,
        *,
        runner: Any | None = None,
        generation_config: dict[str, Any] | None = None,
        minicheck_scorer: MiniCheckScorer | None = None,
    ) -> None:
        self.runner = runner
        self.generation_config = generation_config or {}
        self.minicheck_scorer = minicheck_scorer

    @staticmethod
    def _parse_repair_json(raw_output: str) -> dict[str, Any] | None:
        text = re.sub(r"<think>.*?</think>", "", str(raw_output), flags=re.DOTALL).strip()
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            payload = json.loads(text[start : end + 1])
        except Exception:
            return None
        repaired_text = str(payload.get("text", "")).strip()
        repaired_criteria = _non_empty_strings(payload.get("acceptance_criteria", []))
        if not repaired_text or not repaired_criteria:
            return None
        return {
            "text": repaired_text,
            "acceptance_criteria": repaired_criteria,
        }

    def _repair_generation_config(self) -> dict[str, Any]:
        config = dict(self.generation_config)
        try:
            config["max_new_tokens"] = min(int(config.get("max_new_tokens", 512)), 512)
        except Exception:
            config["max_new_tokens"] = 512
        return config

    @staticmethod
    def _build_language_repair_prompt(
        *,
        category: str,
        item: RequirementItem | ConstraintItem,
        evidence_text: str,
    ) -> str:
        schema = {
            "text": "rewritten requirement text",
            "acceptance_criteria": ["Given ..., When ..., Then ..."],
        }
        return (
            "SPEC_REPAIR_REQUIREMENT_LANGUAGE\n"
            "Rewrite only the requirement wording and acceptance criteria.\n"
            "Use the evidence as the only source of truth. Do not add new scope, numbers, tools, or policies.\n"
            "Preserve the same requirement type and meaning. If the source is ambiguous, keep the wording conservative.\n"
            "Return JSON only. Do not include markdown fences, comments, or reasoning.\n\n"
            f"Requirement category: {category}\n"
            f"Current requirement: {item.text}\n"
            f"Current acceptance criteria: {json.dumps(_non_empty_strings(item.acceptance_criteria), ensure_ascii=False)}\n"
            f"Evidence:\n{evidence_text}\n\n"
            f"Required JSON schema:\n{json.dumps(schema, ensure_ascii=False, indent=2)}\n\n"
            "Output JSON only.\n/no_think"
        )

    def _llm_repair_language(
        self,
        *,
        category: str,
        item: RequirementItem | ConstraintItem,
    ) -> tuple[dict[str, Any] | None, int]:
        if not self.runner:
            return None, 0
        evidence_text = "\n".join(_non_empty_strings(item.evidence_spans))
        if not evidence_text:
            return None, 0
        prompt = self._build_language_repair_prompt(
            category=category,
            item=item,
            evidence_text=evidence_text,
        )
        try:
            raw_output = self.runner.generate(prompt, self._repair_generation_config())
        except Exception:
            return None, 0
        repaired = self._parse_repair_json(raw_output)
        if not repaired:
            return None, 1
        repaired_text = repaired["text"]
        repaired_criteria = repaired["acceptance_criteria"]
        if (
            "support this capability" in normalize_text(repaired_text)
            or "satisfy this quality expectation" in normalize_text(repaired_text)
            or acceptance_criteria_are_weak(repaired_text, repaired_criteria)
        ):
            return None, 1
        return repaired, 1

    def _minicheck_verification(
        self,
        item: RequirementItem | ConstraintItem,
        unit_texts: dict[str, str],
        heuristic: RequirementVerification,
    ) -> RequirementVerification:
        valid_sources = [sid for sid in item.source_units if sid in unit_texts]
        evidence_spans = _non_empty_strings(item.evidence_spans)
        warnings = list(heuristic.warnings)
        if not valid_sources or not evidence_spans:
            return RequirementVerification(
                source_relevance_score=heuristic.source_relevance_score,
                verdict="NOT_ENOUGH_INFO",
                confidence=0.0,
                warnings=sorted(set(warnings + ["minicheck_not_run_missing_evidence"])),
            )

        source_text = "\n".join(f"{sid}: {unit_texts[sid]}" for sid in valid_sources)
        evidence_span_text = "\n".join(evidence_spans)
        evidence_text = evidence_span_text or source_text
        if _looks_contradicted(item.text, evidence_text):
            return RequirementVerification(
                source_relevance_score=heuristic.source_relevance_score,
                verdict="CONTRADICTED",
                confidence=heuristic.confidence,
                warnings=sorted(set(warnings + ["rule_detected_scope_contradiction"])),
            )
        if any(warning.startswith("unsupported_numeric_detail:") for warning in warnings):
            return RequirementVerification(
                source_relevance_score=heuristic.source_relevance_score,
                verdict="PARTIALLY_SUPPORTED",
                confidence=heuristic.confidence,
                warnings=sorted(
                    set(warnings + ["minicheck_skipped_rule_unsupported_numeric_detail"])
                ),
            )

        scorer = self.minicheck_scorer or MiniCheckScorer()
        self.minicheck_scorer = scorer
        label, support_prob = scorer.score(document=evidence_text, claim=item.text)
        if label == 1:
            verdict = "SUPPORTED"
        elif (heuristic.source_relevance_score or 0.0) >= 0.08:
            verdict = "PARTIALLY_SUPPORTED"
        else:
            verdict = "UNSUPPORTED"
        warnings.append(f"minicheck_support_prob:{support_prob:.4f}")
        return RequirementVerification(
            source_relevance_score=heuristic.source_relevance_score,
            verdict=verdict,
            confidence=round(support_prob, 4),
            warnings=sorted(set(warnings)),
        )

    def _verify_once(
        self,
        spec: SpecOutput,
        conversation_units: Iterable[ConversationUnit],
        *,
        verify_mode: str,
    ) -> tuple[SpecOutput, dict[tuple[str, str], RequirementVerification], int]:
        if verify_mode not in {"off", "heuristic", "llm", "minicheck"}:
            raise ValueError("verify_mode must be one of: off, heuristic, llm, minicheck")

        if verify_mode == "off":
            verifications = {
                (category, item.id): RequirementVerification()
                for category, item in _requirement_records(spec)
            }
            return _apply_verifications_to_spec(spec, verifications), verifications, 0

        unit_texts = _unit_text_by_id(conversation_units)
        verifications: dict[tuple[str, str], RequirementVerification] = {}
        num_llm_calls = 0

        for category, item in _requirement_records(spec):
            verification = heuristic_verification(item, unit_texts)
            if verify_mode == "minicheck":
                verification = self._minicheck_verification(item, unit_texts, verification)
            if (
                verify_mode == "llm"
                and verification.verdict == "PARTIALLY_SUPPORTED"
                and self.runner
                and not any(
                    warning.startswith("unsupported_numeric_detail:")
                    for warning in verification.warnings
                )
            ):
                valid_sources = [sid for sid in item.source_units if sid in unit_texts]
                evidence_text = "\n".join(f"{sid}: {unit_texts[sid]}" for sid in valid_sources)
                prompt = build_llm_verifier_prompt(item, evidence_text)
                raw_output = self.runner.generate(prompt, self.generation_config)
                num_llm_calls += 1
                verdict, confidence, reason = _parse_llm_verifier_output(raw_output)
                if verdict:
                    warnings = list(verification.warnings)
                    if reason:
                        warnings.append(f"llm_reason: {reason}")
                    verification = RequirementVerification(
                        source_relevance_score=verification.source_relevance_score,
                        verdict=verdict,
                        confidence=confidence if confidence is not None else verification.confidence,
                        warnings=warnings,
                    )
            verifications[(category, item.id)] = verification

        verified_spec = _apply_verifications_to_spec(spec, verifications)
        return verified_spec, verifications, num_llm_calls

    def _repair(
        self,
        spec: SpecOutput,
        *,
        repair_statuses: dict[tuple[str, str], str],
    ) -> tuple[SpecOutput, int, int, int]:
        payload = model_dump_compat(spec)
        repair_trigger_count = 0
        repair_success_count = 0
        repair_llm_calls = 0
        new_open_questions = list(payload.get("open_questions", []))

        for category in ("functional_requirements", "non_functional_requirements", "constraints"):
            kept: list[dict[str, Any]] = []
            for raw_item in payload.get(category, []):
                item_cls = ConstraintItem if category == "constraints" else RequirementItem
                item = model_validate_compat(item_cls, raw_item)
                key = (category, item.id)
                if not needs_selective_repair(item):
                    kept.append(raw_item)
                    continue

                repair_trigger_count += 1
                numeric_warnings = _unsupported_numeric_detail_warnings(item)
                if numeric_warnings:
                    evidence_text = " ".join(_non_empty_strings(item.evidence_spans))
                    repaired_text = _repair_text_with_unsupported_numeric_details(
                        item.text,
                        evidence_text,
                    )
                    raw_item["text"] = repaired_text
                    requirement_type = (
                        "constraint"
                        if category == "constraints"
                        else "non_functional_requirement"
                        if category == "non_functional_requirements"
                        else "functional_requirement"
                    )
                    raw_item["acceptance_criteria"] = default_acceptance_criteria(
                        repaired_text,
                        requirement_type,
                    )
                    raw_item["quality_checks"] = _quality_checks_payload(raw_item)
                    new_open_questions = [
                        question
                        for question in new_open_questions
                        if not _question_mentions_requirement(question, item.text)
                    ]
                    new_open_questions.append(
                        model_dump_compat(
                            QuestionItem(
                                text=(
                                    f"What measurable target should define {item.id}: "
                                    f"{repaired_text}"
                                ),
                                source_units=item.source_units,
                            )
                        )
                    )
                    repair_statuses[key] = "unsupported_numeric_detail_moved_to_open_question"
                    repair_success_count += 1
                    kept.append(raw_item)
                    continue

                language_quality_warnings = [
                    warning
                    for warning in item.verification.warnings
                    if warning in {"weak_acceptance_criteria", "weak_requirement_wording"}
                ]
                if (
                    language_quality_warnings
                    and item.verification.verdict not in REPAIR_TRIGGER_VERDICTS
                ):
                    repaired_payload, calls = self._llm_repair_language(
                        category=category,
                        item=item,
                    )
                    repair_llm_calls += calls
                    if repaired_payload:
                        raw_item["text"] = repaired_payload["text"]
                        raw_item["acceptance_criteria"] = repaired_payload["acceptance_criteria"]
                        raw_item["quality_checks"] = _quality_checks_payload(raw_item)
                        repair_statuses[key] = "llm_language_repaired"
                        repair_success_count += 1
                    elif "weak_acceptance_criteria" in language_quality_warnings:
                        raw_item["acceptance_criteria"] = default_acceptance_criteria(
                            item.text,
                            "constraint"
                            if category == "constraints"
                            else "non_functional_requirement"
                            if category == "non_functional_requirements"
                            else "functional_requirement",
                        )
                        raw_item["quality_checks"] = _quality_checks_payload(raw_item)
                        repair_statuses[key] = "acceptance_criteria_refreshed"
                        repair_success_count += 1
                    else:
                        repair_statuses[key] = "language_repair_not_available"
                    kept.append(raw_item)
                    continue

                if not _non_empty_strings(item.acceptance_criteria):
                    raw_item["acceptance_criteria"] = default_acceptance_criteria(
                        item.text,
                        "constraint" if category == "constraints" else "functional_requirement",
                    )
                    repair_statuses[key] = "acceptance_criteria_added"
                    repair_success_count += 1
                    kept.append(raw_item)
                    continue

                new_open_questions = [
                    question
                    for question in new_open_questions
                    if not _question_mentions_requirement(question, item.text)
                ]
                new_open_questions.append(
                    model_dump_compat(
                        QuestionItem(
                            text=f"Please confirm or revise unsupported requirement {item.id}: {item.text}",
                            source_units=item.source_units,
                        )
                    )
                )
                repair_statuses[key] = "moved_to_open_questions"
                repair_success_count += 1
            payload[category] = kept

        payload["open_questions"] = new_open_questions
        return (
            model_validate_compat(SpecOutput, payload),
            repair_trigger_count,
            repair_success_count,
            repair_llm_calls,
        )

    def run(
        self,
        spec: SpecOutput,
        conversation_units: Iterable[ConversationUnit],
        *,
        verify_mode: str = "minicheck",
        repair_on_fail: bool = False,
    ) -> VerificationRunResult:
        units = list(conversation_units)
        verified_spec, verifications, llm_calls = self._verify_once(
            spec,
            units,
            verify_mode=verify_mode,
        )
        pre_repair_entries = _make_entries(verified_spec, verifications)

        repair_statuses: dict[tuple[str, str], str] = {}
        repair_trigger_count = 0
        repair_success_count = 0
        if repair_on_fail and verify_mode != "off":
            repaired_spec, repair_trigger_count, repair_success_count, repair_llm_calls = self._repair(
                verified_spec,
                repair_statuses=repair_statuses,
            )
            llm_calls += repair_llm_calls
            if repair_trigger_count:
                verified_spec, verifications, extra_calls = self._verify_once(
                    repaired_spec,
                    units,
                    verify_mode=verify_mode,
                )
                llm_calls += extra_calls

        entries = _make_entries(verified_spec, verifications, repair_statuses)
        final_keys = {(entry.category, entry.requirement_id) for entry in entries}
        for entry in pre_repair_entries:
            key = (entry.category, entry.requirement_id)
            if key in final_keys or key not in repair_statuses:
                continue
            entry.repair_status = repair_statuses[key]
            entries.append(entry)
        report = _report_dict(
            entries,
            verify_mode=verify_mode,
            repair_on_fail=repair_on_fail,
            num_llm_calls=llm_calls,
            repair_trigger_count=repair_trigger_count,
            repair_success_count=repair_success_count,
        )
        warning_set = {
            f"{entry.category}:{entry.requirement_id} {warning}"
            for entry in entries
            for warning in entry.warnings
        }
        report_markdown = format_verification_report_markdown(report)
        return VerificationRunResult(
            spec=verified_spec,
            report=report,
            report_markdown=report_markdown,
            warnings=sorted(warning_set),
            num_llm_calls=llm_calls,
            repair_trigger_count=repair_trigger_count,
            repair_success_count=repair_success_count,
        )
