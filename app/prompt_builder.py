from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from app.schemas import ConversationUnit
from app.utils import load_yaml_file


PROMPT_CONFIG_PATH = Path("configs/prompts.yaml")


def load_prompt_config(path: Path = PROMPT_CONFIG_PATH) -> dict:
    return load_yaml_file(path)


def _unit_block(conversation_units: Iterable[ConversationUnit]) -> str:
    return "\n".join(f"{u.id} | {u.text}" for u in conversation_units)


def _dump_json(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False)


def _single_shot_instructions(prompt_config: dict) -> str:
    text = str(prompt_config.get("single_shot_instructions", "")).strip()
    if text:
        return text
    return (
        "Convert the conversation into a structured software specification draft in one pass. "
        "Classify functional requirements, non-functional requirements, constraints, "
        "open questions, follow-up questions, and notes."
    )


def build_single_shot_spec_prompt(
    conversation_units: Iterable[ConversationUnit],
    prompt_config: dict,
    prompt_style: str = "few_shot",
) -> str:
    if prompt_style not in {"zero_shot", "few_shot"}:
        raise ValueError("prompt_style must be 'zero_shot' or 'few_shot'.")

    schema_hint = {
        "project_summary": "2-4 sentence summary",
        "source_unit_decisions": [
            {
                "source_unit": "U1",
                "decision": "functional_requirement",
                "claim": "short source-backed clause for this one atomic decision",
            }
        ],
    }

    few_shot_block = ""
    if prompt_style == "few_shot":
        few_shot_block = (
            "Few-shot example for MAPPING ONLY, not answer text.\n"
            "Example input:\n"
            "U_EX1 | EX_ACTOR_A should EX_ACTION_A EX_OBJECT_A.\n"
            "U_EX2 | EX_ACTOR_B should EX_ACTION_B EX_OBJECT_B, but EX_FUTURE_FEATURE_A can wait until EX_PHASE_A.\n"
            "U_EX3 | EX_SURFACE_A should EX_QUALITY_A for EX_CONDITION_A.\n"
            "Expected mapping pattern:\n"
            "{\n"
            '  "project_summary": "EX_PROJECT_SUMMARY",\n'
            '  "source_unit_decisions": [\n'
            '    {"source_unit": "U_EX1", "decision": "functional_requirement", "claim": "EX_ACTOR_A should EX_ACTION_A EX_OBJECT_A"},\n'
            '    {"source_unit": "U_EX2", "decision": "functional_requirement", "claim": "EX_ACTOR_B should EX_ACTION_B EX_OBJECT_B"},\n'
            '    {"source_unit": "U_EX2", "decision": "constraint", "claim": "EX_FUTURE_FEATURE_A can wait until EX_PHASE_A"},\n'
            '    {"source_unit": "U_EX3", "decision": "open_question", "claim": "EX_QUALITY_A is not measurable"}\n'
            "  ]\n"
            "}\n"
            "Mapping notes:\n"
            "- Use non_functional_requirement for concrete quality attributes only.\n"
            "- Use open_question for vague quality wording such as undefined simple/easy/fast.\n"
            "- The final JSON must use the real U1/U2/... units below, not U_EX* IDs or EX_* words.\n\n"
        )

    return (
        "SINGLE_SHOT_TRACEABLE_SPEC\n"
        "/no_think\n"
        "You are a requirements analysis assistant.\n"
        f"{_single_shot_instructions(prompt_config)}\n\n"
        "Rules:\n"
        "- Return exactly one valid JSON object. No markdown, no prose, no comments.\n"
        "- Do not output <think>, </think>, reasoning, analysis, or explanations.\n"
        "- Do not invent unsupported content, source unit IDs, evidence, or acceptance criteria.\n"
        "- The few-shot example is format-only; never copy its sentinel tokens, source IDs, terms, or requirement text.\n"
        "- Never output EX_* tokens, U_EX IDs, placeholder text, or schema words such as array<string>.\n"
        "- Each source_unit_decisions item is one atomic decision row.\n"
        "- Do not output nested arrays inside source_unit_decisions.\n"
        "- Do not output an atomic_decisions field.\n"
        "- If a source unit contains multiple intents, repeat the same source_unit in multiple rows.\n"
        "- Split combined statements when they contain a feature plus a release limit, a current requirement plus a future-phase item, a feature plus an unresolved question, or a deadline plus a procurement/scope constraint.\n"
        "- Allowed decision values: functional_requirement, non_functional_requirement, constraint, open_question, note, discard.\n"
        "- For functional_requirement, non_functional_requirement, and constraint, claim must be non-empty and copied or lightly paraphrased from the source unit.\n"
        "- For open_question, claim must name the unresolved or vague decision.\n"
        "- Classify user/system actions such as upload, notify, replace, filter, leave notes, see status, create, update, or submit as functional_requirement.\n"
        "- Use non_functional_requirement only for quality attributes such as performance, usability, security, reliability, accessibility, or mobile support.\n"
        "- Do not classify project background or a general product name as a non_functional_requirement.\n"
        "- Use constraint for release boundaries, deferred features, scope exclusions, deadlines, budgets, or platform limits.\n"
        "- Use open_question for vague style/usability/performance/security wording that needs a measurable target.\n"
        "- Use note for project context that is not independently testable.\n"
        "- Each atomic_decision claim should be short, atomic, and copied from the source meaning; leave it empty if unsure.\n"
        "- Do not generate final functional_requirements, non_functional_requirements, constraints, evidence_spans, acceptance_criteria, quality_checks, or verification.\n"
        "- Do not add numeric thresholds such as seconds, percentages, limits, or counts unless the same value appears in a source unit.\n"
        "- Keep the JSON compact.\n\n"
        f"{few_shot_block}"
        f"Required JSON schema:\n{_dump_json(schema_hint)}\n\n"
        "Conversation units:\n"
        f"{_unit_block(conversation_units)}\n\n"
        "Output JSON only."
        "\n/no_think"
    )
