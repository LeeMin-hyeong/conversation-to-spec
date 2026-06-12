from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field


class ConversationUnit(BaseModel):
    id: str
    text: str


class RequirementQualityChecks(BaseModel):
    is_atomic: bool = False
    is_testable: bool = False
    has_clear_actor: bool = False
    has_traceable_evidence: bool = False
    ambiguity_risk: Literal["low", "medium", "high"] = "medium"


class RequirementVerification(BaseModel):
    source_relevance_score: float | None = None
    verdict: Literal[
        "SUPPORTED",
        "PARTIALLY_SUPPORTED",
        "UNSUPPORTED",
        "CONTRADICTED",
        "NOT_ENOUGH_INFO",
        "NOT_CHECKED",
    ] = "NOT_CHECKED"
    confidence: float | None = None
    warnings: List[str] = Field(default_factory=list)


class AtomicSourceDecision(BaseModel):
    decision: Literal[
        "functional_requirement",
        "non_functional_requirement",
        "constraint",
        "open_question",
        "note",
        "discard",
    ]
    claim: str = ""
    open_question: str = ""
    follow_up_question: str = ""
    note: str = ""


class SourceUnitDecision(BaseModel):
    source_unit: str
    atomic_decisions: List[AtomicSourceDecision] = Field(default_factory=list)


class RequirementItem(BaseModel):
    id: str
    text: str
    source_units: List[str] = Field(default_factory=list)
    evidence_spans: List[str] = Field(default_factory=list)
    acceptance_criteria: List[str] = Field(default_factory=list)
    quality_checks: RequirementQualityChecks = Field(default_factory=RequirementQualityChecks)
    verification: RequirementVerification = Field(default_factory=RequirementVerification)


class ConstraintItem(BaseModel):
    id: str
    text: str
    source_units: List[str] = Field(default_factory=list)
    evidence_spans: List[str] = Field(default_factory=list)
    acceptance_criteria: List[str] = Field(default_factory=list)
    quality_checks: RequirementQualityChecks = Field(default_factory=RequirementQualityChecks)
    verification: RequirementVerification = Field(default_factory=RequirementVerification)


class QuestionItem(BaseModel):
    text: str
    source_units: List[str] = Field(default_factory=list)


class NoteItem(BaseModel):
    text: str
    source_units: List[str] = Field(default_factory=list)


class SpecOutput(BaseModel):
    project_summary: str
    functional_requirements: List[RequirementItem] = Field(default_factory=list)
    non_functional_requirements: List[RequirementItem] = Field(default_factory=list)
    constraints: List[ConstraintItem] = Field(default_factory=list)
    open_questions: List[QuestionItem] = Field(default_factory=list)
    follow_up_questions: List[QuestionItem] = Field(default_factory=list)
    notes: List[NoteItem] = Field(default_factory=list)
    conversation_units: List[ConversationUnit] = Field(default_factory=list)
    verification_warnings: List[str] = Field(default_factory=list)


class EnrichedRequirementItem(BaseModel):
    id: str
    type: str
    text: str
    source_units: List[str] = Field(default_factory=list)
    evidence_spans: List[str] = Field(default_factory=list)
    acceptance_criteria: List[str] = Field(default_factory=list)
    quality_checks: RequirementQualityChecks = Field(default_factory=RequirementQualityChecks)
