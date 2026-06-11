from __future__ import annotations

import json
import re
import time
from typing import Any

from app.model_runner import BaseModelRunner


class FakeSingleShotRunner(BaseModelRunner):
    def __init__(self, payload: dict[str, Any] | None = None, raw_output: str | None = None) -> None:
        super().__init__(model_name="fake-single-shot")
        self.payload = payload
        self.raw_output = raw_output
        self.calls: list[str] = []

    @staticmethod
    def _extract_units(prompt: str) -> list[tuple[str, str]]:
        pattern = re.compile(r"^(U\d+)\s+\|\s+(.+)$", flags=re.MULTILINE)
        return [(m.group(1), m.group(2).strip()) for m in pattern.finditer(prompt)]

    @staticmethod
    def _default_payload(prompt: str) -> dict[str, Any]:
        units = FakeSingleShotRunner._extract_units(prompt)
        unit_id, unit_text = units[0] if units else ("U1", "Need a simple booking system.")
        return {
            "project_summary": "The conversation describes a small software project with traceable requirements.",
            "functional_requirements": [
                {
                    "id": "FR1",
                    "text": "The system shall allow customers to reserve tables online.",
                    "source_units": [unit_id],
                    "evidence_spans": [unit_text],
                    "acceptance_criteria": [
                        "Given reservations are in scope, When a customer reserves a table online, Then the system records the reservation."
                    ],
                }
            ],
            "non_functional_requirements": [],
            "constraints": [],
            "open_questions": [],
            "follow_up_questions": [],
            "notes": [],
            "verification_warnings": [],
        }

    def generate(self, prompt: str, generation_config: dict) -> str:
        started = time.perf_counter()
        self.calls.append(prompt)
        if "SPEC_REPAIR_REQUIREMENT_LANGUAGE" in prompt:
            output = json.dumps(
                {
                    "text": "The system shall provide a dashboard for staff to manage requests.",
                    "acceptance_criteria": [
                        "Given staff open the dashboard, When they manage requests, Then the system shall save the requested changes."
                    ],
                }
            )
        elif "SPEC_VERIFIER_CLAIM_EVIDENCE" in prompt:
            output = json.dumps(
                {
                    "verdict": "SUPPORTED",
                    "confidence": 0.9,
                    "reason": "Fake verifier output for tests.",
                }
            )
        elif self.raw_output is not None:
            output = self.raw_output
        else:
            output = json.dumps(self.payload or self._default_payload(prompt))
        self.last_generation_info = {
            "model_name": self.model_name,
            "latency_sec": time.perf_counter() - started,
            "prompt_tokens": None,
            "completion_tokens": None,
        }
        return output
