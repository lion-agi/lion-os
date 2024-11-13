from typing import Any

from pydantic import Field

from lion.core.form import Report

from .operations.composite.base import OperationType, OperativeForm


class OperativeReport(Report):
    """Enhanced report for tracking operative patterns."""

    success_patterns: dict[OperationType, list[dict]] = Field(
        default_factory=lambda: {op: [] for op in OperationType}
    )

    def add_result(self, form: OperativeForm, analyze: bool = True) -> None:
        """Add and optionally analyze operation result."""
        self.save_completed_form(form)

        if analyze and not form.metadata.get("error"):
            self._analyze_pattern(form)

    def _analyze_pattern(self, form: OperativeForm) -> None:
        """Extract and store success pattern."""
        pattern = {
            "guidance": form.guidance,
            "context_keys": list(form.context.keys()),
            "confidence": form.confidence_score,
            "field_values": {name: getattr(form, name) for name in form.work_fields},
        }
        self.success_patterns[form.operation_type].append(pattern)

    def get_similar_patterns(
        self,
        operation_type: OperationType,
        context_keys: list[str],
        min_confidence: float = 0.8,
    ) -> list[dict]:
        """Find similar successful patterns."""
        return [
            pattern
            for pattern in self.success_patterns[operation_type]
            if (
                pattern["confidence"] >= min_confidence
                and all(k in pattern["context_keys"] for k in context_keys)
            )
        ]
