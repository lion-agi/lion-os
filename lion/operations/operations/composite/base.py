from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

from lion.core.models import FieldModel, NewModelParams
from lion.protocols.operatives.operative import Operative

T = TypeVar("T")
C = TypeVar("C", bound="BaseConfig")


class OperationType(str, Enum):
    SELECT = "select"
    BRAINSTORM = "brainstorm"
    PLAN = "plan"
    COMPOSITE = "composite"


class BaseConfig(BaseModel):
    operation_type: OperationType
    guidance: str | None = None
    context: Any | None = None
    confidence_threshold: float = 0.7


class ReasonModel(BaseModel):
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reasoning: str | None = None
    pattern_matches: list[str] = Field(default_factory=list)


class OperativeForm(BaseModel):
    """Base form for operative tasks."""

    operation_type: OperationType
    guidance: str = ""
    context: dict[str, Any] = Field(default_factory=dict)
    work_fields: list[str] = Field(default_factory=list)
    confidence_score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class OperativeForm(ABC, Generic[T]):
    """Enhanced base form with Operative integration."""

    def __init__(self, config: BaseConfig):
        self.config = config
        self.result: T | None = None
        self.reasoning: ReasonModel = ReasonModel()
        self.operative: Operative | None = None

    def setup_operative(
        self,
        request_params: NewModelParams | None = None,
        field_models: list[FieldModel] | None = None,
        exclude_fields: list[str] | None = None,
    ) -> None:
        """Initialize operative with parameters."""
        self.operative = Operative(
            request_params=request_params,
            field_models=field_models,
            exclude_fields=exclude_fields,
        )

    async def execute(self) -> T:
        """Execute with operative handling."""
        if not self.is_workable():
            raise ValueError("Missing required inputs")

        try:
            if self.operative:
                self.result = await self._execute_with_operative()
            else:
                self.result = await self._execute_operation()

            await self._validate_result()
            return self.result

        except Exception as e:
            self.metadata["error"] = str(e)
            raise

    async def _execute_with_operative(self) -> T:
        """Execute using operative for request/response handling."""
        result = await self._execute_operation()

        if isinstance(result, (str, dict)):
            self.operative.update_response_model(
                text=result if isinstance(result, str) else None,
                data=result if isinstance(result, dict) else None,
            )
            return self.operative.response_model or result

        return result

    @abstractmethod
    async def _execute_operation(self) -> T:
        """Execute specific operation implementation."""
        pass

    @abstractmethod
    def validate(self) -> bool:
        """Validate form inputs and configuration."""
        pass


class PatternMatcher:
    """Tracks and matches operation patterns."""

    def __init__(self):
        self.patterns: dict[str, dict[str, Any]] = {}

    def add_pattern(
        self, pattern_id: str, config: dict[str, Any], success_score: float
    ):
        """Add a successful pattern."""
        self.patterns[pattern_id] = {"config": config, "score": success_score}

    def find_similar_pattern(self, config: dict[str, Any]) -> tuple[str, float] | None:
        """Find similar pattern and return (pattern_id, confidence)."""
        # Basic pattern matching - can be enhanced
        for pid, data in self.patterns.items():
            if (
                data["config"]["operation_type"] == config["operation_type"]
                and data["score"] > 0.8
            ):
                return pid, data["score"]
        return None
