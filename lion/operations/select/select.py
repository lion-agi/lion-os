import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Generic, List, Optional, Set, TypeVar

from pydantic import BaseModel, Field

from lion import Branch
from lion.core.communication.message import Note
from lion.core.forms.form import OperativeForm
from lion.protocols.operatives.instruct import InstructModel
from lion.protocols.operatives.reason import ReasonModel

from .prompt import PROMPT
from .utils import parse_selection, parse_to_representation

T = TypeVar("T")


class SelectionStrategy(Enum):
    """Strategies for making selections."""

    RANDOM = "random"
    WEIGHTED = "weighted"
    OPTIMIZED = "optimized"
    FILTERED = "filtered"


class SelectionStatus(Enum):
    """Status of selection operations."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SelectionMetrics:
    """Metrics for selection operations."""

    start_time: datetime
    end_time: datetime | None = None
    total_choices: int = 0
    selected_count: int = 0
    filtered_count: int = 0
    execution_time: float = 0.0
    average_score: float = 0.0


class SelectionModel(BaseModel):
    """Enhanced model representing the selection output."""

    selected: list[Any] = Field(default_factory=list)
    scores: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SelectionCache:
    """Cache manager for selection results."""

    def __init__(self, ttl: int = 3600):
        self._cache: dict[str, dict[str, Any]] = {}
        self._ttl = ttl

    def get(self, key: str) -> SelectionModel | None:
        """Retrieve cached selection result."""
        if key in self._cache:
            entry = self._cache[key]
            if (
                datetime.now() - datetime.fromisoformat(entry["timestamp"])
            ).total_seconds() <= self._ttl:
                return entry["result"]
            del self._cache[key]
        return None

    def set(self, key: str, result: SelectionModel) -> None:
        """Cache selection result."""
        self._cache[key] = {"result": result, "timestamp": datetime.now().isoformat()}


class SelectForm(OperativeForm, Generic[T]):
    """Enhanced form for selection operations with advanced features."""

    operation_type = "select"
    context: Note
    confidence_score: float
    reasoning: ReasonModel

    def __init__(
        self,
        choices: list[str],
        max_selections: int,
        selection_constraints: dict,
        guidance: str | None = None,
        strategy: SelectionStrategy = SelectionStrategy.OPTIMIZED,
        weights: dict[str, float] | None = None,
        filters: dict[str, Any] | None = None,
        cache_enabled: bool = True,
        cache_ttl: int = 3600,
        min_score: float = 0.5,
    ):
        """Initialize the select form with enhanced parameters.

        Args:
            choices: Available choices
            max_selections: Maximum number of selections
            selection_constraints: Constraints for selection
            guidance: Optional guidance for selection
            strategy: Selection strategy to use
            weights: Optional weights for choices
            filters: Optional filters to apply
            cache_enabled: Whether to enable caching
            cache_ttl: Cache time-to-live in seconds
            min_score: Minimum score for valid selections
        """
        super().__init__()
        self.choices = choices
        self.max_selections = max_selections
        self.selection_constraints = selection_constraints
        self.guidance = guidance
        self.strategy = strategy
        self.weights = weights or {}
        self.filters = filters or {}
        self.cache_enabled = cache_enabled
        self.cache = SelectionCache(cache_ttl) if cache_enabled else None
        self.min_score = min_score
        self.result: SelectionModel | None = None
        self.metrics: SelectionMetrics | None = None
        self.status = SelectionStatus.PENDING
        self.history: list[dict[str, Any]] = []

    async def execute(self) -> SelectionModel:
        """Execute the selection operation with enhanced control.

        Returns:
            SelectionModel: Selection results

        Raises:
            ValueError: If validation fails
            RuntimeError: If selection fails
        """
        try:
            start_time = datetime.now()
            self.status = SelectionStatus.PROCESSING

            # Check cache if enabled
            cache_key = self._generate_cache_key()
            if self.cache_enabled and self.cache:
                cached_result = self.cache.get(cache_key)
                if cached_result:
                    self._record_metrics(start_time, cached_result)
                    self._record_history("cache_hit")
                    return cached_result

            # Validate and filter choices
            validated_choices = self.validate_choices(self.choices)
            filtered_choices = self.apply_filters(validated_choices)

            # Make selections
            selected_choices = await self.collect_selections(filtered_choices)

            # Score selections
            scores = self._score_selections(selected_choices)

            # Create result model
            result = SelectionModel(
                selected=selected_choices,
                scores=scores,
                metadata={
                    "strategy": self.strategy.value,
                    "timestamp": datetime.now().isoformat(),
                },
            )

            # Cache result if enabled
            if self.cache_enabled and self.cache:
                self.cache.set(cache_key, result)

            self._record_metrics(start_time, result)
            self._record_history("success")

            self.result = result
            self.status = SelectionStatus.COMPLETED
            return result

        except Exception as e:
            self.status = SelectionStatus.FAILED
            self._record_history("error", error=str(e))
            raise RuntimeError(f"Selection operation failed: {str(e)}")

    def validate_choices(self, choices: list[str]) -> list[str]:
        """Validate choices against constraints.

        Args:
            choices: Choices to validate

        Returns:
            List[str]: Validated choices
        """
        min_length = self.selection_constraints.get("min_length", 0)
        max_length = self.selection_constraints.get("max_length", float("inf"))

        validated = [
            choice for choice in choices if min_length <= len(choice) <= max_length
        ]

        if not validated:
            raise ValueError("No choices passed validation")

        return validated

    def apply_filters(self, choices: list[str]) -> list[str]:
        """Apply filters to choices.

        Args:
            choices: Choices to filter

        Returns:
            List[str]: Filtered choices
        """
        filtered = choices

        for filter_name, filter_value in self.filters.items():
            if filter_name == "prefix":
                filtered = [c for c in filtered if c.startswith(filter_value)]
            elif filter_name == "suffix":
                filtered = [c for c in filtered if c.endswith(filter_value)]
            elif filter_name == "contains":
                filtered = [c for c in filtered if filter_value in c]
            elif filter_name == "regex":
                import re

                filtered = [c for c in filtered if re.match(filter_value, c)]

        if not filtered:
            raise ValueError("No choices passed filtering")

        return filtered

    async def collect_selections(self, choices: list[str]) -> list[str]:
        """Collect selections based on strategy.

        Args:
            choices: Available choices

        Returns:
            List[str]: Selected choices
        """
        instruct = InstructModel(
            instruction=self.guidance or "Please make a selection", context=choices
        )

        if self.strategy == SelectionStrategy.WEIGHTED:
            kwargs = {"weights": self.weights}
        else:
            kwargs = {}

        selection_result = await select(
            instruct=instruct,
            choices=choices,
            max_num_selections=self.max_selections,
            strategy=self.strategy,
            **kwargs,
        )

        return selection_result.selected

    def _score_selections(self, selections: list[str]) -> dict[str, float]:
        """Score the selected choices.

        Args:
            selections: Selected choices

        Returns:
            Dict[str, float]: Scores for selections
        """
        scores = {}
        for selection in selections:
            # Base score
            score = 1.0

            # Apply weight if available
            if selection in self.weights:
                score *= self.weights[selection]

            # Apply length penalty
            max_length = self.selection_constraints.get("max_length", float("inf"))
            if len(selection) > max_length * 0.8:  # 80% of max length
                score *= 0.9

            scores[selection] = score

        return scores

    def _generate_cache_key(self) -> str:
        """Generate cache key based on inputs.

        Returns:
            str: Cache key
        """
        content = f"{str(self.choices)}{self.max_selections}{self.strategy.value}"
        return hashlib.md5(content.encode()).hexdigest()

    def _record_metrics(self, start_time: datetime, result: SelectionModel) -> None:
        """Record metrics for the selection operation.

        Args:
            start_time: Operation start time
            result: Selection result
        """
        end_time = datetime.now()
        self.metrics = SelectionMetrics(
            start_time=start_time,
            end_time=end_time,
            total_choices=len(self.choices),
            selected_count=len(result.selected),
            filtered_count=len(self.choices) - len(result.selected),
            execution_time=(end_time - start_time).total_seconds(),
            average_score=(
                sum(result.scores.values()) / len(result.scores)
                if result.scores
                else 0.0
            ),
        )

    def _record_history(self, event_type: str, **kwargs) -> None:
        """Record an event in the operation history.

        Args:
            event_type: Type of event
            **kwargs: Additional event details
        """
        self.history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "event_type": event_type,
                "status": self.status.value,
                **kwargs,
            }
        )

    def save_session(self, filepath: str) -> None:
        """Save the selection session state and metrics.

        Args:
            filepath: Path to save the session data
        """
        session_data = {
            "status": self.status.value,
            "metrics": {
                "start_time": (
                    self.metrics.start_time.isoformat() if self.metrics else None
                ),
                "end_time": (
                    self.metrics.end_time.isoformat()
                    if self.metrics and self.metrics.end_time
                    else None
                ),
                "total_choices": self.metrics.total_choices if self.metrics else 0,
                "selected_count": self.metrics.selected_count if self.metrics else 0,
                "filtered_count": self.metrics.filtered_count if self.metrics else 0,
                "execution_time": self.metrics.execution_time if self.metrics else 0,
                "average_score": self.metrics.average_score if self.metrics else 0,
            },
            "history": self.history,
            "parameters": {
                "strategy": self.strategy.value,
                "cache_enabled": self.cache_enabled,
                "min_score": self.min_score,
            },
        }

        with open(filepath, "w") as f:
            json.dump(session_data, f, indent=2)

    def load_session(self, filepath: str) -> None:
        """Load a previously saved selection session.

        Args:
            filepath: Path to load the session data from
        """
        with open(filepath) as f:
            session_data = json.load(f)

        self.status = SelectionStatus(
            session_data.get("status", SelectionStatus.PENDING.value)
        )
        metrics_data = session_data.get("metrics", {})

        if metrics_data.get("start_time"):
            self.metrics = SelectionMetrics(
                start_time=datetime.fromisoformat(metrics_data["start_time"]),
                end_time=(
                    datetime.fromisoformat(metrics_data["end_time"])
                    if metrics_data.get("end_time")
                    else None
                ),
                total_choices=metrics_data.get("total_choices", 0),
                selected_count=metrics_data.get("selected_count", 0),
                filtered_count=metrics_data.get("filtered_count", 0),
                execution_time=metrics_data.get("execution_time", 0.0),
                average_score=metrics_data.get("average_score", 0.0),
            )

        self.history = session_data.get("history", [])
        params = session_data.get("parameters", {})
        self.strategy = SelectionStrategy(
            params.get("strategy", SelectionStrategy.OPTIMIZED.value)
        )
        self.cache_enabled = params.get("cache_enabled", True)
        self.min_score = params.get("min_score", 0.5)


async def select(
    instruct: InstructModel | dict[str, Any],
    choices: list[str] | type[Enum] | dict[str, Any],
    max_num_selections: int = 1,
    branch: Branch | None = None,
    branch_kwargs: dict[str, Any] | None = None,
    return_branch: bool = False,
    verbose: bool = False,
    strategy: SelectionStrategy = SelectionStrategy.OPTIMIZED,
    weights: dict[str, float] | None = None,
    **kwargs: Any,
) -> SelectionModel | tuple[SelectionModel, Branch]:
    """Perform an enhanced selection operation from given choices.

    Args:
        instruct: Instruction model or dictionary.
        choices: Options to select from.
        max_num_selections: Maximum selections allowed.
        branch: Existing branch or None to create a new one.
        branch_kwargs: Additional arguments for branch creation.
        return_branch: If True, return the branch with the selection.
        verbose: Whether to enable verbose output.
        strategy: Selection strategy to use.
        weights: Optional weights for choices.
        **kwargs: Additional keyword arguments.

    Returns:
        A SelectionModel instance, optionally with the branch.
    """
    if verbose:
        print(
            f"Starting selection with up to {max_num_selections} choices using {strategy.value} strategy."
        )

    branch = branch or Branch(**(branch_kwargs or {}))
    selections, contents = parse_to_representation(choices)

    # Apply weights if provided
    if strategy == SelectionStrategy.WEIGHTED and weights:
        weighted_selections = []
        weighted_contents = []
        for sel, cont in zip(selections, contents):
            weight = weights.get(sel, 1.0)
            weighted_selections.extend([sel] * int(weight * 10))
            weighted_contents.extend([cont] * int(weight * 10))
        selections = weighted_selections
        contents = weighted_contents

    prompt = PROMPT.format(max_num_selections=max_num_selections, choices=selections)

    if isinstance(instruct, InstructModel):
        instruct = instruct.clean_dump()

    instruct = instruct or {}

    if instruct.get("instruction", None) is not None:
        instruct["instruction"] = f"{instruct['instruction']}\n\n{prompt} \n\n "
    else:
        instruct["instruction"] = prompt

    context = instruct.get("context", None) or []
    context = [context] if not isinstance(context, list) else context
    context.extend([{k: v} for k, v in zip(selections, contents)])
    instruct["context"] = context

    response_model: SelectionModel = await branch.operate(
        operative_model=SelectionModel,
        **kwargs,
        **instruct,
    )
    if verbose:
        print(f"Received selection: {response_model.selected}")

    selected = response_model
    if isinstance(response_model, BaseModel) and hasattr(response_model, "selected"):
        selected = response_model.selected
    selected = [selected] if not isinstance(selected, list) else selected

    corrected_selections = [parse_selection(i, choices) for i in selected]

    if isinstance(response_model, BaseModel):
        response_model.selected = corrected_selections
        # Add scores if using weighted strategy
        if strategy == SelectionStrategy.WEIGHTED and weights:
            response_model.scores = {
                sel: weights.get(sel, 1.0) for sel in corrected_selections
            }

    elif isinstance(response_model, dict):
        response_model["selected"] = corrected_selections
        if strategy == SelectionStrategy.WEIGHTED and weights:
            response_model["scores"] = {
                sel: weights.get(sel, 1.0) for sel in corrected_selections
            }

    if return_branch:
        return response_model, branch
    return response_model
