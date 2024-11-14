import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union

from lion.core.forms.form import Form
from lion.core.types import Field
from lion.libs.parse import to_dict


class UnitStatus(Enum):
    """Status of unit operations."""

    INITIALIZED = "initialized"
    VALIDATING = "validating"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    RECOVERED = "recovered"


class UnitPriority(Enum):
    """Priority levels for unit operations."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class UnitMetrics:
    """Metrics for unit operations."""

    start_time: datetime
    end_time: datetime | None = None
    execution_time: float = 0.0
    validation_time: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    error_count: int = 0
    recovery_count: int = 0


class UnitForm(Form):
    """Enhanced form for unit operations with advanced features."""

    answer: str | None = Field(
        None,
        description=(
            "Adhere to the prompt and all user instructions. Provide the answer "
            "for the task. If actions are required at this step, set "
            "`action_required` to True and write only `PLEASE_ACTION` to the answer field. "
            "Additionally, if extensions are allowed and needed at this step to provide a "
            "high-quality, accurate answer, set `extension_required` to True and "
            "you will have another chance to provide the answer after the actions are done."
        ),
    )

    extension_required: bool | None = Field(
        None,
        description=(
            "Set to True if more steps are needed to provide an accurate answer. "
            "If True, additional rounds are allowed."
        ),
        examples=[True, False],
    )

    prediction: str | None = Field(
        None,
        description="Provide the likely prediction based on context and instruction.",
    )

    plan: dict | str | None = Field(
        None,
        description=(
            "Provide a step-by-step plan. Format: {step_n: {plan: ..., reason: ...}}. "
            "Achieve the final answer at the last step. Set `extension_required` to True "
            "if the plan requires more steps."
        ),
        examples=["{step_1: {plan: '...', reason: '...'}}"],
    )

    next_steps: dict | str | None = Field(
        None,
        description=(
            "Brainstorm ideas on next actions to take. Format: {next_step_n: {plan: ..., reason: ...}}. "
            "Next steps are about anticipating future actions, not necessarily in sequential order. "
            "Set `extension_required` to True if more steps are needed."
        ),
        examples=["{next_step_1: {plan: '...', reason: '...'}}"],
    )

    score: float | None = Field(
        None,
        description=(
            "A numeric score. Higher is better. If not otherwise instructed, fill this field "
            "with your own performance rating. Be self-critical and strive for improvement."
        ),
        examples=[0.2, 5, 2.7],
    )

    reflection: str | None = Field(
        None,
        description=(
            "Reflect on your reasoning. Specify how you could improve to better achieve the task, "
            "or if the problem can be solved in a better way. Provide a better solution if possible "
            "and fill the necessary fields like `action_required`, `extension_required`, or `next_steps` as appropriate."
        ),
    )

    selection: Enum | str | list | None = Field(
        None,
        description="A single item from the provided choices.",
    )

    tool_schema: list | dict | None = Field(
        None,
        description="The list of tools available for use.",
    )

    assignment: str = Field("task -> answer")

    is_extension: bool = Field(False)

    def __init__(
        self,
        instruction: str = None,
        context: str = None,
        guidance: str = None,
        confidence: bool = None,
        tool_schema: list | dict | None = None,
        predict: bool = False,
        reason: bool = True,
        actions: dict = None,
        score: bool = True,
        select: bool = False,
        plan: bool = False,
        brainstorm: bool = False,
        reflect: bool = False,
        auto_run: bool = None,
        allow_action: bool = False,
        allow_extension: bool = False,
        max_extension: int | None = None,
        confidence_score: bool = False,
        score_num_digits: int | None = None,
        score_range: list | None = None,
        select_choices: list | None = None,
        plan_num_step: int | None = None,
        predict_num_sentences: int | None = None,
        priority: UnitPriority = UnitPriority.MEDIUM,
        max_retries: int = 3,
        timeout: float = 60.0,
        **kwargs,
    ):
        """Initialize the UnitForm with enhanced parameters.

        Args:
            instruction: Additional instruction
            context: Additional context
            guidance: Guidance for the task
            confidence: Include confidence score
            tool_schema: Schema of available tools
            predict: Include prediction
            reason: Include reasoning
            actions: Actions to be performed
            score: Include score
            select: Include selection from choices
            plan: Include a planning step
            brainstorm: Include brainstorming of next steps
            reflect: Include self-reflection
            auto_run: Automatically run nested instructions
            allow_action: Allow actions to be added
            allow_extension: Allow extension for more steps
            max_extension: Maximum number of extensions allowed
            confidence_score: Include a confidence score
            score_num_digits: Number of decimal places for the score
            score_range: Range for the score
            select_choices: Choices for selection
            plan_num_step: Number of steps in the plan
            predict_num_sentences: Number of sentences to predict
            priority: Priority level for the unit
            max_retries: Maximum number of retries
            timeout: Operation timeout in seconds
            **kwargs: Additional keyword arguments
        """
        super().__init__(**kwargs)

        self.task = (
            "Follow the prompt and provide the necessary output.\n"
            f"- Additional instruction: {instruction or 'N/A'}\n"
            f"- Additional context: {context or 'N/A'}\n"
        )

        self.validation_kwargs = {}
        self.priority = priority
        self.max_retries = max_retries
        self.timeout = timeout
        self.status = UnitStatus.INITIALIZED
        self.metrics: UnitMetrics | None = None
        self.history: list[dict[str, Any]] = []
        self.retry_count = 0

        if reason:
            self.append_to_request("reason")

        if allow_action:
            self.append_to_request("actions, action_required, reason")
            self.task += "- Reason and prepare actions with GIVEN TOOLS ONLY.\n"

        if allow_extension:
            self.append_to_request("extension_required")
            self.task += (
                f"- Allow auto-extension up to another {max_extension or 1} rounds.\n"
            )

        if tool_schema:
            self.append_to_input("tool_schema")
            self.tool_schema = tool_schema

        if brainstorm:
            self.append_to_request("next_steps, extension_required")
            self.task += "- Explore ideas on next actions to take.\n"

        if plan:
            plan_num_step = plan_num_step or 3
            max_extension = max_extension or plan_num_step
            self.append_to_request("plan, extension_required")
            self.task += (
                f"- Generate a {plan_num_step}-step plan based on the context.\n"
            )

        if predict:
            self.append_to_request("prediction")
            self.task += (
                f"- Predict the next {predict_num_sentences or 1} sentence(s).\n"
            )

        if select:
            self.append_to_request("selection")
            self.task += (
                f"- Select 1 item from the provided choices: {select_choices or []}.\n"
            )

        if confidence_score:
            self.append_to_request("confidence_score")

        if score:
            self.append_to_request("score")
            score_range = score_range or [0, 10]
            score_num_digits = score_num_digits or 0

            self.validation_kwargs["score"] = {
                "upper_bound": score_range[1],
                "lower_bound": score_range[0],
                "num_type": float if score_num_digits != 0 else int,
                "precision": score_num_digits if score_num_digits != 0 else None,
            }

            self.task += (
                f"- Provide a numeric score in [{score_range[0]}, {score_range[1]}] "
                f"with precision of {score_num_digits or 0} decimal places.\n"
            )

        if reflect:
            self.append_to_request("reflection")

        self.is_extension = allow_extension

    def append_to_request(self, fields: str) -> None:
        """Append fields to the request.

        Args:
            fields: Comma-separated field names to append
        """
        if not hasattr(self, "request_fields"):
            self.request_fields = set()
        field_list = [field.strip() for field in fields.split(",")]
        self.request_fields.update(field_list)

    def append_to_input(self, fields: str) -> None:
        """Append fields to the input.

        Args:
            fields: Comma-separated field names to append
        """
        if not hasattr(self, "input_fields"):
            self.input_fields = set()
        field_list = [field.strip() for field in fields.split(",")]
        self.input_fields.update(field_list)

    def to_dict(self) -> dict[str, Any]:
        """Convert the form to a dictionary.

        Returns:
            Dict[str, Any]: Form as dictionary
        """
        result = {}
        for field_name in self.__annotations__:
            value = getattr(self, field_name, None)
            if value is not None:
                result[field_name] = value
        return result

    def validate(self) -> None:
        """Validate the form fields with enhanced error handling.

        Raises:
            ValueError: If validation fails
        """
        try:
            self.status = UnitStatus.VALIDATING
            validation_start = datetime.now()

            if "score" in self.validation_kwargs:
                score = self.score
                if score is not None:
                    vkwargs = self.validation_kwargs["score"]
                    upper = vkwargs.get("upper_bound", float("inf"))
                    lower = vkwargs.get("lower_bound", float("-inf"))
                    num_type = vkwargs.get("num_type", float)
                    precision = vkwargs.get("precision", None)

                    if not (lower <= score <= upper):
                        raise ValueError(
                            f"Score {score} must be between {lower} and {upper}."
                        )

                    if not isinstance(score, num_type):
                        raise ValueError(f"Score {score} must be of type {num_type}.")

                    if precision is not None:
                        decimal_places = len(str(score).split(".")[-1])
                        if decimal_places > precision:
                            raise ValueError(
                                f"Score {score} must have at most {precision} decimal places."
                            )

            if hasattr(self, "metrics"):
                self.metrics.validation_time = (
                    datetime.now() - validation_start
                ).total_seconds()

        except Exception as e:
            self._record_history("validation_error", error=str(e))
            raise

    async def execute(self) -> Any:
        """Execute the form's task with enhanced control and monitoring.

        Returns:
            Any: Execution result

        Raises:
            RuntimeError: If execution fails
        """
        import asyncio
        import os

        import psutil

        start_time = datetime.now()
        self.metrics = UnitMetrics(start_time=start_time)
        process = psutil.Process(os.getpid())

        try:
            self.status = UnitStatus.EXECUTING
            self._record_history("execution_started")

            while self.retry_count < self.max_retries:
                try:
                    # Execute with timeout
                    async with asyncio.timeout(self.timeout):
                        # Implement your execution logic here
                        result = await self._execute_task()

                    # Update metrics
                    end_time = datetime.now()
                    self.metrics.end_time = end_time
                    self.metrics.execution_time = (
                        end_time - start_time
                    ).total_seconds()
                    self.metrics.memory_usage = (
                        process.memory_info().rss / 1024 / 1024
                    )  # MB
                    self.metrics.cpu_usage = process.cpu_percent()

                    self.status = UnitStatus.COMPLETED
                    self._record_history("execution_completed")
                    return result

                except TimeoutError:
                    self.retry_count += 1
                    self.metrics.error_count += 1
                    self._record_history("timeout_error")
                    if self.retry_count >= self.max_retries:
                        raise RuntimeError("Execution timed out after max retries")

                except Exception as e:
                    self.retry_count += 1
                    self.metrics.error_count += 1
                    self._record_history("execution_error", error=str(e))
                    if self.retry_count >= self.max_retries:
                        raise RuntimeError(
                            f"Execution failed after max retries: {str(e)}"
                        )

                    # Attempt recovery
                    if await self._attempt_recovery():
                        self.metrics.recovery_count += 1
                        continue
                    raise

        except Exception as e:
            self.status = UnitStatus.FAILED
            raise RuntimeError(f"Unit execution failed: {str(e)}")

    async def _execute_task(self) -> Any:
        """Execute the actual task implementation.

        Returns:
            Any: Task result

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement _execute_task")

    async def _attempt_recovery(self) -> bool:
        """Attempt to recover from execution failure.

        Returns:
            bool: True if recovery successful
        """
        try:
            # Implement recovery logic here
            self.status = UnitStatus.RECOVERED
            self._record_history("recovery_attempted")
            return True
        except Exception as e:
            self._record_history("recovery_failed", error=str(e))
            return False

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
        """Save the unit session state and metrics.

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
                "execution_time": self.metrics.execution_time if self.metrics else 0,
                "validation_time": self.metrics.validation_time if self.metrics else 0,
                "memory_usage": self.metrics.memory_usage if self.metrics else 0,
                "cpu_usage": self.metrics.cpu_usage if self.metrics else 0,
                "error_count": self.metrics.error_count if self.metrics else 0,
                "recovery_count": self.metrics.recovery_count if self.metrics else 0,
            },
            "history": self.history,
            "parameters": {
                "priority": self.priority.value,
                "max_retries": self.max_retries,
                "timeout": self.timeout,
                "retry_count": self.retry_count,
            },
        }

        with open(filepath, "w") as f:
            json.dump(session_data, f, indent=2)

    def load_session(self, filepath: str) -> None:
        """Load a previously saved unit session.

        Args:
            filepath: Path to load the session data from
        """
        with open(filepath) as f:
            session_data = json.load(f)

        self.status = UnitStatus(
            session_data.get("status", UnitStatus.INITIALIZED.value)
        )
        metrics_data = session_data.get("metrics", {})

        if metrics_data.get("start_time"):
            self.metrics = UnitMetrics(
                start_time=datetime.fromisoformat(metrics_data["start_time"]),
                end_time=(
                    datetime.fromisoformat(metrics_data["end_time"])
                    if metrics_data.get("end_time")
                    else None
                ),
                execution_time=metrics_data.get("execution_time", 0.0),
                validation_time=metrics_data.get("validation_time", 0.0),
                memory_usage=metrics_data.get("memory_usage", 0.0),
                cpu_usage=metrics_data.get("cpu_usage", 0.0),
                error_count=metrics_data.get("error_count", 0),
                recovery_count=metrics_data.get("recovery_count", 0),
            )

        self.history = session_data.get("history", [])
        params = session_data.get("parameters", {})
        self.priority = UnitPriority(params.get("priority", UnitPriority.MEDIUM.value))
        self.max_retries = params.get("max_retries", 3)
        self.timeout = params.get("timeout", 60.0)
        self.retry_count = params.get("retry_count", 0)
