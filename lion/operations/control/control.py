import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Generic, List, Optional, Set, TypeVar

from lion.core.communication.message import Note
from lion.core.forms.form import OperativeForm
from lion.core.session.branch import Branch
from lion.core.session.session import Session
from lion.core.types import ID
from lion.protocols.operatives.instruct import INSTRUCT_MODEL_FIELD, InstructModel
from lion.protocols.operatives.reason import ReasonModel

from .prompt import PROMPT

T = TypeVar("T")


class ControlState(Enum):
    """States for control operations."""

    INITIALIZED = "initialized"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class ControlPolicy(Enum):
    """Control policies for operation execution."""

    STRICT = "strict"  # Enforce all validations strictly
    LENIENT = "lenient"  # Allow some validation failures
    ADAPTIVE = "adaptive"  # Adapt policy based on context


@dataclass
class ControlMetrics:
    """Metrics for control operations."""

    start_time: datetime
    end_time: datetime | None = None
    operation_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    total_execution_time: float = 0.0
    average_response_time: float = 0.0
    throttle_count: int = 0


async def run_control(
    ins: InstructModel,
    session: Session,
    branch: Branch,
    verbose: bool = False,
    policy: ControlPolicy = ControlPolicy.STRICT,
    throttle_rate: float = 0.0,
    **kwargs: Any,
) -> Any:
    """Execute a control operation within the session with enhanced control.

    Args:
        ins: The instruction model to run.
        session: The current session.
        branch: The branch to operate on.
        verbose: Whether to enable verbose output.
        policy: Control policy to apply.
        throttle_rate: Rate limiting in seconds.
        **kwargs: Additional keyword arguments.

    Returns:
        The result of the control operation.

    Raises:
        ValueError: If validation fails under strict policy
        RuntimeError: If operation execution fails
    """
    if verbose:
        guidance_preview = (
            ins.guidance[:100] + "..." if len(ins.guidance) > 100 else ins.guidance
        )
        print(f"Running control operation: {guidance_preview}")

    # Apply throttling if specified
    if throttle_rate > 0:
        await asyncio.sleep(throttle_rate)

    try:
        # Validate instruction based on policy
        if policy == ControlPolicy.STRICT:
            _validate_instruction(ins)
        elif policy == ControlPolicy.ADAPTIVE:
            policy = _determine_policy(ins)
            if policy == ControlPolicy.STRICT:
                _validate_instruction(ins)

        config = {**ins.model_dump(), **kwargs}
        res = await branch.operate(**config)
        branch.msgs.logger.dump()
        return res

    except Exception as e:
        if policy == ControlPolicy.STRICT:
            raise
        print(f"Warning: Control operation error (non-strict policy): {str(e)}")
        return None


def _validate_instruction(ins: InstructModel) -> None:
    """Validate control instruction.

    Args:
        ins: Instruction to validate

    Raises:
        ValueError: If validation fails
    """
    if not ins.guidance:
        raise ValueError("Empty guidance in control instruction")
    if not hasattr(ins, "context") or not ins.context:
        raise ValueError("Missing context in control instruction")


def _determine_policy(ins: InstructModel) -> ControlPolicy:
    """Determine appropriate control policy based on context.

    Args:
        ins: Instruction to analyze

    Returns:
        ControlPolicy: Determined policy
    """
    # Implement policy determination logic
    return ControlPolicy.STRICT


async def control(
    instruct: InstructModel | dict[str, Any],
    session: Session | None = None,
    branch: Branch | ID.Ref | None = None,
    auto_run: bool = True,
    branch_kwargs: dict[str, Any] | None = None,
    return_session: bool = False,
    verbose: bool = False,
    policy: ControlPolicy = ControlPolicy.STRICT,
    throttle_rate: float = 0.0,
    **kwargs: Any,
) -> Any:
    """Perform a control operation with enhanced management and monitoring.

    Args:
        instruct: Instruction model or dictionary.
        session: Existing session or None to create a new one.
        branch: Existing branch or reference.
        auto_run: If True, automatically run nested instructions.
        branch_kwargs: Additional arguments for branch creation.
        return_session: If True, return the session with results.
        verbose: Whether to enable verbose output.
        policy: Control policy to apply.
        throttle_rate: Rate limiting in seconds.
        **kwargs: Additional keyword arguments.

    Returns:
        The results of the control operation, optionally with the session.

    Raises:
        ValueError: If input validation fails
        RuntimeError: If operation execution fails
    """
    metrics = ControlMetrics(start_time=datetime.now())

    try:
        if verbose:
            print("Starting control operation.")

        field_models: list = kwargs.get("field_models", [])
        if INSTRUCT_MODEL_FIELD not in field_models:
            field_models.append(INSTRUCT_MODEL_FIELD)
        kwargs["field_models"] = field_models

        session = session or Session()
        branch = branch or session.new_branch(**(branch_kwargs or {}))

        if isinstance(instruct, InstructModel):
            instruct = instruct.clean_dump()
        if not isinstance(instruct, dict):
            raise ValueError(
                "instruct needs to be an InstructModel object or a dictionary of valid parameters"
            )

        guidance = instruct.get("guidance", "")
        instruct["guidance"] = f"\n{PROMPT}\n{guidance}"

        operation_start = datetime.now()
        res1 = await branch.operate(**instruct, **kwargs)
        metrics.operation_count += 1
        metrics.total_execution_time += (
            datetime.now() - operation_start
        ).total_seconds()

        if verbose:
            print("Initial control operation complete.")

        if not auto_run:
            metrics.success_count += 1
            metrics.end_time = datetime.now()
            metrics.average_response_time = (
                metrics.total_execution_time / metrics.operation_count
            )

            if return_session:
                return (res1, session), metrics
            return res1, metrics

        results = res1 if isinstance(res1, list) else [res1]
        if hasattr(res1, "instruct_models"):
            instructs: list[InstructModel] = res1.instruct_models
            for i, ins in enumerate(instructs, 1):
                if verbose:
                    print(f"\nExecuting control step {i}/{len(instructs)}")
                try:
                    operation_start = datetime.now()
                    res = await run_control(
                        ins,
                        session,
                        branch,
                        verbose=verbose,
                        policy=policy,
                        throttle_rate=throttle_rate,
                        **kwargs,
                    )
                    metrics.operation_count += 1
                    metrics.success_count += 1
                    metrics.total_execution_time += (
                        datetime.now() - operation_start
                    ).total_seconds()
                    results.append(res)
                except Exception as e:
                    metrics.failure_count += 1
                    if policy == ControlPolicy.STRICT:
                        raise RuntimeError(f"Control step {i} failed: {str(e)}")
                    print(
                        f"Warning: Control step {i} failed (non-strict policy): {str(e)}"
                    )

            if verbose:
                print("\nAll control steps completed successfully!")

        metrics.end_time = datetime.now()
        metrics.average_response_time = (
            metrics.total_execution_time / metrics.operation_count
        )

        if return_session:
            return (results, session), metrics
        return results, metrics

    except Exception as e:
        metrics.end_time = datetime.now()
        metrics.failure_count += 1
        raise RuntimeError(f"Control operation failed: {str(e)}")


class ControlForm(OperativeForm, Generic[T]):
    """Enhanced form for control operations with state management and monitoring."""

    operation_type = "control"
    context: Note
    confidence_score: float
    reasoning: ReasonModel

    def __init__(
        self,
        context: Note,
        guidance: str | None = None,
        policy: ControlPolicy = ControlPolicy.STRICT,
        throttle_rate: float = 0.0,
        max_retries: int = 3,
        timeout: float = 60.0,
    ):
        """Initialize the control form with enhanced parameters.

        Args:
            context: The context for control operations
            guidance: Optional guidance for operations
            policy: Control policy to apply
            throttle_rate: Rate limiting in seconds
            max_retries: Maximum number of retries
            timeout: Operation timeout in seconds
        """
        super().__init__()
        self.context = context
        self.guidance = guidance
        self.policy = policy
        self.throttle_rate = throttle_rate
        self.max_retries = max_retries
        self.timeout = timeout
        self.result: Any | None = None
        self.state = ControlState.INITIALIZED
        self.metrics: ControlMetrics | None = None
        self.history: list[dict[str, Any]] = []

    async def execute(self) -> Any:
        """Execute the control operation with enhanced error handling and monitoring.

        Returns:
            Any: Control operation result

        Raises:
            RuntimeError: If operation execution fails
        """
        self.state = ControlState.RUNNING
        retry_count = 0

        while retry_count < self.max_retries:
            try:
                # Create instruction
                instruct = InstructModel(
                    instruction=self.guidance
                    or "Please perform a control operation based on the context",
                    context=self.context,
                )

                # Execute with timeout
                async with asyncio.timeout(self.timeout):
                    control_result, metrics = await control(
                        instruct=instruct,
                        auto_run=False,
                        verbose=False,
                        policy=self.policy,
                        throttle_rate=self.throttle_rate,
                    )

                self.metrics = metrics
                self.result = control_result
                self.state = ControlState.COMPLETED

                # Record in history
                self._record_history("success")

                return self.result

            except TimeoutError:
                retry_count += 1
                self._record_history("timeout")
                if retry_count >= self.max_retries:
                    self.state = ControlState.FAILED
                    raise RuntimeError("Control operation timed out after max retries")

            except Exception as e:
                retry_count += 1
                self._record_history("error", error=str(e))
                if retry_count >= self.max_retries:
                    self.state = ControlState.FAILED
                    raise RuntimeError(
                        f"Control operation failed after max retries: {str(e)}"
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
                "state": self.state.value,
                **kwargs,
            }
        )

    def save_session(self, filepath: str) -> None:
        """Save the control session state and metrics.

        Args:
            filepath: Path to save the session data
        """
        session_data = {
            "state": self.state.value,
            "metrics": {
                "start_time": (
                    self.metrics.start_time.isoformat() if self.metrics else None
                ),
                "end_time": (
                    self.metrics.end_time.isoformat()
                    if self.metrics and self.metrics.end_time
                    else None
                ),
                "operation_count": self.metrics.operation_count if self.metrics else 0,
                "success_count": self.metrics.success_count if self.metrics else 0,
                "failure_count": self.metrics.failure_count if self.metrics else 0,
                "total_execution_time": (
                    self.metrics.total_execution_time if self.metrics else 0
                ),
                "average_response_time": (
                    self.metrics.average_response_time if self.metrics else 0
                ),
                "throttle_count": self.metrics.throttle_count if self.metrics else 0,
            },
            "history": self.history,
            "parameters": {
                "policy": self.policy.value,
                "throttle_rate": self.throttle_rate,
                "max_retries": self.max_retries,
                "timeout": self.timeout,
            },
        }

        with open(filepath, "w") as f:
            json.dump(session_data, f, indent=2)

    def load_session(self, filepath: str) -> None:
        """Load a previously saved control session.

        Args:
            filepath: Path to load the session data from
        """
        with open(filepath) as f:
            session_data = json.load(f)

        self.state = ControlState(
            session_data.get("state", ControlState.INITIALIZED.value)
        )
        metrics_data = session_data.get("metrics", {})

        if metrics_data.get("start_time"):
            self.metrics = ControlMetrics(
                start_time=datetime.fromisoformat(metrics_data["start_time"]),
                end_time=(
                    datetime.fromisoformat(metrics_data["end_time"])
                    if metrics_data.get("end_time")
                    else None
                ),
                operation_count=metrics_data.get("operation_count", 0),
                success_count=metrics_data.get("success_count", 0),
                failure_count=metrics_data.get("failure_count", 0),
                total_execution_time=metrics_data.get("total_execution_time", 0.0),
                average_response_time=metrics_data.get("average_response_time", 0.0),
                throttle_count=metrics_data.get("throttle_count", 0),
            )

        self.history = session_data.get("history", [])
        params = session_data.get("parameters", {})
        self.policy = ControlPolicy(params.get("policy", ControlPolicy.STRICT.value))
        self.throttle_rate = params.get("throttle_rate", 0.0)
        self.max_retries = params.get("max_retries", 3)
        self.timeout = params.get("timeout", 60.0)
