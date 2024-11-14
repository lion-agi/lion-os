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


class PlanStatus(Enum):
    """Status of plan execution."""

    CREATED = "created"
    VALIDATING = "validating"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class StepPriority(Enum):
    """Priority levels for plan steps."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class PlanMetrics:
    """Metrics for plan execution."""

    start_time: datetime
    end_time: datetime | None = None
    total_steps: int = 0
    completed_steps: int = 0
    failed_steps: int = 0
    total_execution_time: float = 0.0
    average_step_time: float = 0.0


class PlanStep:
    """Enhanced representation of a plan step."""

    def __init__(
        self,
        instruction: InstructModel,
        priority: StepPriority = StepPriority.MEDIUM,
        timeout: float = 60.0,
        retry_count: int = 3,
        dependencies: set[str] = None,
    ):
        self.instruction = instruction
        self.priority = priority
        self.timeout = timeout
        self.retry_count = retry_count
        self.dependencies = dependencies or set()
        self.status = PlanStatus.CREATED
        self.start_time: datetime | None = None
        self.end_time: datetime | None = None
        self.result: Any = None
        self.error: Exception | None = None
        self.attempt = 0

    @property
    def execution_time(self) -> float | None:
        """Calculate execution time in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None


async def run_step(
    step: PlanStep,
    session: Session,
    branch: Branch,
    verbose: bool = False,
    **kwargs: Any,
) -> Any:
    """Execute a single step of the plan with enhanced control.

    Args:
        step: The plan step to execute.
        session: The current session.
        branch: The branch to operate on.
        verbose: Whether to enable verbose output.
        **kwargs: Additional keyword arguments.

    Returns:
        The result of the step execution.

    Raises:
        RuntimeError: If step execution fails
    """
    if verbose:
        guidance_preview = (
            step.instruction.guidance[:100] + "..."
            if len(step.instruction.guidance) > 100
            else step.instruction.guidance
        )
        print(f"Executing step: {guidance_preview}")

    step.start_time = datetime.now()
    step.status = PlanStatus.EXECUTING

    while step.attempt < step.retry_count:
        try:
            step.attempt += 1
            config = {**step.instruction.model_dump(), **kwargs}

            # Execute with timeout
            async with asyncio.timeout(step.timeout):
                res = await branch.operate(**config)
                branch.msgs.logger.dump()

            step.status = PlanStatus.COMPLETED
            step.result = res
            step.end_time = datetime.now()
            return res

        except TimeoutError:
            if step.attempt >= step.retry_count:
                step.status = PlanStatus.FAILED
                step.error = RuntimeError(
                    f"Step timed out after {step.retry_count} attempts"
                )
                raise step.error

        except Exception as e:
            if step.attempt >= step.retry_count:
                step.status = PlanStatus.FAILED
                step.error = e
                raise RuntimeError(
                    f"Step failed after {step.retry_count} attempts: {str(e)}"
                )


async def plan(
    instruct: InstructModel | dict[str, Any],
    num_steps: int = 3,
    session: Session | None = None,
    branch: Branch | ID.Ref | None = None,
    auto_run: bool = True,
    branch_kwargs: dict[str, Any] | None = None,
    return_session: bool = False,
    verbose: bool = False,
    optimize_plan: bool = True,
    **kwargs: Any,
) -> Any:
    """Create and execute a multi-step plan with enhanced control and optimization.

    Args:
        instruct: Instruction model or dictionary.
        num_steps: Number of steps in the plan.
        session: Existing session or None to create a new one.
        branch: Existing branch or reference.
        auto_run: If True, automatically run the steps.
        branch_kwargs: Additional keyword arguments for branch creation.
        return_session: If True, return the session along with results.
        verbose: Whether to enable verbose output.
        optimize_plan: Whether to optimize the plan before execution.
        **kwargs: Additional keyword arguments.

    Returns:
        Results of the plan execution, optionally with the session and metrics.
    """
    metrics = PlanMetrics(start_time=datetime.now())

    try:
        if verbose:
            print(f"Planning execution with {num_steps} steps...")

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
        instruct["guidance"] = f"\n{PROMPT.format(num_steps=num_steps)}\n{guidance}"

        # Generate initial plan
        res1 = await branch.operate(**instruct, **kwargs)
        if verbose:
            print("Initial planning complete.")

        if not auto_run:
            metrics.end_time = datetime.now()
            if return_session:
                return res1, session, metrics
            return res1, metrics

        # Convert results to PlanStep objects
        steps = []
        if hasattr(res1, "instruct_models"):
            instructs: list[InstructModel] = res1.instruct_models
            steps = [PlanStep(ins) for ins in instructs]
            metrics.total_steps = len(steps)

            # Optimize plan if enabled
            if optimize_plan:
                steps = _optimize_plan(steps)

            # Execute steps
            results = [res1]
            for i, step in enumerate(steps, 1):
                if verbose:
                    print(f"\nExecuting step {i}/{len(steps)}")
                try:
                    res = await run_step(
                        step, session, branch, verbose=verbose, **kwargs
                    )
                    results.append(res)
                    metrics.completed_steps += 1
                    if step.execution_time:
                        metrics.total_execution_time += step.execution_time
                except Exception as e:
                    metrics.failed_steps += 1
                    raise RuntimeError(f"Plan execution failed at step {i}: {str(e)}")

            if verbose:
                print("\nAll steps completed successfully!")

            # Update metrics
            metrics.end_time = datetime.now()
            if metrics.completed_steps > 0:
                metrics.average_step_time = (
                    metrics.total_execution_time / metrics.completed_steps
                )

            if return_session:
                return results, session, metrics
            return results, metrics

        metrics.end_time = datetime.now()
        if return_session:
            return res1, session, metrics
        return res1, metrics

    except Exception as e:
        metrics.end_time = datetime.now()
        raise RuntimeError(f"Plan execution failed: {str(e)}")


def _optimize_plan(steps: list[PlanStep]) -> list[PlanStep]:
    """Optimize plan execution order based on dependencies and priorities.

    Args:
        steps: List of plan steps

    Returns:
        List[PlanStep]: Optimized step order
    """
    # Group steps by priority
    priority_groups = {
        StepPriority.HIGH: [],
        StepPriority.MEDIUM: [],
        StepPriority.LOW: [],
    }

    for step in steps:
        priority_groups[step.priority].append(step)

    # Maintain dependency order within priority groups
    optimized_steps = []
    for priority in StepPriority:
        group_steps = priority_groups[priority]
        # Sort by dependencies (steps with fewer dependencies first)
        group_steps.sort(key=lambda s: len(s.dependencies))
        optimized_steps.extend(group_steps)

    return optimized_steps


class PlanForm(OperativeForm, Generic[T]):
    """Enhanced form for plan operations with advanced control and monitoring."""

    operation_type = "plan"
    context: Note
    confidence_score: float
    reasoning: ReasonModel

    def __init__(
        self,
        goal_description: str,
        constraints: dict[str, Any] | None = None,
        dependencies: dict[str, set[str]] | None = None,
        guidance: str | None = None,
        optimize_plan: bool = True,
        validate_steps: bool = True,
        rollback_on_failure: bool = True,
        step_timeout: float = 60.0,
        max_retries: int = 3,
    ):
        """Initialize the plan form with enhanced parameters.

        Args:
            goal_description: Description of the plan's goal
            constraints: Optional constraints for the plan
            dependencies: Optional step dependencies
            guidance: Optional guidance for plan creation
            optimize_plan: Whether to optimize the plan
            validate_steps: Whether to validate steps
            rollback_on_failure: Whether to rollback on failure
            step_timeout: Timeout for each step
            max_retries: Maximum retries per step
        """
        super().__init__()
        self.goal_description = goal_description
        self.constraints = constraints or {}
        self.dependencies = dependencies or {}
        self.guidance = guidance
        self.optimize_plan = optimize_plan
        self.validate_steps = validate_steps
        self.rollback_on_failure = rollback_on_failure
        self.step_timeout = step_timeout
        self.max_retries = max_retries
        self.result: list[InstructModel] | None = None
        self.metrics: PlanMetrics | None = None
        self.status = PlanStatus.CREATED

    async def execute(self) -> list[InstructModel]:
        """Execute the plan with enhanced control and monitoring.

        Returns:
            List[InstructModel]: Plan execution results

        Raises:
            ValueError: If plan validation fails
            RuntimeError: If plan execution fails
        """
        try:
            # Create plan instruction
            instruct = InstructModel(
                instruction=self.guidance or "Create a plan to achieve the goal",
                context=self.goal_description,
            )

            # Execute plan
            plan_result, metrics = await plan(
                instruct=instruct,
                num_steps=len(self.dependencies) or 3,
                auto_run=True,
                optimize_plan=self.optimize_plan,
                verbose=False,
            )

            self.metrics = metrics
            self.result = plan_result
            self.status = PlanStatus.COMPLETED
            return self.result

        except Exception as e:
            self.status = PlanStatus.FAILED
            if self.rollback_on_failure:
                await self._rollback()
            raise RuntimeError(f"Plan execution failed: {str(e)}")

    async def _rollback(self) -> None:
        """Rollback completed steps in reverse order."""
        if self.result and isinstance(self.result, list):
            for step in reversed(self.result):
                if hasattr(step, "rollback"):
                    try:
                        await step.rollback()
                    except Exception as e:
                        print(f"Rollback failed for step: {str(e)}")
        self.status = PlanStatus.ROLLED_BACK

    def save_session(self, filepath: str) -> None:
        """Save the plan session state and metrics.

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
                "total_steps": self.metrics.total_steps if self.metrics else 0,
                "completed_steps": self.metrics.completed_steps if self.metrics else 0,
                "failed_steps": self.metrics.failed_steps if self.metrics else 0,
                "total_execution_time": (
                    self.metrics.total_execution_time if self.metrics else 0
                ),
                "average_step_time": (
                    self.metrics.average_step_time if self.metrics else 0
                ),
            },
            "parameters": {
                "optimize_plan": self.optimize_plan,
                "validate_steps": self.validate_steps,
                "rollback_on_failure": self.rollback_on_failure,
                "step_timeout": self.step_timeout,
                "max_retries": self.max_retries,
            },
        }

        with open(filepath, "w") as f:
            json.dump(session_data, f, indent=2)

    def load_session(self, filepath: str) -> None:
        """Load a previously saved plan session.

        Args:
            filepath: Path to load the session data from
        """
        with open(filepath) as f:
            session_data = json.load(f)

        self.status = PlanStatus(session_data.get("status", PlanStatus.CREATED.value))
        metrics_data = session_data.get("metrics", {})

        if metrics_data.get("start_time"):
            self.metrics = PlanMetrics(
                start_time=datetime.fromisoformat(metrics_data["start_time"]),
                end_time=(
                    datetime.fromisoformat(metrics_data["end_time"])
                    if metrics_data.get("end_time")
                    else None
                ),
                total_steps=metrics_data.get("total_steps", 0),
                completed_steps=metrics_data.get("completed_steps", 0),
                failed_steps=metrics_data.get("failed_steps", 0),
                total_execution_time=metrics_data.get("total_execution_time", 0.0),
                average_step_time=metrics_data.get("average_step_time", 0.0),
            )

        params = session_data.get("parameters", {})
        self.optimize_plan = params.get("optimize_plan", True)
        self.validate_steps = params.get("validate_steps", True)
        self.rollback_on_failure = params.get("rollback_on_failure", True)
        self.step_timeout = params.get("step_timeout", 60.0)
        self.max_retries = params.get("max_retries", 3)
