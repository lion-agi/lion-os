from abc import ABC, abstractmethod
from enum import Enum

from lion.core.form import Form, Report
from lion.core.types import Any, BaseModel, Field, Generic, Note, TypeVar
from lion.operations.operations.brainstorm import brainstorm
from lion.operations.operations.plan import plan
from lion.operations.operations.select.select import SelectionModel, select
from lion.protocols.operatives import InstructModel
from lion.protocols.operatives.reason import ReasonModel

T = TypeVar("T")


class DecisionStage(str, Enum):
    """Stages in agent decision process."""

    INIT = "init"
    SELECTING = "selecting"
    BRAINSTORMING = "brainstorming"
    PLANNING = "planning"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentState(BaseModel):
    stage: DecisionStage = Field(default=DecisionStage.INIT)
    reasoning: dict[DecisionStage, ReasonModel] = Field(default_factory=dict)
    context: Note = Field(default_factory=Note)
    artifacts: dict[str, Any] = Field(default_factory=dict)


class AgentDecisionForm(Form, Generic[T], ABC):
    """Base class for agent decision forms."""

    state: AgentState = Field(default_factory=AgentState)

    @abstractmethod
    async def execute(self) -> T:
        """Execute the decision process."""
        pass

    def update_state(self, stage: DecisionStage, reasoning: ReasonModel | None = None):
        self.state.stage = stage
        if reasoning:
            self.state.reasoning[stage] = reasoning


class SelectiveAgentForm(AgentDecisionForm[SelectionModel]):
    """Form for selection-based decisions."""

    choices: list[Any] = Field(default_factory=list)
    max_selections: int = Field(default=1)

    async def execute(self) -> SelectionModel:
        self.update_state(DecisionStage.SELECTING)
        return await select(
            instruct={"context": self.state.context},
            choices=self.choices,
            max_num_selections=self.max_selections,
        )


class CreativeAgentForm(AgentDecisionForm[list[InstructModel]]):
    """Form for generating and executing creative ideas."""

    num_ideas: int = Field(default=3, description="Number of ideas to generate")
    auto_execute: bool = Field(
        default=True, description="Whether to execute generated ideas"
    )
    idea_filters: list[str] = Field(
        default_factory=list, description="Filters to apply to generated ideas"
    )

    async def execute(self) -> list[InstructModel]:
        """Generate and optionally execute creative ideas."""
        self.update_state(DecisionStage.BRAINSTORMING)

        # Run brainstorming operation
        result = await brainstorm(
            instruct={
                "guidance": self.state.context.get("guidance", ""),
                "context": self.state.context,
            },
            num_instruct=self.num_ideas,
            auto_run=self.auto_execute,
        )

        # Store generated ideas in artifacts
        if isinstance(result, (list, tuple)):
            self.state.artifacts["generated_ideas"] = result
        else:
            self.state.artifacts["generated_ideas"] = [result]

        # Apply filters if specified
        if self.idea_filters:
            filtered_results = []
            for idea in self.state.artifacts["generated_ideas"]:
                if all(self._apply_filter(idea, f) for f in self.idea_filters):
                    filtered_results.append(idea)
            result = filtered_results

        self.update_state(DecisionStage.COMPLETED if result else DecisionStage.FAILED)
        return result

    def _apply_filter(self, idea: InstructModel, filter_name: str) -> bool:
        """Apply a named filter to an idea."""
        if filter_name == "uniqueness":
            return not any(
                self._is_similar(idea, other)
                for other in self.state.artifacts.get("generated_ideas", [])
                if other != idea
            )
        elif filter_name == "relevance":
            return self._check_relevance(idea)
        return True

    def _is_similar(self, idea1: InstructModel, idea2: InstructModel) -> bool:
        """Check if two ideas are too similar."""
        return bool(
            idea1.guidance
            and idea2.guidance
            and idea1.guidance.lower() == idea2.guidance.lower()
        )

    def _check_relevance(self, idea: InstructModel) -> bool:
        """Check if an idea is relevant to the context."""
        context_keywords = (
            set(str(self.state.context).lower().split())
            if self.state.context
            else set()
        )
        idea_keywords = (
            set(str(idea.guidance).lower().split()) if idea.guidance else set()
        )
        return bool(context_keywords & idea_keywords)


class PlanningAgentForm(AgentDecisionForm[list[dict[str, Any]]]):
    """Form for creating and executing multi-step plans."""

    num_steps: int = Field(default=3, description="Number of plan steps")
    sequential: bool = Field(
        default=True, description="Whether steps must be executed in order"
    )
    dependencies: dict[str, list[str]] = Field(
        default_factory=dict, description="Step dependencies {step: [required_steps]}"
    )

    async def execute(self) -> list[dict[str, Any]]:
        """Create and execute a plan."""
        self.update_state(DecisionStage.PLANNING)

        # Generate plan
        plan_result = await plan(
            instruct={
                "guidance": self.state.context.get("guidance", ""),
                "context": self.state.context,
            },
            num_steps=self.num_steps,
        )

        self.state.artifacts["plan"] = plan_result

        # Execute plan steps
        if self.sequential:
            results = await self._execute_sequential()
        else:
            results = await self._execute_with_dependencies()

        self.update_state(DecisionStage.COMPLETED if results else DecisionStage.FAILED)
        return results

    async def _execute_sequential(self) -> list[dict[str, Any]]:
        """Execute plan steps in sequence."""
        results = []
        plan_steps = self.state.artifacts["plan"]

        for step in plan_steps:
            self.update_state(DecisionStage.EXECUTING)
            try:
                result = await self._execute_step(step)
                results.append({"step": step, "result": result, "status": "completed"})
            except Exception as e:
                results.append({"step": step, "error": str(e), "status": "failed"})
                if self.sequential:  # Stop on failure if sequential
                    break

        return results

    async def _execute_with_dependencies(self) -> list[dict[str, Any]]:
        """Execute plan steps respecting dependencies."""
        results = []
        plan_steps = self.state.artifacts["plan"]
        completed_steps = set()

        while len(results) < len(plan_steps):
            for step in plan_steps:
                step_id = str(step.get("id", ""))
                if step_id in completed_steps:
                    continue

                # Check dependencies
                deps = self.dependencies.get(step_id, [])
                if not all(dep in completed_steps for dep in deps):
                    continue

                self.update_state(DecisionStage.EXECUTING)
                try:
                    result = await self._execute_step(step)
                    results.append(
                        {"step": step, "result": result, "status": "completed"}
                    )
                    completed_steps.add(step_id)
                except Exception as e:
                    results.append({"step": step, "error": str(e), "status": "failed"})

        return results

    async def _execute_step(self, step: dict[str, Any]) -> Any:
        """Execute a single plan step."""
        if "instruct" in step:
            return await self._execute_instruction(step["instruct"])
        return step  # Return step as-is if no execution needed

    async def _execute_instruction(self, instruct: dict[str, Any]) -> Any:
        """Execute a step instruction."""
        # Implementation depends on your instruction execution system
        pass


class CompositeAgentForm(AgentDecisionForm[dict[str, Any]]):
    """Form for multi-stage decisions."""

    steps: list[AgentDecisionForm] = Field(default_factory=list)

    async def execute(self) -> dict[str, Any]:
        results = {}
        for step in self.steps:
            self.state.stage = step.state.stage
            result = await step.execute()
            results[step.state.stage] = result
            self.state.artifacts.update(step.state.artifacts)
        return results


class AgentDecisionReport(Report):
    """Report for tracking agent decisions and outcomes."""

    stage_outcomes: dict[DecisionStage, list[dict[str, Any]]] = Field(
        default_factory=lambda: {stage: [] for stage in DecisionStage},
        description="Detailed outcomes at each stage",
    )

    def add_decision_result(
        self,
        form: AgentDecisionForm,
        success: bool,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Track decision outcome with metadata."""
        outcome = {
            "success": success,
            "timestamp": self.current_time,
            "metadata": metadata or {},
            "artifacts": form.state.artifacts.copy(),
            "reasoning": form.state.reasoning.copy(),
        }
        self.stage_outcomes[form.state.stage].append(outcome)
        self.save_completed_form(form)

    def get_stage_stats(self) -> dict[str, dict[str, Any]]:
        """Get detailed statistics for each stage."""
        stats = {}

        for stage, outcomes in self.stage_outcomes.items():
            if outcomes:
                successes = sum(1 for o in outcomes if o["success"])
                total = len(outcomes)

                stats[stage.value] = {
                    "success_rate": successes / total,
                    "total_attempts": total,
                    "successful_attempts": successes,
                    "failed_attempts": total - successes,
                    "common_failures": self._analyze_failures(outcomes),
                    "avg_execution_time": self._calculate_avg_time(outcomes),
                }

        return stats

    def _analyze_failures(self, outcomes: list[dict[str, Any]]) -> dict[str, int]:
        """Analyze common failure patterns."""
        failure_counts = {}
        for outcome in outcomes:
            if not outcome["success"]:
                error = outcome["metadata"].get("error", "unknown")
                failure_counts[error] = failure_counts.get(error, 0) + 1
        return failure_counts

    def _calculate_avg_time(self, outcomes: list[dict[str, Any]]) -> float:
        """Calculate average execution time."""
        times = [o["metadata"].get("execution_time", 0) for o in outcomes]
        return sum(times) / len(times) if times else 0
