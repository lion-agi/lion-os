import asyncio
from abc import abstractmethod
from enum import Enum

from pydantic import Field

from lion.core.form import Form, Report
from lion.core.forms.agent_forms import (
    AgentDecisionForm,
    AgentState,
    CreativeAgentForm,
    DecisionStage,
    PlanningAgentForm,
    SelectiveAgentForm,
)
from lion.core.operatives.agent_operatives import AgentOperative
from lion.core.types import Any, Generic, Note, T, TypeVar
from lion.protocols.operatives import InstructModel
from lion.protocols.operatives.reason import ReasonModel

from .operations.brainstorm import brainstorm
from .operations.composite.base import OperativeForm as BaseOperativeForm
from .operations.plan import plan
from .operations.select.select import SelectionModel, select


class OperationType(str, Enum):
    SELECT = "select"
    BRAINSTORM = "brainstorm"
    PLAN = "plan"
    COMPOSITE = "composite"
    AGENT_DECISION = "agent_decision"
    AGENT_OPERATIVE = "agent_operative"


class OperativeForm(BaseOperativeForm[T]):
    """Concrete implementation of base operative form."""

    operation_type: OperationType
    context: Note = Field(default_factory=Note)
    confidence_score: float = Field(default=0.0)
    reasoning: ReasonModel | None = None
    result: T | None = None
    decision_stage: DecisionStage | None = None
    agent_state: AgentState | None = None
    agent_operative: AgentOperative | None = None

    async def execute(self) -> T:
        if not self.is_workable():
            raise ValueError("Missing required inputs")
        try:
            self.result = await self._execute_operation()
            await self._validate_result()
            return self.result
        except Exception as e:
            self.metadata["error"] = str(e)
            raise

    async def _execute_operation(self) -> T:
        if isinstance(self.operative, AgentDecisionForm):
            self.agent_state = self.operative.state
            return await self.operative.execute()
        if self.operation_type == OperationType.AGENT_DECISION:
            decision_form = self.get_decision_form()
            return await decision_form.make_decision()
        if self.operation_type == OperationType.AGENT_OPERATIVE:
            if not self.agent_operative:
                raise ValueError("No agent operative configured")
            return await self.agent_operative.execute()
        raise NotImplementedError

    async def _validate_result(self) -> None:
        """Default result validation."""
        if self.operative and not self.operative.response_model:
            raise ValueError("Operation result validation failed")

    def get_decision_form(self) -> AgentDecisionForm:
        # Implementation for creating appropriate decision form
        pass


class SelectForm(OperativeForm[SelectionModel]):
    """Form for selection operations."""

    operation_type: OperationType = OperationType.SELECT
    choices: list[str] | type[Enum] | dict[str, Any]
    max_selections: int = Field(default=1, gt=0)

    async def _execute_operation(self) -> SelectionModel:
        instruct = InstructModel(guidance=self.guidance, context=self.context)
        return await select(
            instruct=instruct,
            choices=self.choices,
            max_num_selections=self.max_selections,
            agent_form=(
                self.operative
                if isinstance(self.operative, SelectiveAgentForm)
                else None
            ),
            use_agent_decision=self.decision_stage is not None,
            verbose=self.metadata.get("verbose", False),
        )


class BrainstormForm(OperativeForm[list[Any]]):
    """Form for brainstorming operations."""

    operation_type: OperationType = OperationType.BRAINSTORM
    num_ideas: int = Field(default=3, gt=0)

    async def _execute_operation(self) -> list[Any]:
        instruct = InstructModel(guidance=self.guidance, context=self.context)
        return await brainstorm(
            instruct=instruct,
            num_instruct=self.num_ideas,
            auto_run=self.metadata.get("auto_run", True),
            verbose=self.metadata.get("verbose", False),
        )


class PlanForm(OperativeForm[list[Any]]):
    """Form for planning operations."""

    operation_type: OperationType = OperationType.PLAN
    num_steps: int = Field(default=3, gt=0)

    async def _execute_operation(self) -> list[Any]:
        instruct = InstructModel(guidance=self.guidance, context=self.context)
        return await plan(
            instruct=instruct,
            num_steps=self.num_steps,
            auto_run=self.metadata.get("auto_run", True),
            verbose=self.metadata.get("verbose", False),
        )


class CompositeForm(OperativeForm[Any]):
    """Form for composite operations combining multiple forms."""

    operation_type: OperationType = OperationType.COMPOSITE
    sub_forms: list[OperativeForm] = Field(default_factory=list)
    sequential: bool = Field(default=True)

    async def _execute_operation(self) -> Any:
        results = []

        if self.sequential:
            for form in self.sub_forms:
                form.context.update(self.context)
                result = await form.execute()
                results.append(result)
                # Update context with previous results
                self.context["previous_result"] = result
        else:
            # Execute forms concurrently
            tasks = [form.execute() for form in self.sub_forms]
            results = await asyncio.gather(*tasks)

        return results

    def add_form(self, form: OperativeForm) -> None:
        """Add a sub-form to the composite operation."""
        self.sub_forms.append(form)

    def remove_form(self, index: int) -> None:
        """Remove a sub-form by index."""
        if 0 <= index < len(self.sub_forms):
            self.sub_forms.pop(index)
