from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Type

import yaml

from lion.core.forms.agent_forms import (
    AgentDecisionForm,
    CreativeAgentForm,
    PlanningAgentForm,
    SelectiveAgentForm,
)
from lion.core.types import Note
from lion.protocols.operatives import InstructModel


class AgentOperative(ABC):
    """Base class for agent operatives."""

    def __init__(self, form_class: type[AgentDecisionForm], **config):
        self.form_class = form_class
        self.config = config
        self.context = Note()

    @abstractmethod
    async def execute(self) -> Any:
        """Execute the operative's decision process."""
        pass

    def extend_context(self, **kwargs):
        """Add additional context."""
        self.context.update(kwargs)


class SelectAgentOperative(AgentOperative):
    def __init__(self, **config):
        super().__init__(SelectiveAgentForm, **config)

    async def execute(self):
        form = self.form_class(
            choices=self.config["choices"],
            max_selections=self.config.get("max_selections", 1),
            context=self.context,
        )
        return await form.execute()


class BrainstormAgentOperative(AgentOperative):
    def __init__(self, **config):
        super().__init__(CreativeAgentForm, **config)

    async def execute(self):
        form = self.form_class(
            num_ideas=self.config.get("num_ideas", 3),
            auto_execute=self.config.get("auto_execute", True),
            context=self.context,
        )
        return await form.execute()


class PlanAgentOperative(AgentOperative):
    def __init__(self, **config):
        super().__init__(PlanningAgentForm, **config)

    async def execute(self):
        form = self.form_class(
            num_steps=self.config.get("num_steps", 3), context=self.context
        )
        return await form.execute()


class AgentOperativeFactory:
    OPERATIVE_TYPES = {
        "select": SelectAgentOperative,
        "brainstorm": BrainstormAgentOperative,
        "plan": PlanAgentOperative,
    }

    # ...rest of factory implementation as shown in design...


class AgentOperativeComposer:
    """Composer for agent operative workflows."""

    # ...rest of composer implementation as shown in design...
