from pathlib import Path

import pandas as pd
from pydantic import field_serializer, model_validator

from lion.core.generic import Component, LogManager, Pile, Progression
from lion.core.typing import ID
from lion.integrations.litellm_.imodel import iModel
from lion.settings import Settings

from ..action.action_manager import ActionManager
from ..communication import MESSAGE_FIELDS, MessageManager, RoledMessage, System
from .branch_mixins import BranchActionMixin, BranchOperationMixin


class Branch(Component, BranchActionMixin, BranchOperationMixin):
    """
    Represents a conversation branch with its own message history and tools.

    A branch can be serialized and reconstructed, maintaining its essential state
    including messages, actions, and model configurations.
    """

    user: str | None = None
    name: str | None = None
    msgs: MessageManager = None
    acts: ActionManager = None
    imodel: iModel | None = None
    parse_imodel: iModel | None = None

    @field_serializer("msgs")
    def serialize_msgs(self, msgs: MessageManager) -> dict:
        """Serialize message manager to a dictionary."""
        if not msgs:
            return None
        return {
            "messages": [msg.model_dump() for msg in msgs.messages.values()],
            "system": msgs.system.model_dump() if msgs.system else None,
            "logger_config": msgs.logger.model_dump() if msgs.logger else None,
        }

    @field_serializer("acts")
    def serialize_acts(self, acts: ActionManager) -> dict:
        """Serialize action manager to a dictionary."""
        if not acts:
            return None
        return acts.model_dump()

    @field_serializer("imodel")
    def serialize_imodel(self, imodel: iModel) -> dict:
        """Serialize imodel to a dictionary."""
        if not imodel:
            return None
        return imodel.model_dump()

    @field_serializer("parse_imodel")
    def serialize_parse_imodel(self, parse_imodel: iModel) -> dict:
        """Serialize parse imodel to a dictionary."""
        if not parse_imodel:
            return None
        return parse_imodel.model_dump()

    @model_validator(mode="before")
    def _validate_data(cls, data: dict) -> dict:
        """Validate and reconstruct branch data."""
        user = data.pop("user", None)
        name = data.pop("name", None)

        # Handle MessageManager
        message_manager = data.pop("msgs", None)
        if isinstance(message_manager, dict):
            messages = [
                RoledMessage.model_validate(msg)
                for msg in message_manager.get("messages", [])
            ]
            system = (
                System.model_validate(message_manager["system"])
                if message_manager.get("system")
                else None
            )
            logger = (
                LogManager.model_validate(message_manager["logger_config"])
                if message_manager.get("logger_config")
                else None
            )
            message_manager = MessageManager(
                messages=messages, system=system, logger=logger
            )
        elif not message_manager:
            message_manager = MessageManager(
                messages=data.pop("messages", None),
                logger=data.pop("logger", None),
                system=data.pop("system", None),
            )
        if not message_manager.logger:
            message_manager.logger = LogManager(
                **Settings.Branch.BRANCH.message_log_config.clean_dump()
            )

        # Handle ActionManager
        acts = data.pop("acts", None)
        if isinstance(acts, dict):
            acts = ActionManager.model_validate(acts)
        elif not acts:
            acts = ActionManager()
            acts.logger = LogManager(
                **Settings.Branch.BRANCH.action_log_config.clean_dump()
            )
        if "tools" in data:
            acts.register_tools(data.pop("tools"))

        # Handle iModel
        imodel = data.pop("imodel", None)
        if isinstance(imodel, dict):
            imodel = iModel.model_validate(imodel)
        elif not imodel:
            imodel = iModel(**Settings.iModel.CHAT)

        # Handle parse_imodel
        parse_imodel = data.pop("parse_imodel", None)
        if isinstance(parse_imodel, dict):
            parse_imodel = iModel.model_validate(parse_imodel)

        out = {
            "user": user,
            "name": name,
            "msgs": message_manager,
            "acts": acts,
            "imodel": imodel,
            "parse_imodel": parse_imodel,
            **data,
        }
        return out

    def dump_log(self, clear: bool = True, persist_path: str | Path = None):
        """Dump logs to file."""
        self.msgs.logger.dump(clear, persist_path)
        self.acts.logger.dump(clear, persist_path)

    def to_df(self, *, progress: Progression = None) -> pd.DataFrame:
        """Convert branch messages to DataFrame."""
        if progress is None:
            progress = self.msgs.progress

        msgs = [self.msgs.messages[i] for i in progress if i in self.msgs.messages]
        p = Pile(items=msgs)
        return p.to_df(columns=MESSAGE_FIELDS)

    async def aclone(self, sender: ID.Ref = None) -> "Branch":
        """Clone branch asynchronously."""
        async with self.msgs.messages:
            return self.clone(sender)

    def clone(self, sender: ID.Ref = None) -> "Branch":
        """
        Clone this branch, creating a new branch with the same messages and tools.

        Args:
            sender: Optional sender ID for the new branch

        Returns:
            A new Branch instance with copied state
        """
        if sender is not None:
            if not ID.is_id(sender):
                raise ValueError(
                    "Input value for branch.clone sender is not a valid sender"
                )
            sender = ID.get_id(sender)

        system = self.msgs.system.clone() if self.msgs.system else None
        tools = list(self.acts.registry.values()) if self.acts.registry else None
        branch_clone = Branch(
            system=system,
            user=self.user,
            messages=[i.clone() for i in self.msgs.messages],
            tools=tools,
        )
        for message in branch_clone.msgs.messages:
            message.sender = sender or self.ln_id
            message.recipient = branch_clone.ln_id
        return branch_clone

    def to_dict(self) -> dict:
        """Convert branch to a dictionary of metadata."""
        return {
            "branch_id": self.ln_id,
            "user": self.user,
            "name": self.name,
            "message_count": len(self.msgs.messages) if self.msgs else 0,
            "has_system": bool(self.msgs and self.msgs.system),
            "tool_count": len(self.acts.registry) if self.acts else 0,
            "model": self.imodel.model if self.imodel else None,
        }
