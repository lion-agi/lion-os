from __future__ import annotations

import inspect

from pydantic import field_serializer, field_validator

from lion.core.generic import Component, Log
from lion.core.typing import Any, Communicatable, Enum, Field, Note, override
from lion.libs.utils import copy

from .._class_registry import get_class
from .base_mail import BaseMail


class MessageRole(str, Enum):
    """Enum for possible roles a message can assume in a conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class MessageFlag(str, Enum):
    """Enum to signal constructing a clone Message"""

    MESSAGE_CLONE = "MESSAGE_CLONE"
    MESSAGE_LOAD = "MESSAGE_LOAD"


class MessageField(str, Enum):
    """Enum for possible fields a message can have."""

    TIMESTAMP = "timestamp"
    LION_CLASS = "lion_class"
    ROLE = "role"
    CONTENT = "content"
    ln_id = "ln_id"
    SENDER = "sender"
    RECIPIENT = "recipient"
    METADATA = "metadata"


MESSAGE_FIELDS = [i.value for i in MessageField.__members__.values()]


class RoledMessage(Component, BaseMail):
    """
    A base class representing a message with roles and properties.

    This class combines functionality from Relational, Component, and BaseMail
    to create a versatile message object with role-based behavior.
    """

    content: Note = Field(
        default_factory=Note,
        description="The content of the message.",
    )

    role: MessageRole | None = Field(
        default=None,
        description="The role of the message in the conversation.",
        examples=["system", "user", "assistant"],
    )

    @property
    def image_content(self) -> list[dict[str, Any]] | None:
        """
        Return image content if present in the message.

        Returns:
            Optional[List[Dict[str, Any]]]: A list of image content
                dictionaries, or None if no images are present.
        """
        msg_ = self.chat_msg
        if isinstance(msg_, dict) and isinstance(msg_["content"], list):
            return [i for i in msg_["content"] if i["type"] == "image_url"]
        return None

    @property
    def chat_msg(self) -> dict[str, Any] | None:
        """
        Return message in chat representation.

        Returns:
            Optional[Dict[str, Any]]: The message in chat format, or None
                if formatting fails.
        """
        try:
            return self._format_content()
        except Exception:
            return None

    @field_validator("role")
    def _validate_role(cls, v: Any) -> MessageRole | None:
        """
        Validates the role of the message.

        Args:
            v (Any): The role value to validate.

        Returns:
            MessageRole | None: The validated role, or None if invalid.
        """
        if v in [r.value for r in MessageRole]:
            return MessageRole(v)
        raise ValueError(f"Invalid message role: {v}")

    def clone(self) -> RoledMessage:
        """
        Creates a copy of the current RoledMessage object.

        Returns:
            RoledMessage: A new instance with copied attributes.
        """
        cls = self.__class__
        signature = inspect.signature(cls.__init__)
        param_num = len(signature.parameters) - 2

        init_args = [MessageFlag.MESSAGE_CLONE] * param_num

        obj = cls(*init_args)
        obj.role = self.role
        obj.content = self.content
        obj.metadata.set("clone_from", self)

        return obj

    @override
    @classmethod
    def from_dict(cls, data: dict, /, **kwargs: Any) -> RoledMessage:
        """
        Loads a RoledMessage object from a dictionary.

        Args:
            data: The dictionary containing the message data.
            **kwargs: Additional keyword arguments.

        Returns:
            RoledMessage: An instance of RoledMessage created from
                the dictionary.
        """
        data = copy(data)
        if kwargs:
            data.update(kwargs)
        if "lion_class" in data:
            cls = get_class(data.pop("lion_class"))
        signature = inspect.signature(cls.__init__)
        param_num = len(signature.parameters) - 2

        init_args = [MessageFlag.MESSAGE_LOAD] * param_num

        extra_fields = {}
        for k, v in list(data.items()):
            if k not in cls.model_fields:
                extra_fields[k] = data.pop(k)

        obj = cls(*init_args, protected_init_params=data)

        for k, v in extra_fields.items():
            obj.add_field(name=k, value=v)

        metadata = data.get("metadata", {})
        last_updated = metadata.get("last_updated", None)
        if last_updated is not None:
            obj.metadata.set(["last_updated"], last_updated)
        return obj

    @override
    def __str__(self) -> str:
        """
        Provides a string representation of the message.

        Returns:
            str: A string representation of the message.
        """
        content_preview = (
            f"{str(self.content)[:75]}..."
            if len(str(self.content)) > 75
            else str(self.content)
        )
        return (
            f"Message(role={self.role}, sender={self.sender}, "
            f"content='{content_preview}')"
        )

    def to_log(self) -> Log:
        """
        Convert the message to a Log object.

        Returns:
            Log: A Log object representing the message.
        """
        dict_ = self.to_dict()
        content = dict_.pop("content")
        _log = Log(
            content=content,
            loginfo=dict_,
        )
        return _log

    @field_serializer("content")
    def _serialize_content(self, value: Note) -> dict[str, Any]:
        """
        Serialize the content.

        Args:
            value (Note): The content to serialize.

        Returns:
            dict[str, Any]: The serialized content.
        """
        output_dict = copy(value.content, deep=True)
        origin_obj = output_dict.pop("clone_from", None)

        if origin_obj and isinstance(origin_obj, Communicatable):
            info_dict = {
                "clone_from_info": {
                    "original_ln_id": origin_obj.ln_id,
                    "original_timestamp": origin_obj.timestamp,
                    "original_sender": origin_obj.sender,
                    "original_recipient": origin_obj.recipient,
                }
            }
            output_dict.update(info_dict)
        return output_dict

    @field_serializer("role")
    def _serialize_role(self, value: MessageRole):
        """
        Serialize the role.

        Args:
            value (MessageRole): The role to serialize.

        Returns:
            str: The serialized role.
        """
        return value.value

    def _format_content(self) -> dict[str, Any]:
        """
        Format the message content for chat representation.

        Returns:
            dict[str, Any]: The formatted content.
        """
        if self.content.get("images", None):
            content = self.content.to_dict()
        else:
            content = str(self.content.to_dict())
        return {"role": self.role.value, "content": content}
