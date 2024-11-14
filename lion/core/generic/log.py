"""Base module for logging in the Lion framework."""

from typing import Any, Self

from pydantic import field_serializer

from lion.core.typing import Field, Note, PrivateAttr
from lion.libs.parse import to_dict

from .element import Element


class Log(Element):
    """A log entry in the Lion framework."""

    content: Note = Field(
        default_factory=Note,
        title="Log Content",
        description="The content of the log entry.",
    )

    loginfo: Note = Field(
        default_factory=Note,
        title="Log Info",
        description="Metadata about the log entry.",
    )

    _immutable: bool = PrivateAttr(False)

    def __init__(self, content: Note, loginfo: Note, **kwargs) -> None:
        super().__init__(**kwargs)
        self.content = self._validate_note(content)
        self.loginfo = self._validate_note(loginfo)

    @classmethod
    def _validate_load_data(cls, data: dict, /) -> dict:
        try:
            # Validate required fields
            if not isinstance(data.get("content"), (dict, Note)):
                raise ValueError("Missing or invalid 'content' field")
            if not isinstance(data.get("loginfo"), (dict, Note)):
                raise ValueError("Missing or invalid 'loginfo' field")

            # Convert log_* fields to standard fields
            if "log_id" in data:
                data["ln_id"] = data.pop("log_id")
            if "log_timestamp" in data:
                data["timestamp"] = data.pop("log_timestamp")
            if "log_class" in data:
                data["lion_class"] = data.pop("log_class")
            return data
        except Exception as e:
            raise ValueError(
                "Log can only be loaded from a previously saved log entries.",
            ) from e

    @classmethod
    def from_dict(cls, data: dict, /) -> Self:
        """Create a Log instance from a dictionary."""
        data = cls._validate_load_data(data)
        # Convert content and loginfo to Note objects if they're dicts
        if isinstance(data.get("content"), dict):
            data["content"] = Note(**data["content"])
        if isinstance(data.get("loginfo"), dict):
            data["loginfo"] = Note(**data["loginfo"])
        # Remove lion_class as it's handled by Element
        data.pop("lion_class", None)
        self = cls(**data)
        self._immutable = True
        return self

    def __setattr__(self, name: str, value: Any, /) -> None:
        """Control attribute assignment."""
        if hasattr(self, "_immutable") and self._immutable:
            raise AttributeError("Cannot modify immutable log entry.")
        super().__setattr__(name, value)

    def _validate_note(self, value: Any, /) -> Note:
        """Validate and convert value to Note."""
        if not value:
            return Note()
        if isinstance(value, Note):
            return value
        if isinstance(value, dict):
            return Note(**value)
        try:
            return Note(**to_dict(value))
        except Exception as e:
            raise ValueError(f"Invalid note value: {value}") from e

    @field_serializer("content", "loginfo")
    def _serialize_note(self, value: Note) -> dict:
        """Serialize Note objects."""
        return value.to_dict()

    def to_dict(self) -> dict:
        """Convert Log to dictionary."""
        dict_ = super().to_dict()
        # Convert standard fields to log_* fields
        dict_["log_id"] = dict_.pop("ln_id")
        dict_["log_class"] = dict_.pop("lion_class")
        dict_["log_timestamp"] = dict_.pop("timestamp")

        dict_ = to_dict(
            dict_,
            recursive=True,
            recursive_python_only=False,
            max_recursive_depth=10,  # Increased depth for nested structures
        )
        return dict_

    def to_note(self) -> Note:
        """Convert Log to Note."""
        return Note(**self.to_dict())
