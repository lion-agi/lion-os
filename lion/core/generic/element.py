"""Base element module for the Lion framework."""

from datetime import datetime

from pydantic import field_validator
from typing_extensions import override

from lion.core._class_registry import LION_CLASS_REGISTRY, get_class
from lion.core.typing import (
    ID,
    Any,
    BaseModel,
    ConfigDict,
    Field,
    IDError,
    LnID,
    Observable,
    TypeVar,
)
from lion.libs.utils import time
from lion.settings import Settings

T = TypeVar("T", bound=Observable)


class Element(BaseModel, Observable):
    """Base class in the Lion framework."""

    ln_id: LnID = Field(
        default_factory=ID.id,
        title="Lion ID",
        description="Unique identifier for the element",
        frozen=True,
    )

    timestamp: float = Field(
        default_factory=lambda: time(type_="timestamp"),
        title="Creation Timestamp",
        frozen=True,
        alias="created",
    )

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
        use_enum_values=True,
        populate_by_name=True,
    )

    @classmethod
    def class_name(cls) -> str:
        """Get the name of the class."""
        return cls.__name__

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        """Initialize and register subclasses in the global class registry."""
        super().__pydantic_init_subclass__(**kwargs)
        LION_CLASS_REGISTRY[cls.__name__] = cls

    @property
    def created_datetime(self) -> datetime:
        """Get the creation datetime of the Element."""
        return datetime.fromtimestamp(self.timestamp, tz=Settings.Config.TIMEZONE)

    @field_validator("ln_id", mode="before")
    def _validate_id(cls, value: ID.ID) -> str:
        try:
            return ID.get_id(value)
        except Exception:
            raise IDError(f"Invalid lion id: {value}")

    @field_validator("timestamp", mode="before")
    def _validate_timestamp(cls, value: Any) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, datetime):
            return value.timestamp()
        try:
            if isinstance(value, str):
                try:
                    return float(value)
                except Exception:
                    return datetime.fromisoformat(value).timestamp()
            raise ValueError
        except Exception as e:
            raise ValueError(f"Invalid datetime string format: {value}") from e

    @override
    @classmethod
    def from_dict(cls, data: dict, /, **kwargs: Any) -> T:
        """create an instance of the Element or its subclass"""
        if "lion_class" in data:
            cls = get_class(class_name=data.pop("lion_class"))
        if cls.from_dict.__func__ != Element.from_dict.__func__:
            return cls.from_dict(data, **kwargs)
        return cls.model_validate(data, **kwargs)

    @override
    def to_dict(self, **kwargs: Any) -> dict:
        """Convert the Element to a dictionary representation."""
        dict_ = self.model_dump(**kwargs)
        dict_["lion_class"] = self.class_name()
        return dict_

    @override
    def __str__(self) -> str:
        """Return a string representation of the Element."""
        timestamp_str = self.created_datetime.isoformat(timespec="minutes")
        return (
            f"{self.class_name()}(ln_id={self.ln_id[:6]}.., "
            f"timestamp={timestamp_str})"
        )

    def __hash__(self) -> int:
        return hash(self.ln_id)

    def __bool__(self) -> bool:
        """Always True"""
        return True

    def __len__(self) -> int:
        """Return the length of the Element."""
        return 1


__all__ = ["Element"]
