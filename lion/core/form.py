from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class Form(BaseModel, ABC, Generic[T]):
    """Base form class with validation and work tracking."""

    guidance: str | None = Field(default=None)
    metadata: dict[str, Any] = Field(default_factory=dict)
    work_fields: list[str] = Field(default_factory=list)

    @abstractmethod
    async def execute(self) -> T:
        """Execute form operation."""
        pass

    def is_workable(self) -> bool:
        """Check if form has required fields."""
        return all(getattr(self, field, None) is not None for field in self.work_fields)


class Report(BaseModel):
    """Base report class for tracking form executions."""

    completed_forms: list[Form] = Field(default_factory=list)

    def save_completed_form(self, form: Form) -> None:
        """Save completed form execution."""
        self.completed_forms.append(form)
