"""Field definitions for InstructModel components."""

from pydantic import JsonValue, field_validator

from lion.core.models import FieldModel, SchemaModel
from lion.libs.parse import validate_boolean

from .prompts import (
    actions_field_description,
    context_examples,
    context_field_description,
    guidance_examples,
    guidance_field_description,
    instruction_examples,
    instruction_field_description,
    reason_field_description,
)


def validate_instruction(cls, value) -> JsonValue | None:
    """Validates that instruction is not empty/None and is in correct format"""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    return value


def validate_boolean_field(cls, value) -> bool | None:
    """Validates boolean fields, allowing for flexible input formats"""
    try:
        return validate_boolean(value)
    except Exception:
        return None


# Field Models
INSTRUCTION_FIELD = FieldModel(
    name="instruction",
    annotation=JsonValue | None,
    default=None,
    title="Primary Instruction",
    description=instruction_field_description,
    examples=instruction_examples,
    validator=validate_instruction,
    validator_kwargs={"mode": "before"},
)

GUIDANCE_FIELD = FieldModel(
    name="guidance",
    annotation=JsonValue | None,
    default=None,
    title="Execution Guidance",
    description=guidance_field_description,
    examples=guidance_examples,
)

CONTEXT_FIELD = FieldModel(
    name="context",
    annotation=JsonValue | None,
    default=None,
    title="Task Context",
    description=context_field_description,
    examples=context_examples,
)

REASON_FIELD = FieldModel(
    name="reason",
    annotation=bool,
    default=False,
    title="Include Reasoning",
    description=reason_field_description,
    validator=validate_boolean_field,
    validator_kwargs={"mode": "before"},
)

ACTIONS_FIELD = FieldModel(
    name="actions",
    annotation=bool,
    default=False,
    title="Require Actions",
    description=actions_field_description,
    validator=validate_boolean_field,
    validator_kwargs={"mode": "before"},
)


class InstructModel(SchemaModel):
    """Model for defining instruction parameters and execution requirements."""

    instruction: JsonValue | None = INSTRUCTION_FIELD.field_info
    guidance: JsonValue | None = GUIDANCE_FIELD.field_info
    context: JsonValue | None = CONTEXT_FIELD.field_info
    reason: bool = REASON_FIELD.field_info
    actions: bool = ACTIONS_FIELD.field_info

    @field_validator("instruction", **INSTRUCTION_FIELD.validator_kwargs)
    def _validate_instruction(cls, v):
        return INSTRUCTION_FIELD.validator(cls, v)

    @field_validator("reason", **REASON_FIELD.validator_kwargs)
    def _validate_reason(cls, v):
        return REASON_FIELD.validator(cls, v)

    @field_validator("actions", **ACTIONS_FIELD.validator_kwargs)
    def _validate_actions(cls, v):
        return ACTIONS_FIELD.validator(cls, v)


INSTRUCT_MODEL_FIELD = FieldModel(
    name="instruct_models",
    annotation=list[InstructModel],
    default_factory=list,
    title="Instruction Model",
    description="Model for defining instruction parameters and execution requirements.",
)

# Export all components
__all__ = [
    "INSTRUCTION_FIELD",
    "GUIDANCE_FIELD",
    "CONTEXT_FIELD",
    "REASON_FIELD",
    "ACTIONS_FIELD",
    "InstructModel",
    "INSTRUCT_MODEL_FIELD",
]
