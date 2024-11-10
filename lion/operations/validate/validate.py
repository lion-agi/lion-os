from typing import Any, Optional
from pydantic import BaseModel, Field

from lion.core.session.branch import Branch
from lion.core.session.session import Session
from lion.core.types import ID
from lion.protocols.operatives.instruct import INSTRUCT_MODEL_FIELD, InstructModel

VALIDATE_PROMPT = """Conduct a {num_steps}-step validation process. The validation should:
1. Start with core assumptions
2. Progress through logical dependencies
3. Consider system interactions
4. Build cumulative confidence

Each step should:
- Build on previous validation
- Address key uncertainties
- Document evidence
- Update confidence level"""

class ValidationStep(BaseModel):
    """Single step in validation chain"""
    step: str = Field(..., description="Current validation focus")
    depends_on: list[str] = Field(default_factory=list, description="Previous validation steps this relies on")
    method: str = Field(..., description="Validation approach")
    findings: list[dict] = Field(..., description="What was found")
    confidence_delta: float = Field(..., ge=-1, le=1, description="How this step affects overall confidence")
    next_concerns: list[str] = Field(..., description="What should be validated next")

async def validate(
    instruct: InstructModel | dict,
    num_steps: int = 3,
    session: Session | None = None,
    branch: Branch | ID.Ref | None = None,
    branch_kwargs: dict = {},
    operative_model: Optional[type[BaseModel]] = ValidationStep,
    **kwargs,
) -> Any:
    """Execute validation process - steps build on previous validations"""
    field_models: list = kwargs.get("field_models", [])
    if INSTRUCT_MODEL_FIELD not in field_models:
        field_models.append(INSTRUCT_MODEL_FIELD)
    kwargs["field_models"] = field_models

    if session is not None:
        if branch is not None:
            branch: Branch = session.branches[branch]
        else:
            branch = session.new_branch(**branch_kwargs)
    else:
        session = Session()
        if isinstance(branch, Branch):
            session.branches.include(branch)
            session.default_branch = branch
        if branch is None:
            branch = session.new_branch(**branch_kwargs)

    if isinstance(instruct, InstructModel):
        instruct = instruct.clean_dump()
    if not isinstance(instruct, dict):
        raise ValueError(
            "instruct needs to be an InstructModel obj or a dictionary of valid parameters"
        )

    guidance = instruct.get("guidance", "")
    instruct["guidance"] = f"\n{VALIDATE_PROMPT.format(num_steps=num_steps)}" + guidance

    return await branch.operate(**instruct, operative_model=operative_model, **kwargs)

__all__ = [
    "validate",
    "ValidationStep"
]
