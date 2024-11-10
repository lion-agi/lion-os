from typing import Any, Optional
from pydantic import BaseModel, Field

from lion.core.session.branch import Branch
from lion.core.session.session import Session
from lion.core.types import ID
from lion.protocols.operatives.instruct import INSTRUCT_MODEL_FIELD, InstructModel

class SynthesisElement(BaseModel):
    """Single element of a larger synthesis"""
    element: str = Field(..., description="The synthesized element")
    components: list[dict] = Field(..., description="Combined components")
    novel_aspects: list[str] = Field(..., description="New/unique features")
    rationale: str = Field(..., description="Why this combination works")
    implications: list[str] = Field(..., description="Potential impacts and applications")

SYNTHESIZE_PROMPT = """Create a synthesis with up to {num_elements} key elements. The synthesis should:
1. Identify core combinable elements
2. Find novel connection points
3. Address key gaps or needs
4. Form a coherent whole

Focus on:
- Creating meaningful connections
- Building practical value
- Maintaining internal consistency
- Enabling further development"""

async def synthesize(
    instruct: InstructModel | dict,
    num_elements: int = 3,
    session: Session | None = None,
    branch: Branch | ID.Ref | None = None,
    branch_kwargs: dict = {},
    operative_model: Optional[type[BaseModel]] = SynthesisElement,
    auto_explore: bool = False,  # Only branch if explicitly requested
    **kwargs,
) -> Any:
    """Execute synthesis process - with optional exploration of alternatives"""
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
    instruct["guidance"] = f"\n{SYNTHESIZE_PROMPT.format(num_elements=num_elements)}" + guidance

    result = await branch.operate(**instruct, operative_model=operative_model, **kwargs)

    # Only explore alternatives if explicitly requested
    if auto_explore and hasattr(result, "instruct_models") and result.instruct_models:
        alternatives = []
        for ins in result.instruct_models:
            b_ = session.split(branch)
            alt = await synthesize(ins, num_elements, session, b_, branch_kwargs, operative_model, False, **kwargs)
            alternatives.append(alt)
        result.alternatives = alternatives

    return result
