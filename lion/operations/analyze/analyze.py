from typing import Any, Optional
from pydantic import BaseModel, Field

from lion.core.session.branch import Branch
from lion.core.session.session import Session
from lion.core.types import ID
from lion.protocols.operatives.instruct import INSTRUCT_MODEL_FIELD, InstructModel

# Core prompts focused on purpose and process
ANALYZE_PROMPT = """Perform a systematic analysis through {num_perspectives} complementary perspectives. Build a cohesive understanding by:
1. Starting with fundamental patterns
2. Progressively adding analytical layers
3. Connecting insights across perspectives
4. Building toward comprehensive understanding

Each perspective should:
- Add unique analytical value
- Build on previous insights
- Connect to other perspectives
- Contribute to the whole picture"""

class AnalysisPerspective(BaseModel):
    """Single analytical perspective that builds on others"""
    perspective: str = Field(..., description="The analytical lens being used")
    builds_on: list[str] = Field(default_factory=list, description="Previous perspectives this builds upon")
    key_patterns: list[dict] = Field(..., description="Core patterns identified")
    insights: list[str] = Field(..., description="New insights generated")
    connections: list[dict] = Field(..., description="Connections to other perspectives")
    contribution: str = Field(..., description="How this adds to overall understanding")

async def analyze(
    instruct: InstructModel | dict,
    num_perspectives: int = 3,
    session: Session | None = None,
    branch: Branch | ID.Ref | None = None,
    branch_kwargs: dict = {},
    operative_model: Optional[type[BaseModel]] = AnalysisPerspective,
    **kwargs,
) -> Any:
    """Execute multi-perspective analysis process - perspectives build on each other"""
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
    instruct["guidance"] = f"\n{ANALYZE_PROMPT.format(num_perspectives=num_perspectives)}" + guidance

    return await branch.operate(**instruct, operative_model=operative_model, **kwargs)

__all__ = [
    "analyze",
    "AnalysisPerspective"
]
