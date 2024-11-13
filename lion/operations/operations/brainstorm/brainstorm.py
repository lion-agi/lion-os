from typing import Any

from lion.core.session.branch import Branch
from lion.core.session.session import Session
from lion.core.types import ID
from lion.libs.func import alcall
from lion.protocols.operatives.instruct import INSTRUCT_MODEL_FIELD, InstructModel

from .prompt import PROMPT


async def run_instruct(
    ins: InstructModel,
    session: Session,
    branch: Branch,
    auto_run: bool,
    verbose: bool = False,
    **kwargs: Any,
) -> Any:
    """Execute an instruction within a brainstorming session.

    Args:
        ins: The instruction model to run.
        session: The current session.
        branch: The branch to operate on.
        auto_run: Whether to automatically run nested instructions.
        verbose: Whether to enable verbose output.
        **kwargs: Additional keyword arguments.

    Returns:
        The result of the instruction execution.
    """
    if verbose:
        guidance_preview = (
            ins.guidance[:100] + "..." if len(ins.guidance) > 100 else ins.guidance
        )
        print(f"Running instruction: {guidance_preview}")

    async def run(ins_):
        b_ = session.split(branch)
        return await run_instruct(ins_, session, b_, False, **kwargs)

    config = {**ins.model_dump(), **kwargs}
    res = await branch.operate(**config)
    branch.msgs.logger.dump()
    instructs = []

    if hasattr(res, "instruct_models"):
        instructs = res.instruct_models

    if auto_run is True and instructs:
        ress = await alcall(instructs, run)
        response_ = []
        for res in ress:
            if isinstance(res, list):
                response_.extend(res)
            else:
                response_.append(res)
        response_.insert(0, res)
        return response_

    return res


async def brainstorm(
    instruct: InstructModel | dict[str, Any],
    num_instruct: int = 3,
    session: Session | None = None,
    branch: Branch | ID.Ref | None = None,
    auto_run: bool = True,
    branch_kwargs: dict[str, Any] | None = None,
    return_session: bool = False,
    verbose: bool = False,
    **kwargs: Any,
) -> Any:
    """Perform a brainstorming session.

    Args:
        instruct: Instruction model or dictionary.
        num_instruct: Number of instructions to generate.
        session: Existing session or None to create a new one.
        branch: Existing branch or reference.
        auto_run: If True, automatically run generated instructions.
        branch_kwargs: Additional arguments for branch creation.
        return_session: If True, return the session with results.
        verbose: Whether to enable verbose output.
        **kwargs: Additional keyword arguments.

    Returns:
        The results of the brainstorming session, optionally with the session.
    """
    if verbose:
        print(f"Starting brainstorming with {num_instruct} instructions.")

    field_models: list = kwargs.get("field_models", [])
    if INSTRUCT_MODEL_FIELD not in field_models:
        field_models.append(INSTRUCT_MODEL_FIELD)

    kwargs["field_models"] = field_models

    if session is not None:
        if branch is not None:
            branch: Branch = session.branches[branch]
        else:
            branch = session.new_branch(**(branch_kwargs or {}))
    else:
        session = Session()
        if isinstance(branch, Branch):
            session.branches.include(branch)
            session.default_branch = branch
        if branch is None:
            branch = session.new_branch(**(branch_kwargs or {}))

    if isinstance(instruct, InstructModel):
        instruct = instruct.clean_dump()
    if not isinstance(instruct, dict):
        raise ValueError(
            "instruct needs to be an InstructModel obj or a dictionary of valid parameters"
        )

    guidance = instruct.get("guidance", "")
    instruct["guidance"] = f"\n{PROMPT.format(num_instruct=num_instruct)}" + guidance

    res1 = await branch.operate(**instruct, **kwargs)
    if verbose:
        print("Initial brainstorming complete.")

    instructs = None

    async def run(ins_):
        b_ = session.split(branch)
        return await run_instruct(
            ins_, session, b_, auto_run, verbose=verbose, **kwargs
        )

    if not auto_run:
        return res1

    async with session.branches:
        if hasattr(res1, "instruct_models"):
            instructs: list[InstructModel] = res1.instruct_models
            ress = await alcall(instructs, run)
            response_ = []

            for res in ress:
                if isinstance(res, list):
                    response_.extend(res)
                else:
                    response_.append(res)

            response_.insert(0, res1)
            if return_session:
                return response_, session
            return response_

    if return_session:
        return res1, session

    return res1
