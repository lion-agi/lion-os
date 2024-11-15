from lion.core.session.branch import Branch
from lion.core.session.session import Session
from lion.core.typing import ID, Any, BaseModel
from lion.libs.func import alcall
from lion.libs.parse import to_flat_list
from lion.protocols.operatives.instruct import INSTRUCT_MODEL_FIELD, Instruct

from .prompt import PROMPT


class BrainStormInstruct(Instruct):
    response: Any | None = None


class BrainstormOperation(BaseModel):
    initial: Any
    brainstorm: list[Instruct] | None = None
    explore: list[BrainStormInstruct] | None = None


async def run_instruct(
    ins: Instruct,
    session: Session,
    branch: Branch,
    auto_run: bool,
    verbose: bool = True,
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

    async def run(ins_):
        if verbose:
            msg_ = (
                ins_.guidance[:100] + "..."
                if len(ins_.guidance) > 100
                else ins_.guidance
            )
            print(f"\n-----Running instruction-----\n{msg_}")
        b_ = session.split(branch)
        return await run_instruct(ins_, session, b_, False, verbose=verbose, **kwargs)

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
    instruct: Instruct | dict[str, Any],
    num_instruct: int = 3,
    session: Session | None = None,
    branch: Branch | ID.Ref | None = None,
    auto_run: bool = True,
    auto_explore: bool = False,
    explore_kwargs: dict[str, Any] | None = None,
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
    if auto_explore and not auto_run:
        raise ValueError("auto_explore requires auto_run to be True.")

    if verbose:
        print(f"Starting brainstorming...")

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

    if isinstance(instruct, Instruct):
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
        if verbose:
            msg_ = (
                ins_.guidance[:100] + "..."
                if len(ins_.guidance) > 100
                else ins_.guidance
            )
            print(f"\n-----Running instruction-----\n{msg_}")
        b_ = session.split(branch)
        return await run_instruct(
            ins_, session, b_, auto_run, verbose=verbose, **kwargs
        )

    out = BrainstormOperation(initial=res1)

    if not auto_run:
        if return_session:
            return out, session
        return out

    async with session.branches:
        response_ = []
        if hasattr(res1, "instruct_models"):
            instructs: list[Instruct] = res1.instruct_models
            ress = await alcall(instructs, run)
            ress = to_flat_list(ress, dropna=True)

            response_ = [
                res if not isinstance(res, str | dict) else None for res in ress
            ]
            response_ = to_flat_list(response_, unique=True, dropna=True)
            out.brainstorm = response_ if isinstance(response_, list) else [response_]
            response_.insert(0, res1)

        if response_ and auto_explore:

            async def explore(ins_: Instruct):
                if verbose:
                    msg_ = (
                        ins_.guidance[:100] + "..."
                        if len(ins_.guidance) > 100
                        else ins_.guidance
                    )
                    print(f"\n-----Exploring Idea-----\n{msg_}")
                b_ = session.split(branch)
                response = await b_.communicate(
                    instruction=ins_.instruction,
                    guidance=ins_.guidance,
                    context=ins_.context,
                    **(explore_kwargs or {}),
                )
                return BrainStormInstruct(
                    instruction=ins_.instruction,
                    guidance=ins_.guidance,
                    context=ins_.context,
                    response=response,
                )

            response_ = to_flat_list(
                [i.instruct_models for i in response_ if hasattr(i, "instruct_models")],
                dropna=True,
                unique=True,
            )
            res_explore = await alcall(response_, explore)
            out.explore = res_explore

    if return_session:
        return out, session

    return out
