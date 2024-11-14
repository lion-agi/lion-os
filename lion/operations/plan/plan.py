from typing import Any

from lion.core.session.branch import Branch
from lion.core.session.session import Session
from lion.core.typing import ID
from lion.protocols.operatives.instruct import INSTRUCT_MODEL_FIELD, Instruct

from .prompt import PROMPT


async def run_step(
    ins: Instruct,
    session: Session,
    branch: Branch,
    verbose: bool = False,
    **kwargs: Any,
) -> Any:
    """Execute a single step of the plan.

    Args:
        ins: The instruction model for the step.
        session: The current session.
        branch: The branch to operate on.
        verbose: Whether to enable verbose output.
        **kwargs: Additional keyword arguments.

    Returns:
        The result of the branch operation.
    """
    if verbose:
        guidance_preview = (
            ins.guidance[:100] + "..." if len(ins.guidance) > 100 else ins.guidance
        )
        print(f"Executing step: {guidance_preview}")
    config = {**ins.model_dump(), **kwargs}
    res = await branch.operate(**config)
    branch.msgs.logger.dump()
    return res


async def plan(
    instruct: Instruct | dict[str, Any],
    num_steps: int = 3,
    session: Session | None = None,
    branch: Branch | ID.Ref | None = None,
    auto_run: bool = True,
    branch_kwargs: dict[str, Any] | None = None,
    return_session: bool = False,
    verbose: bool = False,
    **kwargs: Any,
) -> Any:
    """Create and execute a multi-step plan.

    Args:
        instruct: Instruction model or dictionary.
        num_steps: Number of steps in the plan.
        session: Existing session or None to create a new one.
        branch: Existing branch or reference.
        auto_run: If True, automatically run the steps.
        branch_kwargs: Additional keyword arguments for branch creation.
        return_session: If True, return the session along with results.
        verbose: Whether to enable verbose output.
        **kwargs: Additional keyword arguments.

    Returns:
        Results of the plan execution, optionally with the session.
    """
    if verbose:
        print(f"Planning execution with {num_steps} steps...")

    field_models: list = kwargs.get("field_models", [])
    if INSTRUCT_MODEL_FIELD not in field_models:
        field_models.append(INSTRUCT_MODEL_FIELD)
    kwargs["field_models"] = field_models

    session = session or Session()
    branch = branch or session.new_branch(**(branch_kwargs or {}))

    if isinstance(instruct, Instruct):
        instruct = instruct.clean_dump()
    if not isinstance(instruct, dict):
        raise ValueError(
            "instruct needs to be an InstructModel object or a dictionary of valid parameters"
        )

    guidance = instruct.get("guidance", "")
    instruct["guidance"] = f"\n{PROMPT.format(num_steps=num_steps)}\n{guidance}"

    res1 = await branch.operate(**instruct, **kwargs)
    if verbose:
        print("Initial planning complete. Starting step execution.")

    if not auto_run:
        if return_session:
            return res1, session
        return res1

    results = res1 if isinstance(res1, list) else [res1]
    if hasattr(res1, "instruct_models"):
        instructs: list[Instruct] = res1.instruct_models
        for i, ins in enumerate(instructs, 1):
            if verbose:
                print(f"\nExecuting step {i}/{len(instructs)}")
            res = await run_step(ins, session, branch, verbose=verbose, **kwargs)
            results.append(res)

        if verbose:
            print("\nAll steps completed successfully!")
    if return_session:
        return results, session
    return results
