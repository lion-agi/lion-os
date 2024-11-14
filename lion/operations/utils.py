from typing import Any, Optional, Tuple, Union

from lion.core.session.branch import Branch
from lion.core.session.session import Session
from lion.core.typing import ID
from lion.protocols.operatives.instruct import INSTRUCT_MODEL_FIELD, InstructModel


class BaseOperation:
    """Base class for operations that process instructions with prompts."""

    def __init__(self, prompt: str):
        """Initialize the operation with a specific prompt.

        Args:
            prompt: The prompt template to use for this operation.
        """
        self.prompt = prompt

    async def __call__(
        self,
        instruct: InstructModel | dict[str, Any],
        session: Session | None = None,
        branch: Branch | None = None,
        branch_kwargs: dict[str, Any] | None = None,
        return_session: bool = False,
        verbose: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Process the instruction by adding the prompt and executing it in a branch.

        Args:
            instruct: Instruction model or dictionary.
            session: Existing session or None to create a new one.
            branch: Existing branch or None to create a new one.
            branch_kwargs: Additional arguments for branch creation.
            return_session: If True, return the session along with results.
            verbose: Whether to enable verbose output.
            **kwargs: Additional keyword arguments.

        Returns:
            The result of the operation, optionally with the session.
        """
        if verbose:
            print(f"Processing {self.__class__.__name__.lower()}...")

        session = session or Session()
        branch = branch or session.new_branch(**(branch_kwargs or {}))

        if isinstance(instruct, InstructModel):
            instruct = instruct.clean_dump()
        elif not isinstance(instruct, dict):
            raise ValueError(
                "instruct needs to be an InstructModel object or a dictionary of valid parameters"
            )

        instruct["instruction"] = f"{self.prompt}\n{instruct.get('instruction', '')}"
        result = await branch.operate(**instruct, **kwargs)

        if verbose:
            print(f"{self.__class__.__name__} operation complete.")

        if return_session:
            return result, session
        return result


class RecursiveOperation(BaseOperation):
    """Base class for operations that support recursive instruction execution."""

    async def run_instruction(
        self,
        ins: InstructModel,
        session: Session,
        branch: Branch,
        auto_run: bool,
        verbose: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Execute a single instruction within the operation.

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

        config = {**ins.model_dump(), **kwargs}
        res = await branch.operate(**config)
        branch.msgs.logger.dump()

        if auto_run and hasattr(res, "instruct_models") and res.instruct_models:
            nested_results = []
            for nested_ins in res.instruct_models:
                nested_branch = session.split(branch)
                nested_result = await self.run_instruction(
                    nested_ins, session, nested_branch, False, verbose=verbose, **kwargs
                )
                nested_results.append(nested_result)
            return [res] + nested_results

        return res

    def format_prompt(self, num_instruct: int, **kwargs: Any) -> str:
        """Format the prompt with the given parameters.

        Args:
            num_instruct: Number of instructions to generate.
            **kwargs: Additional format parameters.

        Returns:
            The formatted prompt string.
        """
        return self.prompt.format(num_instruct=num_instruct, **kwargs)

    async def __call__(
        self,
        instruct: InstructModel | dict[str, Any],
        num_instruct: int = 3,
        session: Session | None = None,
        branch: Branch | ID.Ref | None = None,
        auto_run: bool = True,
        branch_kwargs: dict[str, Any] | None = None,
        return_session: bool = False,
        verbose: bool = False,
        **kwargs: Any,
    ) -> Any | tuple[Any, Session]:
        """Execute the operation with support for recursive instruction processing.

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
            The results of the operation, optionally with the session.
        """
        if verbose:
            print(
                f"Starting {self.__class__.__name__.lower()} with {num_instruct} instructions."
            )

        field_models: list = kwargs.get("field_models", [])
        if INSTRUCT_MODEL_FIELD not in field_models:
            field_models.append(INSTRUCT_MODEL_FIELD)
        kwargs["field_models"] = field_models

        session = session or Session()
        if branch is not None:
            if isinstance(branch, ID.Ref):
                branch = session.branches[branch]
        else:
            branch = session.new_branch(**(branch_kwargs or {}))

        if isinstance(instruct, InstructModel):
            instruct = instruct.clean_dump()
        if not isinstance(instruct, dict):
            raise ValueError(
                "instruct needs to be an InstructModel object or a dictionary of valid parameters"
            )

        guidance = instruct.get("guidance", "")
        formatted_prompt = self.format_prompt(num_instruct, **kwargs)
        instruct["guidance"] = f"\n{formatted_prompt}" + guidance

        res1 = await branch.operate(**instruct, **kwargs)
        if verbose:
            print("Initial operation complete.")

        if not auto_run:
            if return_session:
                return res1, session
            return res1

        results = [res1]
        if hasattr(res1, "instruct_models"):
            for ins in res1.instruct_models:
                nested_branch = session.split(branch)
                nested_result = await self.run_instruction(
                    ins, session, nested_branch, auto_run, verbose=verbose, **kwargs
                )
                if isinstance(nested_result, list):
                    results.extend(nested_result)
                else:
                    results.append(nested_result)

        if return_session:
            return results, session
        return results


async def process_instruction(
    instruct: InstructModel | dict[str, Any],
    prompt: str,
    session: Session | None = None,
    branch: Branch | None = None,
    branch_kwargs: dict[str, Any] | None = None,
    return_session: bool = False,
    verbose: bool = False,
    **kwargs: Any,
) -> Any:
    """Legacy function for backward compatibility.

    New code should use BaseOperation instead.
    """
    operation = BaseOperation(prompt)
    return await operation(
        instruct,
        session=session,
        branch=branch,
        branch_kwargs=branch_kwargs,
        return_session=return_session,
        verbose=verbose,
        **kwargs,
    )
