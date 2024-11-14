from typing import Any

from lion.core.session.branch import Branch
from lion.core.session.session import Session
from lion.operations.utils import RecursiveOperation
from lion.protocols.operatives.instruct import InstructModel

from .prompt import PROMPT


class BrainstormOperation(RecursiveOperation):
    """Operation for generating and executing multiple creative solutions."""

    def __init__(self):
        super().__init__(PROMPT)

    def format_prompt(self, num_instruct: int, **kwargs: Any) -> str:
        """Format the prompt with the correct parameter name.

        Args:
            num_instruct: Number of instructions to generate.
            **kwargs: Additional format parameters.

        Returns:
            The formatted prompt string.
        """
        # Remove num_instruct from kwargs if it exists to avoid duplicate
        kwargs.pop("num_instruct", None)
        return self.prompt.format(num_instruct=num_instruct, **kwargs)

    async def __call__(
        self,
        instruct: InstructModel | dict[str, Any],
        num_instruct: int = 3,
        session: Session | None = None,
        branch: Branch | None = None,
        auto_run: bool = True,
        branch_kwargs: dict[str, Any] | None = None,
        return_session: bool = False,
        verbose: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Override to ensure we use the provided session."""
        if "num_instruct" in kwargs:
            num_instruct = kwargs.pop("num_instruct")

        result = await super().__call__(
            instruct,
            num_instruct=num_instruct,
            session=session,
            branch=branch,
            auto_run=auto_run,
            branch_kwargs=branch_kwargs,
            return_session=return_session,
            verbose=verbose,
            **kwargs,
        )
        return result


# Create a singleton instance for the default brainstorm operation
brainstorm = BrainstormOperation()
