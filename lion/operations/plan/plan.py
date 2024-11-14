from typing import Any

from lion.core.session.branch import Branch
from lion.core.session.session import Session
from lion.operations.utils import RecursiveOperation
from lion.protocols.operatives.instruct import InstructModel

from .prompt import PROMPT


class PlanOperation(RecursiveOperation):
    """Operation for creating and executing multi-step plans."""

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
        # Remove num_steps from kwargs if it exists to avoid duplicate
        kwargs.pop("num_instruct", None)
        return self.prompt.format(num_instruct=num_instruct, **kwargs)

    async def run_instruction(
        self,
        ins: InstructModel,
        session: Session,
        branch: Branch,
        auto_run: bool,
        verbose: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Execute a single step of the plan.

        Args:
            ins: The instruction model for the step.
            session: The current session.
            branch: The branch to operate on.
            auto_run: Whether to automatically run nested instructions.
            verbose: Whether to enable verbose output.
            **kwargs: Additional keyword arguments.

        Returns:
            The result of the step execution.
        """
        if verbose:
            guidance_preview = (
                ins.guidance[:100] + "..." if len(ins.guidance) > 100 else ins.guidance
            )
            print(f"Running instruction: {guidance_preview}")

        config = {**ins.model_dump(), **kwargs}
        res = await branch.operate(**config)
        branch.msgs.logger.dump()
        return res  # Return the result directly, not wrapped in a list


# Create a singleton instance for the default plan operation
plan = PlanOperation()
