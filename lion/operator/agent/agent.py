from datetime import datetime
from functools import partial
from typing import Any, Optional

from lion.core.generic import LogManager
from lion.core.session.branch import Branch
from lion.core.session.session import Session
from lion.core.session.session_manager import SessionManager
from lion.libs.parse import to_dict
from lion.operations.brainstorm.brainstorm import brainstorm
from lion.operations.plan.plan import plan
from lion.operations.select.select import select
from lion.protocols.operatives.instruct import Instruct


def serialize_result(result: Any) -> Any:
    """Serialize result to JSON-compatible format."""
    if hasattr(result, "clean_dump"):
        return result.clean_dump()
    if hasattr(result, "model_dump"):
        return result.model_dump()
    if isinstance(result, list):
        return [serialize_result(r) for r in result]
    if isinstance(result, dict):
        return {k: serialize_result(v) for k, v in result.items()}
    if isinstance(result, type):  # Handle ModelMetaclass
        return result.__name__
    return result


class Agent:
    """Agent that orchestrates operations and maintains state through sessions."""

    def __init__(
        self,
        session_id: str | None = None,
        session: Session | None = None,
        branch: Branch | None = None,
        session_manager: SessionManager | None = None,
    ):
        """
        Initialize Agent.

        Args:
            session_id: Optional session ID to load existing session
            session: Optional session object
            branch: Optional branch object
            session_manager: Optional session manager instance
        """
        self.session_manager = session_manager or SessionManager()

        if session_id:
            self.session = self.session_manager.get_session(session_id)
            self.session_id = session_id
        else:
            self.session = session or Session()
            self.session_id = self.session_manager.create_session()

        self.branch = branch or self.session.default_branch or self.session.new_branch()

    def _save_state(self) -> None:
        """Save current session state."""
        self.session_manager.save_session(self.session_id, self.session)

    def _track_operation(self, operation: str, instruct: dict, result: Any) -> None:
        """Track operation in history."""
        self.session_manager.save_operation_history(
            self.session_id,
            operation,
            serialize_result(instruct),
            serialize_result(result),
        )

    async def brainstorm(
        self, instruct: Instruct | dict[str, Any], **kwargs: Any
    ) -> Any:
        """
        Execute brainstorm operation.

        Args:
            instruct: Instructions for brainstorming
            **kwargs: Additional arguments

        Returns:
            Brainstorming results
        """
        try:
            result, session = await brainstorm(
                instruct=instruct,
                session=self.session,
                branch=self.branch,
                return_session=True,
                **kwargs,
            )

            # Update session and branch
            self.session = session
            self.branch = session.default_branch

            # Track operation
            self._track_operation("brainstorm", instruct, result)

            # Save state
            self._save_state()

            return result

        except Exception as e:
            # Log error and re-raise
            self._track_operation("brainstorm_error", instruct, {"error": str(e)})
            raise

    async def plan(self, instruct: Instruct | dict[str, Any], **kwargs: Any) -> Any:
        """
        Execute plan operation.

        Args:
            instruct: Planning instructions
            **kwargs: Additional arguments

        Returns:
            Planning results
        """
        try:
            result, session = await plan(
                instruct=instruct,
                session=self.session,
                branch=self.branch,
                return_session=True,
                **kwargs,
            )

            # Update session and branch
            self.session = session
            self.branch = session.default_branch

            # Track operation
            self._track_operation("plan", instruct, result)

            # Save state
            self._save_state()

            return result

        except Exception as e:
            # Log error and re-raise
            self._track_operation("plan_error", instruct, {"error": str(e)})
            raise

    async def select(
        self, instruct: Instruct | dict[str, Any], choices: list[str], **kwargs: Any
    ) -> Any:
        """
        Execute select operation.

        Args:
            instruct: Selection instructions
            choices: List of choices
            **kwargs: Additional arguments

        Returns:
            Selection results
        """
        try:
            result, branch = await select(
                instruct=instruct,
                choices=choices,
                branch=self.branch,
                return_branch=True,
                **kwargs,
            )

            # Update branch
            self.branch = branch

            # Track operation
            self._track_operation(
                "select", {"instruct": instruct, "choices": choices}, result
            )

            # Save state
            self._save_state()

            return result

        except Exception as e:
            # Log error and re-raise
            self._track_operation(
                "select_error",
                {"instruct": instruct, "choices": choices},
                {"error": str(e)},
            )
            raise

    async def run(self, instruct: Instruct | dict[str, Any], **kwargs: Any) -> Any:
        """
        Run a complete agent workflow (brainstorm -> plan).

        Args:
            instruct: Initial instructions
            **kwargs: Additional arguments

        Returns:
            Workflow results
        """
        try:
            # Execute brainstorm
            brainstorm_result = await self.brainstorm(instruct, **kwargs)

            # Use brainstorm result for planning
            plan_instruct = (
                brainstorm_result.initial
                if hasattr(brainstorm_result, "initial")
                else instruct
            )
            plan_instruct = (
                plan_instruct.instruct_models[0]
                if isinstance(plan_instruct, Instruct)
                else plan_instruct.model_dump()
            )

            # Execute plan
            plan_result = await self.plan(plan_instruct, **kwargs)

            return plan_result

        except Exception as e:
            # Log error and re-raise
            self._track_operation("run_error", instruct, {"error": str(e)})
            raise

    def get_history(self) -> list[dict]:
        """Get operation history for this agent's session."""
        return self.session_manager.get_operation_history(self.session_id)
