from typing_extensions import override

from lion.core.typing import ID, Any, Note
from lion.libs.utils import copy

from .action_request import ActionRequest
from .message import MessageFlag, MessageRole, RoledMessage


def prepare_action_response_content(
    action_request: ActionRequest,
    output: Any,
) -> Note:
    """
    Prepare the content for an action response.

    Args:
        action_request: The original action request.
        func_output: The output from the function execution.

    Returns:
        Note: A Note object containing the action response content.
    """

    return Note(
        action_request_id=action_request.ln_id,
        action_response={
            "function": action_request.function,
            "arguments": action_request.arguments,
            "output": output,
        },
    )


class ActionResponse(RoledMessage):
    """Represents a response to an action request in the system."""

    @override
    def __init__(
        self,
        action_request: ID[ActionRequest].Item,
        output: Any = None,
        protected_init_params: dict | None = None,
    ) -> None:
        """
        Initialize an ActionResponse instance.

        Args:
            action_request: The original action request to respond to.
            sender: The sender of the action response.
            func_output: The output from the function in the request.
            protected_init_params: Protected initialization parameters.
        """
        message_flags = [
            action_request,
            output,
        ]

        if all(x == MessageFlag.MESSAGE_LOAD for x in message_flags):
            protected_init_params = protected_init_params or {}
            super().__init__(**protected_init_params)
            return

        if all(x == MessageFlag.MESSAGE_CLONE for x in message_flags):
            super().__init__(role=MessageRole.ASSISTANT)
            return

        super().__init__(
            role=MessageRole.ASSISTANT,
            recipient=action_request.sender,
            sender=action_request.recipient,
            content=prepare_action_response_content(
                action_request=action_request,
                output=output,
            ),
        )
        action_request.content["action_response_id"] = self.ln_id

    @property
    def function(self) -> str:
        """Get the function name from the action response."""
        return copy(self.content.get(["action_response", "function"]))

    @property
    def arguments(self) -> dict[str, Any]:
        """Get the function arguments from the action response."""
        return copy(self.content.get(["action_response", "arguments"]))

    @property
    def output(self) -> Any:
        """Get the function output from the action response."""
        return self.content.get(["action_response", "output"])

    @property
    def response(self) -> dict[str, Any]:
        """Get the action response as a dictionary."""
        return copy(self.content.get("action_response", {}))

    @property
    def action_request_id(self) -> ID[ActionRequest].ID | None:
        """Get the ID of the corresponding action request."""
        return copy(self.content.get("action_request_id", None))

    @override
    def _format_content(self) -> dict[str, Any]:
        return {"role": self.role.value, "content": self.response}


# File: autoos/communication/action_response.py
