from typing_extensions import override

from lion.core.typing import ID, Any, BaseModel, JsonValue, Note
from lion.libs.parse import to_str
from lion.libs.utils import copy

from .message import MessageFlag, MessageRole, RoledMessage


def prepare_assistant_response(
    assistant_response: BaseModel | list[BaseModel] | dict | str | Any, /
):
    if assistant_response:
        content = Note()
        # must be response.choices[0].message.content
        if isinstance(assistant_response, BaseModel):
            content["assistant_response"] = (
                assistant_response.choices[0].message.content or ""
            )
            content["model_response"] = assistant_response.model_dump(
                exclude_none=True, exclude_unset=True
            )
        # or response[i].choices[0].delta.content
        elif isinstance(assistant_response, list):
            msg = "".join(
                [i.choices[0].delta.content or "" for i in assistant_response]
            )
            content["assistant_response"] = msg
            content["model_response"] = [
                i.model_dump(
                    exclude_none=True,
                    exclude_unset=True,
                )
                for i in assistant_response
            ]
        elif isinstance(assistant_response, dict) and "content" in assistant_response:
            content["assistant_response"] = assistant_response["content"]
        elif isinstance(assistant_response, str):
            content["assistant_response"] = assistant_response
        else:
            content["assistant_response"] = to_str(assistant_response)
        return content
    else:
        return Note(assistant_response="")


class AssistantResponse(RoledMessage):
    """Represents a response from an assistant in the system."""

    @override
    def __init__(
        self,
        assistant_response: BaseModel | JsonValue | MessageFlag,
        sender: ID.Ref | MessageFlag,
        recipient: ID.Ref | MessageFlag,
        protected_init_params: dict | None = None,
    ) -> None:
        """
        Initialize an AssistantResponse instance.

        Args:
            assistant_response: The content of the assistant's response.
            sender: The sender of the response, typically the assistant.
            recipient: The recipient of the response.
            protected_init_params: Optional parameters for protected init.
        """
        message_flags = [assistant_response, sender, recipient]

        if all(x == MessageFlag.MESSAGE_LOAD for x in message_flags):
            protected_init_params = protected_init_params or {}
            super().__init__(**protected_init_params)
            return

        if all(x == MessageFlag.MESSAGE_CLONE for x in message_flags):
            super().__init__(role=MessageRole.ASSISTANT)
            return

        super().__init__(
            role=MessageRole.ASSISTANT,
            sender=sender or "N/A",
            recipient=recipient,
        )

        content = prepare_assistant_response(assistant_response)
        if "model_response" in content:
            self.metadata["model_response"] = content.pop("model_response")
        self.content = content

    @property
    def response(self) -> str:
        """
        Return the assistant response content.

        Returns:
            Any: The content of the assistant's response.
        """
        return copy(self.content["assistant_response"])

    @property
    def model_response(self) -> dict | list[dict]:
        """
        Return the model response content.

        Returns:
            Any: The content of the model response.
        """
        return copy(self.metadata.get("model_response", {}))

    @override
    def _format_content(self) -> dict[str, str]:
        return {"role": self.role.value, "content": self.response}


# File: autoos/communication/assistant_response.py
