from typing import Any

from pydantic import field_validator

from lion.core.generic.element import Element
from lion.core.typing import ID, Communicatable, Field

from .utils import validate_sender_recipient


class BaseMail(Element, Communicatable):
    """
    Base class for mail-like communication in the LION system.

    Attributes:
        sender (str): The ID of the sender node.
        recipient (str): The ID of the recipient node.
    """

    sender: ID.SenderRecipient = Field(
        default="N/A",
        title="Sender",
        description="The ID of the sender node or a role.",
    )

    recipient: ID.SenderRecipient = Field(
        default="N/A",
        title="Recipient",
        description="The ID of the recipient node, or a role",
    )

    @field_validator("sender", "recipient", mode="before")
    @classmethod
    def _validate_sender_recipient(cls, value: Any) -> ID.SenderRecipient:
        return validate_sender_recipient(value)


# File: autoos/communication/base.py
