from typing import Any, Dict, Literal, Optional

from pydantic import Field

from .base import BaseConfig
from .imodel_config import iModelConfig
from .log_config import LogConfig
from .retry_config import RetryConfig


class MessageConfig(BaseConfig):
    """Configuration for message handling in Branch"""

    validation_mode: Literal["raise", "return_value", "return_none"] = Field(
        default="return_value",
        description="How to handle message validation failures",
    )
    auto_retries: bool = Field(
        False, description="Whether to automatically retry message parsing"
    )
    max_retries: int = Field(
        default=0, description="Maximum retries for message parsing"
    )
    allow_actions: bool = Field(
        default=True,
        description="Whether to allow action requests in messages",
    )
    auto_invoke_action: bool = Field(
        default=True, description="Whether to automatically invoke actions"
    )


class BranchConfig(BaseConfig):
    """Main configuration for Branch class.

    Combines all aspects of Branch configuration including logging,
    message handling, tool management, and iModel integration.
    """

    name: str | None = Field(default=None, description="Branch name for identification")
    user: str | None = Field(default=None, description="User ID/name for the branch")
    message_log_config: LogConfig = Field(
        default_factory=LogConfig,
        description="Configuration for log management",
    )
    action_log_config: LogConfig = Field(
        default_factory=LogConfig,
        description="Configuration for action log management",
    )
    message_config: MessageConfig = Field(
        default_factory=MessageConfig,
        description="Configuration for message handling",
    )
    auto_register_tools: bool = Field(
        default=True,
        description="Whether to automatically register tools when needed",
    )
    action_call_config: RetryConfig = Field(
        default_factory=RetryConfig,
        description="Configuration for action execution",
    )
    imodel_config: iModelConfig | None = Field(
        default=None, description="Configuration for iModel integration"
    )
    retry_imodel_config: iModelConfig | None = Field(
        default=None, description="Configuration for iModel integration"
    )
    kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional branch-specific configurations",
    )

    def __init__(self, **data):
        super().__init__(**data)
        if self.is_production:
            # Production environment restrictions
            self.auto_register_tools = (
                False  # Disable auto tool registration in production
            )
            self.message_config.auto_retries = False  # Disable auto retries
            self.message_config.validation_mode = (
                "raise"  # Strict validation in production
            )

    def validate_security(self) -> None:
        """Validate security-critical configuration settings."""
        super().validate_security()
        if self.is_production:
            if self.auto_register_tools:
                raise ValueError(
                    "Auto tool registration must be disabled in production"
                )
            if self.message_config.validation_mode != "raise":
                raise ValueError("Message validation must be strict in production")
            if self.message_config.auto_retries:
                raise ValueError("Auto retries must be disabled in production")


__all__ = [
    "MessageConfig",
    "BranchConfig",
]
