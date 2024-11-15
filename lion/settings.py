"""
Lion Configuration System

This module provides a centralized configuration interface for the Lion framework.
It uses the new configuration system while maintaining backward compatibility.
"""

import os
from datetime import timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

from lion.core.config import (
    BranchConfig,
    LionIDConfig,
    LogConfig,
    MessageConfig,
    RetryConfig,
    iModelConfig,
)

# Initialize configuration environment before other imports
from lion.core.config.init_config import config_env
from lion.core.config.manager import ConfigurationManager


class BaseSystemFields(str, Enum):
    """Base system field names."""

    ln_id = "ln_id"
    TIMESTAMP = "timestamp"
    METADATA = "metadata"
    EXTRA_FIELDS = "extra_fields"
    CONTENT = "content"
    CREATED = "created"
    EMBEDDING = "embedding"


# Initialize configuration manager using environment from init_config
config_manager = ConfigurationManager.get_instance(
    config_dir=config_env["config_dir"],
    environment=config_env["environment"],
    encryption_key=config_env["encryption_key"],
)

# Legacy configurations maintained for backward compatibility
DEFAULT_ID_CONFIG = LionIDConfig(
    n=36,
    random_hyphen=True,
    num_hyphens=4,
    hyphen_start_index=6,
    hyphen_end_index=-6,
    prefix="ao",
    postfix="",
)

DEFAULT_RETRY_CONFIG = RetryConfig(
    max_retries=3,
    initial_delay=1.0,
    backoff_factor=2.0,
    max_delay=60.0,
    timeout=30.0,
    connect_timeout=10.0,
    read_timeout=30.0,
    circuit_breaker_enabled=True,
    rate_limit_enabled=True,
    log_retries=True,
    metrics_enabled=True,
)

# Use new model configuration system
model_config = config_manager.model_config

# Convert ModelConfig to iModelConfig
DEFAULT_CHAT_CONFIG = iModelConfig(
    provider=model_config.provider,
    api_key_var=model_config.api_key_env_var,
    model_name=model_config.model_name,
    rate_limit_enabled=True,
    requests_per_minute=60,  # Set a default value
    concurrent_requests=10,  # Set a default value
    temperature=model_config.temperature,
    max_tokens=model_config.max_tokens,
    stop_sequences=(
        list(model_config.stop_sequences) if model_config.stop_sequences else None
    ),
)

DEFAULT_RETRY_iMODEL_CONFIG = iModelConfig(
    provider=model_config.provider,
    api_key_var=model_config.api_key_env_var,
    model_name=model_config.model_name,
    rate_limit_enabled=True,
    requests_per_minute=60,  # Set a default value
    concurrent_requests=10,  # Set a default value
    temperature=0.5,
    max_tokens=model_config.max_tokens,
    stop_sequences=(
        list(model_config.stop_sequences) if model_config.stop_sequences else None
    ),
)

DEFAULT_MESSAGE_CONFIG = MessageConfig(
    validation_mode="raise",
    auto_retries=False,
    max_retries=0,
    allow_actions=True,
    auto_invoke_action=True,
)

DEFAULT_MESSAGE_LOG_CONFIG = LogConfig(
    enabled=True,
    level="INFO",
    file_logging=True,
    log_dir="./data/logs/messages",
    filename_pattern="%Y%m%d_%H%M%S.log",
    max_file_size=10 * 1024 * 1024,  # 10MB
    backup_count=5,
    retention_days=30,
    secure_logging=True,
    mask_sensitive_data=True,
)

DEFAULT_ACTION_LOG_CONFIG = LogConfig(
    enabled=True,
    level="INFO",
    file_logging=True,
    log_dir="./data/logs/actions",
    filename_pattern="%Y%m%d_%H%M%S.log",
    max_file_size=10 * 1024 * 1024,  # 10MB
    backup_count=5,
    retention_days=30,
    secure_logging=True,
    mask_sensitive_data=True,
)

DEFAULT_BRANCH_CONFIG = BranchConfig(
    name=None,
    user=None,
    message_log_config=DEFAULT_MESSAGE_LOG_CONFIG,
    action_log_config=DEFAULT_ACTION_LOG_CONFIG,
    message_config=DEFAULT_MESSAGE_CONFIG,
    auto_register_tools=True,
    action_call_config=DEFAULT_RETRY_CONFIG,
    imodel_config=DEFAULT_CHAT_CONFIG,
    retry_imodel_config=DEFAULT_RETRY_iMODEL_CONFIG,
)

DEFAULT_TIMEZONE = timezone.utc
BASE_LION_FIELDS = set(BaseSystemFields.__members__.values())


class Settings:
    """Global settings container with new configuration system integration."""

    class Config:
        """Basic configuration settings."""

        ID: LionIDConfig = DEFAULT_ID_CONFIG
        RETRY: RetryConfig = DEFAULT_RETRY_CONFIG
        TIMEZONE: timezone = DEFAULT_TIMEZONE

    class Branch:
        """Branch-related settings."""

        BRANCH: BranchConfig = DEFAULT_BRANCH_CONFIG

    class iModel:
        """Model integration settings using new configuration system."""

        @property
        def CHAT(self) -> dict[str, Any]:
            """Get current model configuration."""
            return config_manager.model_config.get_model_config()

        @property
        def PARSE(self) -> Any:
            """Get model configuration for parsing."""
            return config_manager.model_config

    class API:
        """API settings using new configuration system."""

        @property
        def KEYS(self) -> set:
            """Get current API keys."""
            return config_manager.api_config.api_keys

        @property
        def MIN_KEY_LENGTH(self) -> int:
            """Get minimum API key length."""
            return config_manager.api_config.min_key_length

        @property
        def ALLOW_DEV_KEYS(self) -> bool:
            """Check if development keys are allowed."""
            return config_manager.api_config.allow_dev_keys

    @classmethod
    def reload(cls) -> None:
        """Reload all configurations."""
        config_manager.load_all()
        config_manager.validate_all()

    @classmethod
    def is_production(cls) -> bool:
        """Check if running in production environment."""
        return config_manager.is_production

    @classmethod
    def is_development(cls) -> bool:
        """Check if running in development environment."""
        return config_manager.is_development

    @classmethod
    def get_config_manager(cls) -> ConfigurationManager:
        """Get the configuration manager instance."""
        return config_manager


# Initialize settings
settings = Settings()

__all__ = [
    "settings",
    "config_manager",
    "BaseSystemFields",
    "BASE_LION_FIELDS",
    "DEFAULT_TIMEZONE",
]
