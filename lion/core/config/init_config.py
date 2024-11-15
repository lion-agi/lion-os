"""
Configuration initialization module.
This module handles setting up default environment variables and initial configuration
before any other imports to avoid circular dependencies.
"""

import os
from pathlib import Path


def init_config():
    """Initialize configuration environment variables."""
    environment = os.getenv("LION_ENV", "development")

    # Set secure default API key for development
    if environment == "development" and "LION_API_KEY" not in os.environ:
        os.environ["LION_API_KEY"] = (
            "dev_key_12345678901234567890123456789012345678901234567890"
        )

    # Set default config directory
    if "LION_CONFIG_DIR" not in os.environ:
        os.environ["LION_CONFIG_DIR"] = str(
            Path(__file__).parent.parent.parent / "config"
        )

    return {
        "environment": environment,
        "config_dir": os.getenv("LION_CONFIG_DIR"),
        "encryption_key": os.getenv("LION_ENCRYPTION_KEY"),
    }


# Initialize configuration on module import
config_env = init_config()

__all__ = ["config_env", "init_config"]
