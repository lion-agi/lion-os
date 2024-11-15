import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Set, Union

import yaml
from pydantic import ConfigDict, Field, field_validator

from .base import BaseConfig, ConfigurationError


class ModelConfig(BaseConfig):
    """Configuration for AI model settings with enhanced security."""

    # Model Provider Settings
    provider: str = Field(
        ..., description="AI model provider (e.g., 'openai', 'anthropic', 'local')"
    )
    model_name: str = Field(..., description="Name of the model to use")

    # API Authentication
    api_key_env_var: str = Field(
        ..., description="Environment variable name containing the API key"
    )

    # Rate Limiting
    requests_per_minute: int | None = Field(
        default=None, description="Maximum API requests per minute"
    )
    tokens_per_minute: int | None = Field(
        default=None, description="Maximum tokens per minute"
    )

    # Model Parameters
    max_tokens: int | None = Field(
        default=None, description="Maximum tokens per request"
    )
    temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="Temperature for response generation"
    )

    # Fallback Configuration
    fallback_models: dict[str, str] = Field(
        default_factory=dict, description="Fallback models mapped by scenario"
    )
    max_retries: int = Field(
        default=3, description="Maximum number of retries before fallback"
    )

    # Security Settings
    allowed_models: set[str] = Field(
        default_factory=set, description="Set of allowed models"
    )
    require_encryption: bool = Field(
        default=True, description="Whether to require encryption for API communication"
    )

    # Advanced Settings
    context_window: int | None = Field(
        default=None, description="Maximum context window size in tokens"
    )
    response_format: str | None = Field(
        default=None, description="Expected format of model responses"
    )
    stop_sequences: set[str] = Field(
        default_factory=set, description="Custom stop sequences for text generation"
    )

    model_config = ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=True,
        protected_namespaces=(),  # Disable protected namespaces to avoid model_name warning
    )

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        """Validate the model provider."""
        valid_providers = {"openai", "anthropic", "local", "azure", "cohere"}
        if v.lower() not in valid_providers:
            raise ValueError(f"Provider must be one of {valid_providers}")
        return v.lower()

    @field_validator("api_key_env_var")
    @classmethod
    def validate_api_key_env_var(cls, v: str) -> str:
        """Validate the API key environment variable."""
        if not os.getenv(v):
            raise ConfigurationError(
                f"Environment variable '{v}' not set. Please set your API key."
            )
        return v

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        """Validate temperature is within acceptable range."""
        if not 0.0 <= v <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        return v

    @field_validator("fallback_models")
    @classmethod
    def validate_fallback_models(cls, v: dict[str, str], info) -> dict[str, str]:
        """Validate fallback models exist in allowed models."""
        values = info.data
        allowed = values.get("allowed_models", set())
        if allowed:
            for model in v.values():
                if model not in allowed:
                    raise ValueError(f"Fallback model {model} not in allowed models")
        return v

    def get_api_key(self) -> str:
        """Securely retrieve API key from environment."""
        api_key = os.getenv(self.api_key_env_var)
        if not api_key:
            raise ConfigurationError(
                f"API key not found in environment variable {self.api_key_env_var}"
            )
        return self._decrypt_value(api_key) if self.require_encryption else api_key

    @classmethod
    def from_environment(cls, prefix: str = "LION_MODEL_") -> "ModelConfig":
        """Create model configuration from environment variables."""
        env_vars = {
            k[len(prefix) :].lower(): v
            for k, v in os.environ.items()
            if k.startswith(prefix)
        }

        # Required fields
        provider = env_vars.get("provider")
        model_name = env_vars.get("model_name")
        api_key_env_var = env_vars.get("api_key_var")

        if not all([provider, model_name, api_key_env_var]):
            raise ConfigurationError(
                "Missing required environment variables. "
                f"Please set {prefix}PROVIDER, {prefix}MODEL_NAME, "
                f"and {prefix}API_KEY_VAR"
            )

        # Optional fields with defaults
        config_data = {
            "provider": provider,
            "model_name": model_name,
            "api_key_env_var": api_key_env_var,
            "temperature": float(env_vars.get("temperature", 0.7)),
            "max_retries": int(env_vars.get("max_retries", 3)),
            "require_encryption": env_vars.get("require_encryption", "1").lower()
            in ("1", "true"),
        }

        # Optional numeric fields
        for field in [
            "requests_per_minute",
            "tokens_per_minute",
            "max_tokens",
            "context_window",
        ]:
            if field in env_vars:
                config_data[field] = int(env_vars[field])

        # Optional string fields
        for field in ["response_format"]:
            if field in env_vars:
                config_data[field] = env_vars[field]

        # Set fields
        for field in ["allowed_models", "stop_sequences"]:
            if field in env_vars:
                config_data[field] = set(env_vars[field].split(","))

        # Dictionary fields
        if "fallback_models" in env_vars:
            try:
                config_data["fallback_models"] = json.loads(env_vars["fallback_models"])
            except json.JSONDecodeError:
                raise ConfigurationError(
                    f"Invalid JSON format for {prefix}FALLBACK_MODELS"
                )

        return cls(**config_data)

    def get_model_config(self) -> dict:
        """Get model-specific configuration."""
        return {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stop_sequences": (
                list(self.stop_sequences) if self.stop_sequences else None
            ),
            "response_format": self.response_format,
        }

    def validate_security(self) -> None:
        """Validate security-critical configuration settings."""
        super().validate_security()

        if self.is_production:
            if not self.require_encryption:
                raise ConfigurationError("Encryption must be enabled in production")
            if not self.allowed_models:
                raise ConfigurationError(
                    "Allowed models must be specified in production"
                )
            if self.temperature > 1.0:
                raise ConfigurationError(
                    "Temperature must not exceed 1.0 in production"
                )
