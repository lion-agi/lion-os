import re
from typing import Any, Dict, List, Optional, Union

from pydantic import ConfigDict, Field, field_validator

from .base import BaseConfig, ConfigurationError


class iModelConfig(BaseConfig):
    """Configuration for intelligent model integration with enhanced security features."""

    # Override model_config to disable protected namespace warnings
    model_config = ConfigDict(
        protected_namespaces=(),
        validate_assignment=True,
        extra="allow",  # Allow extra fields like kwargs
        arbitrary_types_allowed=True,
    )

    # Model settings
    provider: str = Field(
        default="openai", description="AI model provider (e.g., openai, anthropic)"
    )
    model_name: str = Field(default="gpt-4o", description="Name of the model to use")
    api_key_var: str = Field(
        default="OPENAI_API_KEY", description="Environment variable name for API key"
    )
    api_base: str | None = Field(default=None, description="Base URL for API requests")

    # Request settings
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for model outputs",
    )
    max_tokens: int = Field(
        default=2000, description="Maximum number of tokens in response"
    )
    top_p: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Nucleus sampling parameter"
    )

    # Security settings
    require_encryption: bool = Field(
        default=False, description="Require encryption for API communication"
    )
    allowed_models: list[str] = Field(
        default_factory=list, description="List of allowed models"
    )
    content_filtering: bool = Field(
        default=True, description="Enable content filtering"
    )
    max_request_size: int = Field(
        default=4096, description="Maximum request size in tokens"
    )

    # Rate limiting
    rate_limit_enabled: bool = Field(default=True, description="Enable rate limiting")
    requests_per_minute: int = Field(
        default=60, description="Maximum requests per minute"
    )
    concurrent_requests: int = Field(
        default=10, description="Maximum concurrent requests"
    )

    # Cost management
    cost_tracking_enabled: bool = Field(
        default=True, description="Enable cost tracking"
    )
    budget_limit: float | None = Field(
        default=None, description="Monthly budget limit in USD"
    )
    cost_per_token: dict[str, float] = Field(
        default_factory=dict, description="Cost per token for different models"
    )

    # Caching
    cache_enabled: bool = Field(default=True, description="Enable response caching")
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")

    # Model-specific settings
    kwargs: dict[str, Any] = Field(
        default_factory=dict, description="Additional model-specific settings"
    )

    def __init__(self, **data):
        # Extract kwargs if present in data
        kwargs = data.pop("kwargs", {})
        super().__init__(**data)
        self.kwargs = kwargs

        if self.is_production:
            # Force security settings in production
            self.require_encryption = True
            self.content_filtering = True
            self.rate_limit_enabled = True
            self.cost_tracking_enabled = True

            # Mark security-critical fields as secure
            self.mark_field_secure("api_key_var")

            # Allow certain fields to be modified in production
            self.mark_field_mutable("temperature")
            self.mark_field_mutable("max_tokens")
            self.mark_field_mutable("requests_per_minute")

    @field_validator("api_base")
    @classmethod
    def validate_api_base(cls, v: str | None) -> str | None:
        """Validate API base URL."""
        if v is None:
            return v
        if not re.match(r"^https?://", v):
            raise ValueError("API base URL must start with http:// or https://")
        return v

    @field_validator("allowed_models")
    @classmethod
    def validate_allowed_models(cls, v: list[str]) -> list[str]:
        """Validate allowed models list."""
        if not v:
            return ["gpt-4", "gpt-3.5-turbo"]  # Default allowed models
        return v

    def validate_security(self) -> None:
        """Validate security-critical configuration settings."""
        super().validate_security()
        if self.is_production:
            if not self.require_encryption:
                raise ConfigurationError("Encryption must be enabled in production")
            if not self.content_filtering:
                raise ConfigurationError(
                    "Content filtering must be enabled in production"
                )
            if not self.rate_limit_enabled:
                raise ConfigurationError("Rate limiting must be enabled in production")
            if not self.cost_tracking_enabled:
                raise ConfigurationError("Cost tracking must be enabled in production")
            if not self.budget_limit:
                raise ConfigurationError("Budget limit must be set in production")
            if not self.api_base or not self.api_base.startswith("https://"):
                raise ConfigurationError(
                    "HTTPS is required for API communication in production"
                )
            if self.cache_enabled and self.cache_ttl > 86400:  # 24 hours
                raise ConfigurationError(
                    "Cache TTL cannot exceed 24 hours in production"
                )

    def get_model_cost(self, model_name: str, num_tokens: int) -> float:
        """Calculate cost for token usage."""
        if model_name not in self.cost_per_token:
            raise ValueError(f"No cost information available for model {model_name}")
        return self.cost_per_token[model_name] * num_tokens

    def is_model_allowed(self, model_name: str) -> bool:
        """Check if a model is allowed."""
        return model_name in self.allowed_models

    def get_rate_limit_config(self) -> dict[str, Any]:
        """Get rate limiting configuration."""
        return {
            "enabled": self.rate_limit_enabled,
            "requests_per_minute": self.requests_per_minute,
            "concurrent_requests": self.concurrent_requests,
        }

    def get_cache_config(self) -> dict[str, Any]:
        """Get caching configuration."""
        return {"enabled": self.cache_enabled, "ttl": self.cache_ttl}

    def get_security_config(self) -> dict[str, Any]:
        """Get security configuration."""
        return {
            "require_encryption": self.require_encryption,
            "content_filtering": self.content_filtering,
            "max_request_size": self.max_request_size,
        }
