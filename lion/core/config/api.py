import os
import re
from typing import Optional, Set

from pydantic import ConfigDict, Field, field_validator

from .base import BaseConfig, ConfigurationError


class APIConfig(BaseConfig):
    """Configuration for API-related settings with enhanced security."""

    # API Authentication
    api_keys: set[str] = Field(default_factory=set, description="Set of valid API keys")
    min_key_length: int = Field(
        default=32, description="Minimum length requirement for API keys"
    )
    key_pattern: str = Field(
        default=r"^[a-zA-Z0-9-_]{32,}$",
        description="Regular expression pattern for valid API keys",
    )

    # Rate Limiting
    rate_limit_enabled: bool = Field(
        default=True, description="Enable/disable rate limiting"
    )
    rate_limit_requests: int = Field(
        default=100, description="Maximum requests per time window"
    )
    rate_limit_window: int = Field(
        default=60, description="Time window in seconds for rate limiting"
    )

    # Security Settings
    allow_dev_keys: bool = Field(
        default=False, description="Whether to allow development API keys"
    )
    require_https: bool = Field(
        default=True, description="Require HTTPS for all API requests"
    )
    cors_origins: set[str] = Field(
        default_factory=set, description="Allowed CORS origins"
    )

    # Timeouts
    request_timeout: int = Field(default=30, description="Request timeout in seconds")
    long_polling_timeout: int = Field(
        default=300, description="Long polling timeout in seconds"
    )

    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)

    @field_validator("api_keys")
    @classmethod
    def validate_api_keys(cls, v: set[str], info):
        """Validate API keys against security requirements."""
        values = info.data
        min_length = values.get("min_key_length", 32)
        pattern = values.get("key_pattern", r"^[a-zA-Z0-9-_]{32,}$")

        for key in v:
            if len(key) < min_length:
                raise ValueError(
                    f"API key length must be at least {min_length} characters"
                )
            if not re.match(pattern, key):
                raise ValueError(
                    f"API key format invalid. Must match pattern: {pattern}"
                )
        return v

    @field_validator("cors_origins")
    @classmethod
    def validate_cors_origins(cls, v: set[str], info):
        """Validate CORS origins."""
        values = info.data
        if values.get("environment") == "production":
            for origin in v:
                if origin == "*":
                    raise ValueError("Wildcard CORS origin not allowed in production")
                if not origin.startswith(("https://", "http://localhost")):
                    raise ValueError(f"Invalid CORS origin in production: {origin}")
        return v

    @classmethod
    def from_environment(cls) -> "APIConfig":
        """Create API configuration from environment variables."""
        api_key = os.environ.get("LION_API_KEY")
        if not api_key:
            raise ConfigurationError("LION_API_KEY environment variable not set")

        environment = os.environ.get("LION_ENV", "development")
        allow_dev_keys = environment != "production"

        cors_origins = set(
            filter(None, os.environ.get("LION_CORS_ORIGINS", "").split(","))
        )

        return cls(
            api_keys={api_key},
            environment=environment,
            allow_dev_keys=allow_dev_keys,
            cors_origins=cors_origins,
            rate_limit_enabled=os.environ.get("LION_RATE_LIMIT_ENABLED", "1").lower()
            in ("1", "true"),
            rate_limit_requests=int(os.environ.get("LION_RATE_LIMIT_REQUESTS", "100")),
            rate_limit_window=int(os.environ.get("LION_RATE_LIMIT_WINDOW", "60")),
        )

    def validate_api_key(self, key: str) -> bool:
        """Validate if an API key is valid."""
        if not self.allow_dev_keys and len(key) < self.min_key_length:
            return False
        return key in self.api_keys

    def add_api_key(self, key: str) -> None:
        """Add a new API key."""
        if not re.match(self.key_pattern, key):
            raise ValueError(
                f"API key format invalid. Must match pattern: {self.key_pattern}"
            )
        self.api_keys.add(key)

    def remove_api_key(self, key: str) -> None:
        """Remove an API key."""
        self.api_keys.discard(key)

    def is_cors_allowed(self, origin: str) -> bool:
        """Check if a CORS origin is allowed."""
        if "*" in self.cors_origins and not self.is_production:
            return True
        return origin in self.cors_origins

    def validate_security(self) -> None:
        """Validate security-critical configuration settings."""
        super().validate_security()

        if self.is_production:
            if not self.require_https:
                raise ConfigurationError("HTTPS must be required in production")
            if self.allow_dev_keys:
                raise ConfigurationError(
                    "Development API keys not allowed in production"
                )
            if "*" in self.cors_origins:
                raise ConfigurationError(
                    "Wildcard CORS origin not allowed in production"
                )
