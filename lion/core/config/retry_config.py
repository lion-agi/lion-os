from datetime import timedelta
from typing import Any, Dict, List, Optional, Union

from pydantic import Field

from .base import BaseConfig


class RetryConfig(BaseConfig):
    """Configuration for retry behavior with enhanced security features."""

    # Basic retry settings
    max_retries: int = Field(default=3, description="Maximum number of retry attempts")
    initial_delay: float = Field(
        default=1.0, description="Initial delay between retries in seconds"
    )
    max_delay: float = Field(
        default=60.0, description="Maximum delay between retries in seconds"
    )
    backoff_factor: float = Field(
        default=2.0, description="Multiplicative factor for backoff between retries"
    )

    # Retry conditions
    retry_on_exceptions: list[str] = Field(
        default_factory=lambda: ["TimeoutError", "ConnectionError"],
        description="Exception types that trigger retries",
    )
    retry_on_status_codes: list[int] = Field(
        default_factory=lambda: [429, 500, 502, 503, 504],
        description="HTTP status codes that trigger retries",
    )

    # Timeout settings
    timeout: float = Field(default=30.0, description="Operation timeout in seconds")
    connect_timeout: float = Field(
        default=10.0, description="Connection timeout in seconds"
    )
    read_timeout: float = Field(default=30.0, description="Read timeout in seconds")

    # Circuit breaker settings
    circuit_breaker_enabled: bool = Field(
        default=True, description="Enable circuit breaker pattern"
    )
    failure_threshold: int = Field(
        default=5, description="Number of failures before circuit opens"
    )
    recovery_timeout: float = Field(
        default=60.0, description="Time in seconds before attempting recovery"
    )

    # Rate limiting
    rate_limit_enabled: bool = Field(default=True, description="Enable rate limiting")
    rate_limit_requests: int = Field(
        default=100, description="Number of requests allowed per window"
    )
    rate_limit_window: float = Field(
        default=60.0, description="Time window for rate limiting in seconds"
    )

    # Monitoring and logging
    log_retries: bool = Field(default=True, description="Log retry attempts")
    metrics_enabled: bool = Field(
        default=True, description="Enable retry metrics collection"
    )

    def __init__(self, **data):
        super().__init__(**data)
        if self.is_production:
            # Force security settings in production
            self.circuit_breaker_enabled = True
            self.rate_limit_enabled = True

            # Mark security-critical fields as secure
            self.mark_field_secure("failure_threshold")

            # Allow certain fields to be modified in production
            self.mark_field_mutable("max_retries")
            self.mark_field_mutable("timeout")
            self.mark_field_mutable("rate_limit_requests")

    def validate_security(self) -> None:
        """Validate security-critical configuration settings."""
        super().validate_security()
        if self.is_production:
            if not self.circuit_breaker_enabled:
                raise ValueError("Circuit breaker must be enabled in production")
            if not self.rate_limit_enabled:
                raise ValueError("Rate limiting must be enabled in production")
            if self.max_retries > 5:
                raise ValueError("Maximum retries cannot exceed 5 in production")
            if self.timeout > 60.0:
                raise ValueError("Timeout cannot exceed 60 seconds in production")
            if self.failure_threshold < 3:
                raise ValueError("Failure threshold must be at least 3 in production")

    def get_retry_delay(self, attempt: int) -> float:
        """Calculate the delay for a specific retry attempt."""
        delay = self.initial_delay * (self.backoff_factor ** (attempt - 1))
        return min(delay, self.max_delay)

    def should_retry(
        self, exception: Exception | None = None, status_code: int | None = None
    ) -> bool:
        """Determine if a retry should be attempted."""
        if exception and exception.__class__.__name__ in self.retry_on_exceptions:
            return True
        if status_code and status_code in self.retry_on_status_codes:
            return True
        return False

    def get_timeout(self, operation: str | None = None) -> float | dict[str, float]:
        """Get timeout settings for an operation."""
        if operation == "connect":
            return self.connect_timeout
        elif operation == "read":
            return self.read_timeout
        elif operation == "full":
            return {
                "connect": self.connect_timeout,
                "read": self.read_timeout,
                "total": self.timeout,
            }
        return self.timeout

    def get_circuit_breaker_config(self) -> dict[str, Any]:
        """Get circuit breaker configuration."""
        return {
            "enabled": self.circuit_breaker_enabled,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout": self.recovery_timeout,
        }

    def get_rate_limit_config(self) -> dict[str, Any]:
        """Get rate limiting configuration."""
        return {
            "enabled": self.rate_limit_enabled,
            "requests": self.rate_limit_requests,
            "window": self.rate_limit_window,
        }
