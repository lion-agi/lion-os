from datetime import timedelta
from typing import Any, Dict, List, Optional

from pydantic import Field

from .base import BaseConfig


class LogConfig(BaseConfig):
    """Configuration for log management with enhanced security features."""

    # Basic logging settings
    enabled: bool = Field(default=True, description="Whether logging is enabled")
    level: str = Field(default="INFO", description="Default logging level")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log message format",
    )

    # File logging settings
    file_logging: bool = Field(default=False, description="Enable logging to file")
    log_dir: str | None = Field(default="logs", description="Directory for log files")
    filename_pattern: str = Field(
        default="%Y%m%d_%H%M%S.log", description="Pattern for log filenames"
    )
    max_file_size: int = Field(
        default=10 * 1024 * 1024,  # 10MB
        description="Maximum size of log files in bytes",
    )
    backup_count: int = Field(default=5, description="Number of backup files to keep")

    # Retention settings
    retention_days: int = Field(default=30, description="Number of days to retain logs")

    # Security settings
    secure_logging: bool = Field(
        default=False, description="Enable secure logging features"
    )
    mask_sensitive_data: bool = Field(
        default=True, description="Mask sensitive data in logs"
    )
    sensitive_fields: list[str] = Field(
        default_factory=lambda: ["password", "token", "key", "secret"],
        description="Fields to mask in logs",
    )

    # Performance settings
    buffer_size: int = Field(default=1000, description="Size of the logging buffer")
    flush_interval: timedelta = Field(
        default=timedelta(seconds=5), description="Interval for flushing logs"
    )

    # Additional settings
    extra_handlers: dict[str, dict[str, Any]] = Field(
        default_factory=dict, description="Additional log handlers configuration"
    )

    def __init__(self, **data):
        super().__init__(**data)
        if self.is_production:
            # Force security settings in production
            self.secure_logging = True
            self.mask_sensitive_data = True
            self.level = "INFO"  # Enforce INFO level or higher in production

            # Mark security-critical fields as secure
            self.mark_field_secure("sensitive_fields")

            # Allow certain fields to be modified in production
            self.mark_field_mutable("level")
            self.mark_field_mutable("buffer_size")
            self.mark_field_mutable("flush_interval")

    def validate_security(self) -> None:
        """Validate security-critical configuration settings."""
        super().validate_security()
        if self.is_production:
            if not self.secure_logging:
                raise ValueError("Secure logging must be enabled in production")
            if not self.mask_sensitive_data:
                raise ValueError("Sensitive data masking must be enabled in production")
            if self.level.upper() not in ["INFO", "WARNING", "ERROR", "CRITICAL"]:
                raise ValueError("Log level must be INFO or higher in production")
            if self.file_logging and not self.log_dir.startswith("/var/log/"):
                raise ValueError("Production logs must be stored in /var/log/")
            if self.retention_days < 90:
                raise ValueError("Log retention must be at least 90 days in production")

    def get_masked_value(self, value: str) -> str:
        """Get masked version of a sensitive value."""
        if not value:
            return value
        return f"{value[:2]}{'*' * (len(value) - 4)}{value[-2:]}"

    def should_mask_field(self, field_name: str) -> bool:
        """Check if a field should be masked."""
        return self.mask_sensitive_data and any(
            sensitive in field_name.lower() for sensitive in self.sensitive_fields
        )

    def get_log_path(self, name: str | None = None) -> str:
        """Get the full path for a log file."""
        import os
        from datetime import datetime

        filename = datetime.now().strftime(self.filename_pattern)
        if name:
            filename = f"{name}_{filename}"

        return os.path.join(self.log_dir, filename)

    def configure_handler(self, handler_name: str, **kwargs) -> None:
        """Configure a logging handler with the given settings."""
        if handler_name not in self.extra_handlers:
            self.extra_handlers[handler_name] = {}
        self.extra_handlers[handler_name].update(kwargs)
