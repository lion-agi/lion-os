import hashlib
import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Set, Type, TypeVar

import yaml
from cryptography.fernet import Fernet
from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="BaseConfig")


class ConfigurationError(Exception):
    """Base exception for configuration related errors."""

    pass


class EnvironmentError(ConfigurationError):
    """Raised when environment-specific configuration issues occur."""

    pass


class ValidationError(ConfigurationError):
    """Raised when configuration validation fails."""

    pass


class BaseConfig(BaseModel):
    """Base configuration class with enhanced security and validation features.

    This class provides the foundation for all configuration types with support for:
    - Thread-safe operations
    - Environment-specific behavior
    - Secure field handling
    - Version compatibility
    - Production environment restrictions
    """

    # Version tracking
    version: str = Field(default="1.0.0", description="Configuration version")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    environment: str = Field(
        default="development",
        description="Current environment (development/staging/production)",
    )

    # Metadata
    config_name: str = Field(default="base", description="Configuration identifier")
    description: str | None = Field(
        default=None, description="Configuration description"
    )

    # Security
    _encryption_key: bytes | None = None
    _secure_fields: set[str] = set()
    _mutable_fields: set[str] = set()

    model_config = ConfigDict(
        validate_assignment=True, extra="forbid", arbitrary_types_allowed=True
    )

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate environment setting."""
        valid_environments = {"development", "staging", "production"}
        if v not in valid_environments:
            raise ValueError(f"Environment must be one of {valid_environments}")
        return v

    def __init__(self, **data):
        super().__init__(**data)
        # Initialize thread lock as instance variable
        self._lock = threading.Lock()
        # Initialize mutable fields for non-production environments
        if self.environment != "production":
            self._mutable_fields = {
                field
                for field in self.model_fields.keys()
                if field not in {"version", "created_at", "environment"}
            }

    def configure_encryption(self, key: str | None = None) -> None:
        """Configure encryption for secure fields."""
        with self._lock:
            if key:
                self._encryption_key = key.encode()
            else:
                self._encryption_key = Fernet.generate_key()

    def _encrypt_value(self, value: str) -> str:
        """Encrypt sensitive configuration values."""
        if not self._encryption_key:
            raise ConfigurationError("Encryption key not configured")
        f = Fernet(self._encryption_key)
        return f.encrypt(value.encode()).decode()

    def _decrypt_value(self, value: str) -> str:
        """Decrypt sensitive configuration values."""
        if not self._encryption_key:
            raise ConfigurationError("Encryption key not configured")
        f = Fernet(self._encryption_key)
        return f.decrypt(value.encode()).decode()

    def get_secure_field(self, field_name: str) -> Any:
        """Safely retrieve a secure field value."""
        with self._lock:
            if field_name not in self._secure_fields:
                raise ValueError(f"{field_name} is not marked as a secure field")
            value = getattr(self, field_name)
            return self._decrypt_value(value) if value else None

    def mark_field_mutable(self, field_name: str) -> None:
        """Mark a field as mutable in production environment."""
        if field_name not in self.model_fields:
            raise ValueError(f"Field {field_name} does not exist")
        self._mutable_fields.add(field_name)

    def mark_field_secure(self, field_name: str) -> None:
        """Mark a field as secure, enabling encryption."""
        if field_name not in self.model_fields:
            raise ValueError(f"Field {field_name} does not exist")
        self._secure_fields.add(field_name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Override setattr to enforce production environment restrictions."""
        if name.startswith("_") or name not in self.model_fields:
            super().__setattr__(name, value)
            return

        if hasattr(
            self, "_lock"
        ):  # Check if lock exists (it won't during initialization)
            with self._lock:
                if (
                    self.environment == "production"
                    and name not in self._mutable_fields
                ):
                    raise ConfigurationError(
                        f"Cannot modify {name} in production environment"
                    )

                if name in self._secure_fields and value is not None:
                    value = self._encrypt_value(str(value))

                super().__setattr__(name, value)
        else:
            super().__setattr__(name, value)

    @classmethod
    def load_from_env(cls: type[T], prefix: str = "") -> T:
        """Load configuration from environment variables."""
        env_vars = {
            k.lower(): v
            for k, v in os.environ.items()
            if k.lower().startswith(prefix.lower())
        }
        return cls(**{k[len(prefix) :]: v for k, v in env_vars.items()})

    @classmethod
    def load_from_file(cls: type[T], path: Path) -> T:
        """Load configuration from a file (supports JSON and YAML)."""
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        content = path.read_text()
        if path.suffix == ".json":
            data = json.loads(content)
        elif path.suffix in {".yaml", ".yml"}:
            data = yaml.safe_load(content)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        return cls(**data)

    def save_to_file(self, path: Path, format: str = "json") -> None:
        """Save configuration to a file."""
        with self._lock:
            data = self.model_dump(exclude={"_encryption_key", "_lock"})

            # Create directory if it doesn't exist
            path.parent.mkdir(parents=True, exist_ok=True)

            if format == "json":
                path.write_text(json.dumps(data, indent=2, default=str))
            elif format == "yaml":
                path.write_text(yaml.dump(data, default_flow_style=False))
            else:
                raise ValueError(f"Unsupported format: {format}")

    def get_hash(self) -> str:
        """Generate a hash of the configuration for verification."""
        with self._lock:
            data = self.model_dump_json(exclude={"_encryption_key", "_lock"})
            return hashlib.sha256(data.encode()).hexdigest()

    def validate_security(self) -> None:
        """Validate security-critical configuration settings."""
        with self._lock:
            if self.environment == "production":
                # Ensure all secure fields are encrypted
                for field in self._secure_fields:
                    value = getattr(self, field)
                    if value is not None:
                        try:
                            self._decrypt_value(value)
                        except Exception as e:
                            raise ConfigurationError(
                                f"Field {field} is not properly encrypted: {str(e)}"
                            )

    def merge(self, other: "BaseConfig") -> None:
        """Merge another configuration into this one."""
        with self._lock:
            if self.environment == "production":
                raise ConfigurationError(
                    "Cannot merge configurations in production environment"
                )

            for field, value in other.model_dump().items():
                if hasattr(self, field) and value is not None:
                    setattr(self, field, value)

    @property
    def is_production(self) -> bool:
        """Check if current environment is production."""
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        """Check if current environment is development."""
        return self.environment == "development"

    def __str__(self) -> str:
        """String representation excluding sensitive data."""
        with self._lock:
            safe_dict = self.model_dump(exclude={"_encryption_key", "_lock"})
            return f"{self.__class__.__name__}({safe_dict})"

    def model_dump(self, *args, **kwargs):
        """Override model_dump to exclude unpicklable attributes."""
        kwargs["exclude"] = kwargs.get("exclude", set()) | {"_lock"}
        return super().model_dump(*args, **kwargs)

    def model_dump_json(self, *args, **kwargs):
        """Override model_dump_json to exclude unpicklable attributes."""
        kwargs["exclude"] = kwargs.get("exclude", set()) | {"_lock"}
        return super().model_dump_json(*args, **kwargs)
