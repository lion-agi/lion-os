import json
import os
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar, Union

import yaml

from .api import APIConfig
from .base import BaseConfig, ConfigurationError
from .model import ModelConfig

T = TypeVar("T", bound=BaseConfig)


class ConfigurationManager:
    """Central manager for all Lion configuration.

    This class manages loading, validation, and access to all configuration
    components while ensuring proper security and environment handling.

    Thread-safe singleton implementation ensures consistent global state
    across all application components.
    """

    _instance = None
    _lock = threading.Lock()

    def __init__(
        self,
        config_dir: str | Path | None = None,
        environment: str | None = None,
        encryption_key: str | None = None,
    ):
        self._init_lock = threading.Lock()
        self.config_dir = Path(config_dir) if config_dir else Path("config")
        self.environment = environment or os.getenv("LION_ENV", "development")
        self._encryption_key = encryption_key
        self._configs: dict[str, BaseConfig] = {}

        # Initialize core configurations
        self.api_config = self._load_or_create(APIConfig)
        self.model_config = self._load_or_create(ModelConfig)

        # Configure encryption if key provided
        if encryption_key:
            for config in self._configs.values():
                config.configure_encryption(encryption_key)

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "development"

    def _load_or_create(self, config_class: type[T], **kwargs) -> T:
        """Load configuration from file or create new instance."""
        with self._init_lock:
            config_name = config_class.__name__.lower()

            # Try loading from environment first
            try:
                config = config_class.from_environment()
                self._configs[config_name] = config
                return config
            except ConfigurationError:
                pass

            # Try loading from file
            config_file = self.config_dir / f"{config_name}.{self.environment}.yaml"
            if config_file.exists():
                try:
                    config = config_class.load_from_file(config_file)
                    self._configs[config_name] = config
                    return config
                except Exception as e:
                    raise ConfigurationError(
                        f"Error loading {config_name} configuration: {str(e)}"
                    )

            # Create new configuration
            config = config_class(environment=self.environment, **kwargs)
            self._configs[config_name] = config
            return config

    def save_all(self, format: str = "yaml") -> None:
        """Save all configurations to files."""
        with self._init_lock:
            self.config_dir.mkdir(parents=True, exist_ok=True)

            for name, config in self._configs.items():
                file_path = self.config_dir / f"{name}.{self.environment}.{format}"
                config.save_to_file(file_path, format=format)

    def load_all(self) -> None:
        """Reload all configurations from files."""
        with self._init_lock:
            for config_class in [APIConfig, ModelConfig]:
                self._load_or_create(config_class)

    def validate_all(self) -> None:
        """Validate all configurations."""
        with self._init_lock:
            # Validate individual configurations
            for config in self._configs.values():
                config.validate_security()

            # Additional cross-config validations
            if self.is_production:
                if not self.api_config.require_https:
                    raise ConfigurationError("HTTPS must be required in production")
                if not self.model_config.require_encryption:
                    raise ConfigurationError(
                        "Model API encryption must be enabled in production"
                    )

                # Version compatibility check
                versions = {name: cfg.version for name, cfg in self._configs.items()}
                if len(set(versions.values())) > 1:
                    raise ConfigurationError(
                        f"Version mismatch in production configurations: {versions}"
                    )

    def get_config(self, config_class: type[T]) -> T:
        """Get configuration instance by class."""
        with self._init_lock:
            config_name = config_class.__name__.lower()
            config = self._configs.get(config_name)
            if not config:
                config = self._load_or_create(config_class)
            return config

    @lru_cache
    def get_merged_config(self, *config_classes: type[BaseConfig]) -> dict:
        """Get merged configuration dictionary from multiple configs."""
        with self._init_lock:
            merged = {}
            for config_class in config_classes:
                config = self.get_config(config_class)
                merged.update(config.model_dump())
            return merged

    def reload_config(self, config_class: type[T]) -> T:
        """Reload specific configuration."""
        with self._init_lock:
            config_name = config_class.__name__.lower()
            self._configs.pop(config_name, None)
            return self._load_or_create(config_class)

    def update_config(self, config_class: type[T], **updates) -> T:
        """Update configuration with new values."""
        with self._init_lock:
            config = self.get_config(config_class)

            # Prevent modifications in production unless explicitly allowed
            if self.is_production:
                for key in updates:
                    if not hasattr(config, f"allow_{key}_update"):
                        raise ConfigurationError(
                            f"Cannot modify {key} in production environment"
                        )

            for key, value in updates.items():
                setattr(config, key, value)
            return config

    def export_config(
        self, config_class: type[T], file_path: str | Path, format: str = "yaml"
    ) -> None:
        """Export configuration to file."""
        with self._init_lock:
            config = self.get_config(config_class)
            config.save_to_file(file_path, format=format)

    def import_config(
        self, config_class: type[T], file_path: str | Path, format: str = None
    ) -> T:
        """Import configuration from file."""
        with self._init_lock:
            config = config_class.load_from_file(file_path)
            config_name = config_class.__name__.lower()
            self._configs[config_name] = config
            return config

    @classmethod
    def get_instance(
        cls,
        config_dir: str | Path | None = None,
        environment: str | None = None,
        encryption_key: str | None = None,
    ) -> "ConfigurationManager":
        """Get or create singleton configuration manager instance."""
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = cls(
                        config_dir=config_dir,
                        environment=environment,
                        encryption_key=encryption_key,
                    )
        return cls._instance

    def __getattr__(self, name: str) -> Any:
        """Allow direct access to configuration instances."""
        for config in self._configs.values():
            if hasattr(config, name):
                return getattr(config, name)
        raise AttributeError(f"No configuration found with attribute {name}")
