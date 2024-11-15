import concurrent.futures
import tempfile
import threading
import time
from pathlib import Path
from typing import List, Set

import pytest

from lion.core.config.api import APIConfig
from lion.core.config.base import ConfigurationError
from lion.core.config.branch_config import BranchConfig
from lion.core.config.id_config import LionIDConfig
from lion.core.config.imodel_config import iModelConfig
from lion.core.config.log_config import LogConfig
from lion.core.config.manager import ConfigurationManager
from lion.core.config.model import ModelConfig
from lion.core.config.retry_config import RetryConfig


class TestAdvancedConfigurationManager:
    """Advanced test suite for ConfigurationManager focusing on thread-safety and global settings."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_thread_safe_singleton(self, temp_dir):
        """Test thread-safety of the singleton pattern."""
        instance_ids: set[int] = set()
        error_occurred = False

        def get_instance():
            nonlocal error_occurred
            try:
                instance = ConfigurationManager.get_instance(
                    config_dir=temp_dir, environment="development"
                )
                instance_ids.add(id(instance))
            except Exception:
                error_occurred = True

        # Create multiple threads to access the singleton simultaneously
        threads = [threading.Thread(target=get_instance) for _ in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # Verify only one instance was created
        assert len(instance_ids) == 1
        assert not error_occurred

    def test_concurrent_config_access(self, temp_dir):
        """Test concurrent access to configuration settings."""
        manager = ConfigurationManager.get_instance(
            config_dir=temp_dir, environment="development"
        )

        def update_and_read():
            # Update configuration
            manager.update_config(
                APIConfig, rate_limit_requests=200, rate_limit_window=120
            )
            # Read configuration
            return manager.api_config.rate_limit_requests

        # Execute concurrent updates and reads
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(lambda _: update_and_read(), range(10)))

        # Verify all threads see the same configuration
        assert all(result == 200 for result in results)

    def test_config_inheritance(self, temp_dir):
        """Test configuration inheritance and overrides."""

        class CustomAPIConfig(APIConfig):
            custom_field: str = "default"

        manager = ConfigurationManager(config_dir=temp_dir, environment="development")

        # Test that custom config inherits base settings
        custom_config = manager._load_or_create(CustomAPIConfig)
        assert hasattr(custom_config, "rate_limit_enabled")
        assert hasattr(custom_config, "custom_field")

    def test_production_security_enhanced(self, temp_dir):
        """Test enhanced production environment security restrictions."""
        manager = ConfigurationManager(
            config_dir=temp_dir,
            environment="production",
            encryption_key="test_encryption_key_12345",
        )

        # Test strict production settings
        assert manager.api_config.require_https
        assert manager.model_config.require_encryption
        assert not manager.api_config.allow_dev_keys

        # Test immutability in production
        with pytest.raises(ConfigurationError):
            manager.api_config.require_https = False

        with pytest.raises(ConfigurationError):
            manager.model_config.require_encryption = False

    def test_global_settings_propagation(self, temp_dir):
        """Test propagation of global settings across different config types."""
        manager = ConfigurationManager(
            config_dir=temp_dir,
            environment="production",
            encryption_key="test_encryption_key_12345",
        )

        # Load all config types
        configs = [
            manager._load_or_create(config_class)
            for config_class in [
                APIConfig,
                ModelConfig,
                BranchConfig,
                LionIDConfig,
                iModelConfig,
                LogConfig,
                RetryConfig,
            ]
        ]

        # Verify global settings are consistent
        for config in configs:
            assert config.environment == "production"
            assert config._encryption_key is not None
            assert hasattr(config, "version")

    def test_config_validation_chain(self, temp_dir):
        """Test validation chain across dependent configurations."""
        manager = ConfigurationManager(config_dir=temp_dir, environment="production")

        # Set up dependent configurations
        manager.update_config(APIConfig, api_version="2.0")
        manager.update_config(ModelConfig, min_api_version="2.0")

        # Test validation chain
        manager.validate_all()  # Should pass

        # Test validation failure on version mismatch
        manager.update_config(APIConfig, api_version="1.0")
        with pytest.raises(ConfigurationError):
            manager.validate_all()

    def test_config_hot_reload(self, temp_dir):
        """Test hot reloading of configuration changes."""
        manager = ConfigurationManager(config_dir=temp_dir, environment="development")

        # Initial save
        manager.save_all()

        # Modify file directly
        api_config_path = temp_dir / "apiconfig.development.yaml"
        with api_config_path.open("r") as f:
            content = f.read()
        with api_config_path.open("w") as f:
            f.write(
                content.replace("rate_limit_requests: 100", "rate_limit_requests: 500")
            )

        # Reload and verify
        reloaded_config = manager.reload_config(APIConfig)
        assert reloaded_config.rate_limit_requests == 500

    def test_concurrent_validation(self, temp_dir):
        """Test concurrent validation of configurations."""
        manager = ConfigurationManager(config_dir=temp_dir, environment="development")

        def validate_configs():
            try:
                manager.validate_all()
                return True
            except Exception:
                return False

        # Run concurrent validations
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(lambda _: validate_configs(), range(10)))

        # All validations should succeed
        assert all(results)

    def test_config_version_compatibility(self, temp_dir):
        """Test configuration version compatibility checks."""
        manager = ConfigurationManager(config_dir=temp_dir, environment="development")

        # Test version compatibility validation
        manager.update_config(APIConfig, version="2.0.0")
        manager.update_config(ModelConfig, version="1.0.0")

        with pytest.raises(ConfigurationError):
            manager.validate_all()  # Should fail due to version mismatch

        # Fix version compatibility
        manager.update_config(ModelConfig, version="2.0.0")
        manager.validate_all()  # Should pass
