import json
import os
import tempfile
from pathlib import Path

import pytest
import yaml

from lion.core.config.api import APIConfig
from lion.core.config.base import ConfigurationError
from lion.core.config.manager import ConfigurationManager
from lion.core.config.model import ModelConfig


class TestConfigurationManager:
    """Test suite for ConfigurationManager."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def env_setup(self):
        """Setup environment variables for testing."""
        os.environ["TEST_API_KEY"] = "test_key_12345678901234567890123456789012"
        os.environ["LION_API_KEY"] = "api_key_12345678901234567890123456789012"
        os.environ["LION_MODEL_PROVIDER"] = "openai"
        os.environ["LION_MODEL_MODEL_NAME"] = "gpt-4"
        os.environ["LION_MODEL_API_KEY_VAR"] = "TEST_API_KEY"
        yield
        # Cleanup
        del os.environ["TEST_API_KEY"]
        del os.environ["LION_API_KEY"]
        del os.environ["LION_MODEL_PROVIDER"]
        del os.environ["LION_MODEL_MODEL_NAME"]
        del os.environ["LION_MODEL_API_KEY_VAR"]

    @pytest.fixture
    def config_manager(self, temp_dir, env_setup):
        """Create a test configuration manager instance."""
        return ConfigurationManager(
            config_dir=temp_dir,
            environment="development",
            encryption_key="test_encryption_key_12345",
        )

    def test_initialization(self, config_manager):
        """Test configuration manager initialization."""
        assert config_manager.environment == "development"
        assert config_manager._encryption_key is not None
        assert isinstance(config_manager.api_config, APIConfig)
        assert isinstance(config_manager.model_config, ModelConfig)

    def test_environment_properties(self, config_manager):
        """Test environment helper properties."""
        assert config_manager.is_development
        assert not config_manager.is_production

        config_manager.environment = "production"
        assert config_manager.is_production
        assert not config_manager.is_development

    def test_config_loading(self, config_manager, temp_dir):
        """Test configuration loading from files."""
        # Save configurations
        config_manager.save_all()

        # Create new manager instance
        new_manager = ConfigurationManager(
            config_dir=temp_dir, environment="development"
        )

        # Verify configurations loaded correctly
        assert isinstance(new_manager.api_config, APIConfig)
        assert isinstance(new_manager.model_config, ModelConfig)

    def test_config_validation(self, config_manager):
        """Test configuration validation."""
        # Development environment - should pass
        config_manager.validate_all()

        # Production environment - should enforce stricter rules
        config_manager.environment = "production"
        config_manager.api_config.require_https = False

        with pytest.raises(ConfigurationError):
            config_manager.validate_all()

    def test_config_reloading(self, config_manager):
        """Test configuration reloading."""
        original_api_key = config_manager.api_config.api_keys.copy()

        # Modify environment
        os.environ["LION_API_KEY"] = "new_key_12345678901234567890123456789012"

        # Reload configuration
        new_api_config = config_manager.reload_config(APIConfig)
        assert "new_key_12345678901234567890123456789012" in new_api_config.api_keys
        assert original_api_key != new_api_config.api_keys

    def test_config_updating(self, config_manager):
        """Test configuration updating."""
        # Update API configuration
        config_manager.update_config(
            APIConfig, rate_limit_requests=200, rate_limit_window=120
        )

        assert config_manager.api_config.rate_limit_requests == 200
        assert config_manager.api_config.rate_limit_window == 120

    def test_merged_config(self, config_manager):
        """Test merged configuration generation."""
        merged = config_manager.get_merged_config(APIConfig, ModelConfig)

        # Verify merged configuration contains both API and Model settings
        assert "rate_limit_enabled" in merged  # API config
        assert "provider" in merged  # Model config

    def test_config_export_import(self, config_manager, temp_dir):
        """Test configuration export and import."""
        # Export configuration
        export_path = temp_dir / "exported_config.yaml"
        config_manager.export_config(APIConfig, export_path)

        # Import configuration
        imported_config = config_manager.import_config(APIConfig, export_path)
        assert isinstance(imported_config, APIConfig)
        assert (
            imported_config.rate_limit_enabled
            == config_manager.api_config.rate_limit_enabled
        )

    def test_singleton_behavior(self, temp_dir, env_setup):
        """Test configuration manager singleton behavior."""
        manager1 = ConfigurationManager.get_instance(
            config_dir=temp_dir, environment="development"
        )

        manager2 = ConfigurationManager.get_instance()

        assert manager1 is manager2
        assert manager1.config_dir == manager2.config_dir

    def test_encryption_handling(self, config_manager):
        """Test encryption handling across configurations."""
        # Verify encryption key propagation
        assert config_manager.api_config._encryption_key is not None
        assert config_manager.model_config._encryption_key is not None

        # Test secure field handling
        secure_value = "secure_test_value"
        encrypted = config_manager.api_config._encrypt_value(secure_value)
        decrypted = config_manager.api_config._decrypt_value(encrypted)
        assert decrypted == secure_value

    def test_environment_specific_loading(self, temp_dir):
        """Test environment-specific configuration loading."""
        # Create development manager
        dev_manager = ConfigurationManager(
            config_dir=temp_dir, environment="development"
        )
        dev_manager.save_all()

        # Create production manager
        prod_manager = ConfigurationManager(
            config_dir=temp_dir, environment="production"
        )
        prod_manager.save_all()

        # Verify separate configurations
        assert (temp_dir / "apiconfig.development.yaml").exists()
        assert (temp_dir / "apiconfig.production.yaml").exists()

    def test_error_handling(self, temp_dir):
        """Test error handling in configuration manager."""
        # Invalid environment
        with pytest.raises(ValueError):
            ConfigurationManager(config_dir=temp_dir, environment="invalid")

        # Invalid configuration loading
        manager = ConfigurationManager(config_dir=temp_dir)
        with pytest.raises(Exception):
            manager.import_config(APIConfig, temp_dir / "nonexistent.yaml")

    def test_attribute_access(self, config_manager):
        """Test attribute access behavior."""
        # Direct access to configuration attributes
        assert config_manager.api_config.rate_limit_enabled
        assert config_manager.model_config.provider == "openai"

        # Access to non-existent attributes
        with pytest.raises(AttributeError):
            config_manager.nonexistent_attribute

    def test_production_constraints(self, temp_dir):
        """Test production environment constraints."""
        manager = ConfigurationManager(
            config_dir=temp_dir,
            environment="production",
            encryption_key="test_encryption_key_12345",
        )

        # Verify production security settings
        assert manager.api_config.require_https
        assert manager.model_config.require_encryption
        assert not manager.api_config.allow_dev_keys
