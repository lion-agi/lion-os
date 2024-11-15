import json
import os
import tempfile
from pathlib import Path

import pytest
import yaml

from lion.core.config.base import ConfigurationError
from lion.core.config.model import ModelConfig


class TestModelConfig:
    """Test suite for ModelConfig."""

    @pytest.fixture
    def model_config(self):
        """Create a test model configuration instance."""
        os.environ["TEST_API_KEY"] = "test_key_12345678901234567890123456789012"
        return ModelConfig(
            provider="openai",
            model_name="gpt-4",
            api_key_env_var="TEST_API_KEY",
            environment="development",
        )

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_initialization(self, model_config):
        """Test basic model configuration initialization."""
        assert model_config.provider == "openai"
        assert model_config.model_name == "gpt-4"
        assert model_config.api_key_env_var == "TEST_API_KEY"
        assert model_config.temperature == 0.7
        assert model_config.environment == "development"

    def test_provider_validation(self):
        """Test provider validation."""
        # Valid providers
        valid_providers = ["openai", "anthropic", "local", "azure", "cohere"]
        for provider in valid_providers:
            config = ModelConfig(
                provider=provider,
                model_name="test-model",
                api_key_env_var="TEST_API_KEY",
            )
            assert config.provider == provider.lower()

        # Invalid provider
        with pytest.raises(ValueError):
            ModelConfig(
                provider="invalid_provider",
                model_name="test-model",
                api_key_env_var="TEST_API_KEY",
            )

    def test_api_key_validation(self, model_config):
        """Test API key environment variable validation."""
        # Valid API key
        assert model_config.get_api_key() == "test_key_12345678901234567890123456789012"

        # Missing API key
        with pytest.raises(ConfigurationError):
            ModelConfig(
                provider="openai", model_name="gpt-4", api_key_env_var="NONEXISTENT_KEY"
            )

    def test_temperature_validation(self, model_config):
        """Test temperature validation."""
        # Valid temperatures
        valid_temps = [0.0, 0.5, 1.0, 1.5, 2.0]
        for temp in valid_temps:
            model_config.temperature = temp
            assert model_config.temperature == temp

        # Invalid temperatures
        invalid_temps = [-0.1, 2.1]
        for temp in invalid_temps:
            with pytest.raises(ValueError):
                model_config.temperature = temp

    def test_fallback_models(self, model_config):
        """Test fallback model configuration."""
        # Set allowed models
        model_config.allowed_models = {"gpt-4", "gpt-3.5-turbo"}

        # Valid fallback configuration
        model_config.fallback_models = {"rate_limit": "gpt-3.5-turbo"}
        assert "rate_limit" in model_config.fallback_models

        # Invalid fallback model
        with pytest.raises(ValueError):
            model_config.fallback_models = {"rate_limit": "nonexistent-model"}

    def test_environment_loading(self):
        """Test loading from environment variables."""
        # Setup environment variables
        os.environ["LION_MODEL_PROVIDER"] = "openai"
        os.environ["LION_MODEL_MODEL_NAME"] = "gpt-4"
        os.environ["LION_MODEL_API_KEY_VAR"] = "TEST_API_KEY"
        os.environ["LION_MODEL_TEMPERATURE"] = "0.8"

        config = ModelConfig.from_environment()
        assert config.provider == "openai"
        assert config.model_name == "gpt-4"
        assert config.temperature == 0.8

    def test_file_operations(self, model_config, temp_dir):
        """Test configuration file operations."""
        # JSON format
        json_path = temp_dir / "model_config.json"
        model_config.save_to_file(json_path)
        loaded_json = ModelConfig.from_file(json_path)
        assert loaded_json.provider == model_config.provider
        assert loaded_json.model_name == model_config.model_name

        # YAML format
        yaml_path = temp_dir / "model_config.yaml"
        model_config.save_to_file(yaml_path)
        loaded_yaml = ModelConfig.from_file(yaml_path)
        assert loaded_yaml.provider == model_config.provider
        assert loaded_yaml.model_name == model_config.model_name

    def test_secure_api_key_handling(self, model_config):
        """Test secure API key handling."""
        model_config.configure_encryption()
        model_config.require_encryption = True

        # Get API key should return decrypted value
        api_key = model_config.get_api_key()
        assert api_key == "test_key_12345678901234567890123456789012"

        # API key should not appear in string representation
        assert api_key not in str(model_config)

    def test_production_security(self, model_config):
        """Test production environment security requirements."""
        model_config.environment = "production"

        # Production should require encryption
        model_config.require_encryption = False
        with pytest.raises(ConfigurationError):
            model_config.validate_security()

        # Production should require allowed models
        model_config.require_encryption = True
        model_config.allowed_models = set()
        with pytest.raises(ConfigurationError):
            model_config.validate_security()

        # Production should limit temperature
        model_config.allowed_models = {"gpt-4"}
        model_config.temperature = 1.5
        with pytest.raises(ConfigurationError):
            model_config.validate_security()

    def test_model_config_generation(self, model_config):
        """Test model configuration dictionary generation."""
        model_config.stop_sequences = {"stop1", "stop2"}
        model_config.response_format = "json"

        config_dict = model_config.get_model_config()
        assert config_dict["model"] == "gpt-4"
        assert config_dict["temperature"] == 0.7
        assert set(config_dict["stop_sequences"]) == {"stop1", "stop2"}
        assert config_dict["response_format"] == "json"

    def test_rate_limiting(self, model_config):
        """Test rate limiting configuration."""
        model_config.requests_per_minute = 60
        model_config.tokens_per_minute = 40000

        assert model_config.requests_per_minute == 60
        assert model_config.tokens_per_minute == 40000

    def test_context_window(self, model_config):
        """Test context window configuration."""
        model_config.context_window = 8192
        assert model_config.context_window == 8192

        # Optional - can be None
        model_config.context_window = None
        assert model_config.context_window is None

    def test_encryption_requirements(self, model_config):
        """Test encryption requirements in different environments."""
        # Development - encryption optional
        model_config.environment = "development"
        model_config.require_encryption = False
        model_config.validate_security()  # Should not raise

        # Production - encryption required
        model_config.environment = "production"
        with pytest.raises(ConfigurationError):
            model_config.validate_security()

        model_config.require_encryption = True
        model_config.allowed_models = {"gpt-4"}
        model_config.temperature = 0.7
        model_config.validate_security()  # Should not raise
