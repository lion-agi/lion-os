import json
import os
import tempfile
from pathlib import Path

import pytest
import yaml
from cryptography.fernet import Fernet

from lion.core.config.base import BaseConfig, ConfigurationError


# Test configuration class defined outside test class
class SimpleTestConfig(BaseConfig):
    """Simple test configuration class."""

    test_field: str
    optional_field: str = "default"
    secure_field: str = None


class TestBaseConfig:
    """Test suite for BaseConfig."""

    @pytest.fixture
    def config(self):
        """Create a test configuration instance."""
        return SimpleTestConfig(test_field="test", environment="development")

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_initialization(self, config):
        """Test basic configuration initialization."""
        assert config.test_field == "test"
        assert config.optional_field == "default"
        assert config.environment == "development"
        assert config.version == "1.0.0"

    def test_environment_validation(self):
        """Test environment validation."""
        # Valid environments
        for env in ["development", "staging", "production"]:
            config = SimpleTestConfig(test_field="test", environment=env)
            assert config.environment == env

        # Invalid environment
        with pytest.raises(ValueError):
            SimpleTestConfig(test_field="test", environment="invalid")

    def test_encryption(self, config):
        """Test encryption functionality."""
        # Configure encryption
        config.configure_encryption()
        assert config._encryption_key is not None

        # Test encryption/decryption
        test_value = "sensitive_data"
        encrypted = config._encrypt_value(test_value)
        decrypted = config._decrypt_value(encrypted)
        assert decrypted == test_value
        assert encrypted != test_value

    def test_secure_field_handling(self, config):
        """Test secure field handling."""
        config.configure_encryption()

        # Set secure field
        secure_value = "secret123"
        config.secure_field = secure_value
        config._secure_fields.add("secure_field")

        # Get secure field
        retrieved = config.get_secure_field("secure_field")
        assert retrieved == secure_value

        # Invalid secure field
        with pytest.raises(ValueError):
            config.get_secure_field("nonexistent_field")

    def test_file_operations(self, config, temp_dir):
        """Test file operations."""
        # JSON
        json_path = temp_dir / "config.json"
        config.save_to_file(json_path)
        loaded_config = SimpleTestConfig.load_from_file(json_path)
        assert loaded_config.test_field == config.test_field

        # YAML
        yaml_path = temp_dir / "config.yaml"
        config.save_to_file(yaml_path)
        loaded_config = SimpleTestConfig.load_from_file(yaml_path)
        assert loaded_config.test_field == config.test_field

        # Invalid format
        with pytest.raises(ValueError):
            config.save_to_file(temp_dir / "config.invalid")

    def test_environment_loading(self):
        """Test loading from environment variables."""
        os.environ["TEST_FIELD"] = "env_value"
        os.environ["OPTIONAL_FIELD"] = "env_optional"

        config = SimpleTestConfig.load_from_env(prefix="")
        assert config.test_field == "env_value"
        assert config.optional_field == "env_optional"

    def test_validation_security(self, config):
        """Test security validation."""
        # Development environment
        config.environment = "development"
        config.validate_security()  # Should not raise

        # Production environment
        config.environment = "production"
        config.validate_security()  # Base implementation should not raise

    def test_hash_generation(self, config):
        """Test configuration hash generation."""
        hash1 = config.get_hash()
        assert isinstance(hash1, str)

        # Modify config
        config.test_field = "modified"
        hash2 = config.get_hash()
        assert hash1 != hash2

    def test_merge_configurations(self, config):
        """Test configuration merging."""
        other_config = SimpleTestConfig(test_field="other", optional_field="merged")

        config.merge(other_config)
        assert config.test_field == "other"
        assert config.optional_field == "merged"

    def test_environment_properties(self, config):
        """Test environment helper properties."""
        config.environment = "development"
        assert config.is_development is True
        assert config.is_production is False

        config.environment = "production"
        assert config.is_development is False
        assert config.is_production is True

    def test_string_representation(self, config):
        """Test string representation."""
        str_rep = str(config)
        assert "SimpleTestConfig" in str_rep
        assert "test_field" in str_rep

        # Secure fields should not be in string representation
        config._secure_fields.add("secure_field")
        config.secure_field = "secret"
        str_rep = str(config)
        assert "secret" not in str_rep

    @pytest.mark.parametrize("env", ["development", "staging", "production"])
    def test_environment_specific_behavior(self, env):
        """Test environment-specific behavior."""
        config = SimpleTestConfig(test_field="test", environment=env)

        if env == "production":
            assert config.is_production
            # Add production-specific assertions
        else:
            assert not config.is_production
            # Add non-production assertions

    def test_error_handling(self):
        """Test error handling."""
        # Missing required field
        with pytest.raises(Exception):
            SimpleTestConfig()

        # Invalid encryption operation
        config = SimpleTestConfig(test_field="test")
        with pytest.raises(ConfigurationError):
            config._encrypt_value("test")  # No encryption key configured

    def test_configuration_immutability(self):
        """Test configuration immutability in production."""
        config = SimpleTestConfig(test_field="test", environment="production")

        # Production configurations should be more restrictive
        assert config.is_production
        # Add assertions for production-specific restrictions
