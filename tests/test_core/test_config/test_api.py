import os
import tempfile
from pathlib import Path

import pytest

from lion.core.config.api import APIConfig
from lion.core.config.base import ConfigurationError


class TestAPIConfig:
    """Test suite for APIConfig."""

    @pytest.fixture
    def api_config(self):
        """Create a test API configuration instance."""
        return APIConfig(
            api_keys={"test_key_12345678901234567890123456789012"},
            environment="development",
        )

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_initialization(self, api_config):
        """Test basic API configuration initialization."""
        assert len(api_config.api_keys) == 1
        assert api_config.min_key_length == 32
        assert api_config.rate_limit_enabled is True
        assert api_config.environment == "development"

    def test_api_key_validation(self):
        """Test API key validation."""
        # Valid key
        valid_key = "test_key_12345678901234567890123456789012"
        config = APIConfig(api_keys={valid_key})
        assert valid_key in config.api_keys

        # Invalid key (too short)
        with pytest.raises(ValueError):
            APIConfig(api_keys={"short_key"})

        # Invalid key (invalid characters)
        with pytest.raises(ValueError):
            APIConfig(api_keys={"invalid!@#$%^&*()"})

    def test_cors_validation(self):
        """Test CORS origin validation."""
        # Development environment
        config = APIConfig(
            api_keys={"test_key_12345678901234567890123456789012"},
            environment="development",
            cors_origins={"*"},
        )
        assert "*" in config.cors_origins

        # Production environment
        with pytest.raises(ValueError):
            APIConfig(
                api_keys={"test_key_12345678901234567890123456789012"},
                environment="production",
                cors_origins={"*"},
            )

        # Valid production CORS
        config = APIConfig(
            api_keys={"test_key_12345678901234567890123456789012"},
            environment="production",
            cors_origins={"https://api.example.com"},
        )
        assert "https://api.example.com" in config.cors_origins

    def test_rate_limiting(self, api_config):
        """Test rate limiting configuration."""
        api_config.rate_limit_enabled = True
        api_config.rate_limit_requests = 100
        api_config.rate_limit_window = 60

        assert api_config.rate_limit_enabled
        assert api_config.rate_limit_requests == 100
        assert api_config.rate_limit_window == 60

    def test_environment_loading(self):
        """Test loading from environment variables."""
        # Setup environment variables
        os.environ["LION_API_KEY"] = "test_key_12345678901234567890123456789012"
        os.environ["LION_ENV"] = "development"
        os.environ["LION_RATE_LIMIT_ENABLED"] = "1"
        os.environ["LION_CORS_ORIGINS"] = "https://api1.com,https://api2.com"

        config = APIConfig.from_environment()
        assert "test_key_12345678901234567890123456789012" in config.api_keys
        assert config.rate_limit_enabled is True
        assert len(config.cors_origins) == 2

    def test_security_validation(self, api_config):
        """Test security validation in different environments."""
        # Development environment
        api_config.environment = "development"
        api_config.allow_dev_keys = True
        api_config.require_https = False
        api_config.validate_security()  # Should not raise

        # Production environment
        api_config.environment = "production"
        api_config.allow_dev_keys = True  # Should trigger error
        api_config.require_https = False  # Should trigger error

        with pytest.raises(ConfigurationError):
            api_config.validate_security()

    def test_api_key_management(self, api_config):
        """Test API key management functions."""
        new_key = "new_key_12345678901234567890123456789012"

        # Add key
        api_config.add_api_key(new_key)
        assert new_key in api_config.api_keys

        # Remove key
        api_config.remove_api_key(new_key)
        assert new_key not in api_config.api_keys

        # Invalid key format
        with pytest.raises(ValueError):
            api_config.add_api_key("invalid_key")

    def test_cors_management(self, api_config):
        """Test CORS origin management."""
        # Development environment
        assert api_config.is_cors_allowed("http://localhost:3000")

        # Production environment
        api_config.environment = "production"
        api_config.cors_origins = {"https://api.example.com"}

        assert api_config.is_cors_allowed("https://api.example.com")
        assert not api_config.is_cors_allowed("http://invalid.com")

    def test_file_operations(self, api_config, temp_dir):
        """Test configuration file operations."""
        # Save configuration
        config_path = temp_dir / "api_config.yaml"
        api_config.save_to_file(config_path)

        # Load configuration
        loaded_config = APIConfig.from_file(config_path)
        assert loaded_config.api_keys == api_config.api_keys
        assert loaded_config.rate_limit_enabled == api_config.rate_limit_enabled

    def test_production_constraints(self):
        """Test production environment constraints."""
        config = APIConfig(
            api_keys={"test_key_12345678901234567890123456789012"},
            environment="production",
            require_https=True,
            allow_dev_keys=False,
            cors_origins={"https://api.example.com"},
        )

        # Verify production settings
        assert config.require_https
        assert not config.allow_dev_keys
        assert all(origin.startswith("https://") for origin in config.cors_origins)

    def test_rate_limit_validation(self, api_config):
        """Test rate limit validation."""
        # Valid rate limits
        api_config.rate_limit_requests = 100
        api_config.rate_limit_window = 60
        assert api_config.rate_limit_requests == 100
        assert api_config.rate_limit_window == 60

        # Invalid rate limits
        with pytest.raises(ValueError):
            api_config.rate_limit_requests = -1

        with pytest.raises(ValueError):
            api_config.rate_limit_window = 0

    def test_secure_key_handling(self, api_config):
        """Test secure key handling."""
        api_config.configure_encryption()

        # Add a secure key
        secure_key = "secure_key_12345678901234567890123456789012"
        encrypted_key = api_config._encrypt_value(secure_key)
        api_config.api_keys.add(encrypted_key)

        # Verify key is stored securely
        assert secure_key not in str(api_config)
        assert encrypted_key in api_config.api_keys
