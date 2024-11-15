import concurrent.futures
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

import pytest

from lion.core.config.base import (
    BaseConfig,
    ConfigurationError,
    EnvironmentError,
    ValidationError,
)


class SecureTestConfig(BaseConfig):
    """Test configuration with secure and mutable fields."""

    api_key: str = None
    rate_limit: int = 100
    debug_mode: bool = False

    def __init__(self, **data):
        super().__init__(**data)
        self.mark_field_secure("api_key")
        self.mark_field_mutable("rate_limit")  # Allows modification in production


class TestAdvancedBaseConfig:
    """Advanced test suite for BaseConfig focusing on thread-safety and security."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def secure_config(self):
        """Create a secure test configuration instance."""
        config = SecureTestConfig(api_key="test_key_12345", environment="development")
        config.configure_encryption("test_encryption_key_12345")
        return config

    def test_thread_safe_field_access(self, secure_config):
        """Test thread-safe field access and modification."""

        def modify_config():
            for i in range(10):
                secure_config.rate_limit = i
                time.sleep(0.01)  # Simulate work
                assert secure_config.rate_limit == i

        threads = [threading.Thread(target=modify_config) for _ in range(5)]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    def test_concurrent_secure_field_access(self, secure_config):
        """Test concurrent access to secure fields."""
        results = []

        def access_secure_field():
            try:
                value = secure_config.get_secure_field("api_key")
                results.append(value)
                return True
            except Exception:
                return False

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(access_secure_field) for _ in range(10)]
            outcomes = [f.result() for f in futures]

        assert all(outcomes)
        assert all(r == "test_key_12345" for r in results)

    def test_production_field_immutability(self):
        """Test field immutability in production environment."""
        config = SecureTestConfig(api_key="prod_key_12345", environment="production")
        config.configure_encryption("test_encryption_key_12345")

        # Should allow modification of explicitly mutable fields
        config.rate_limit = 200
        assert config.rate_limit == 200

        # Should prevent modification of non-mutable fields
        with pytest.raises(ConfigurationError):
            config.debug_mode = True

    def test_secure_field_encryption(self, secure_config):
        """Test secure field encryption consistency."""
        original_value = "test_key_12345"
        encrypted_value = getattr(secure_config, "api_key")
        decrypted_value = secure_config.get_secure_field("api_key")

        assert encrypted_value != original_value
        assert decrypted_value == original_value

        # Test encryption is consistent
        secure_config.api_key = original_value
        new_encrypted = getattr(secure_config, "api_key")
        assert new_encrypted != original_value
        assert secure_config.get_secure_field("api_key") == original_value

    def test_concurrent_file_operations(self, secure_config, temp_dir):
        """Test concurrent file operations."""

        def save_and_load():
            path = temp_dir / f"config_{time.time()}.yaml"
            secure_config.save_to_file(path)
            loaded = SecureTestConfig.load_from_file(path)
            assert loaded.rate_limit == secure_config.rate_limit
            return True

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(lambda _: save_and_load(), range(10)))

        assert all(results)

    def test_version_validation(self):
        """Test version validation and compatibility."""
        config1 = SecureTestConfig(version="2.0.0")
        config2 = SecureTestConfig(version="1.0.0")

        # Version comparison
        assert config1.version > config2.version

        # Version format validation
        with pytest.raises(ValueError):
            SecureTestConfig(version="invalid")

    def test_environment_specific_behavior(self):
        """Test environment-specific configuration behavior."""
        # Development environment
        dev_config = SecureTestConfig(environment="development")
        dev_config.debug_mode = True  # Should work

        # Production environment
        prod_config = SecureTestConfig(environment="production")
        with pytest.raises(ConfigurationError):
            prod_config.debug_mode = True  # Should fail

        # Staging environment
        staging_config = SecureTestConfig(environment="staging")
        staging_config.debug_mode = True  # Should work

    def test_secure_field_management(self, secure_config):
        """Test secure field management."""
        # Add new secure field
        secure_config.mark_field_secure("rate_limit")
        secure_config.rate_limit = 150

        # Verify encryption
        encrypted_value = getattr(secure_config, "rate_limit")
        assert str(150) != encrypted_value
        assert secure_config.get_secure_field("rate_limit") == "150"

        # Remove secure field marking
        secure_config._secure_fields.remove("rate_limit")
        secure_config.rate_limit = 200
        assert secure_config.rate_limit == 200

    def test_mutable_field_management(self):
        """Test mutable field management in production."""
        config = SecureTestConfig(environment="production")

        # Test explicitly mutable field
        assert "rate_limit" in config._mutable_fields
        config.rate_limit = 300  # Should work

        # Test making field mutable
        config.mark_field_mutable("debug_mode")
        config.debug_mode = True  # Should now work

        # Test invalid field
        with pytest.raises(ValueError):
            config.mark_field_mutable("nonexistent_field")

    def test_concurrent_validation(self, secure_config):
        """Test concurrent validation operations."""

        def validate():
            try:
                secure_config.validate_security()
                return True
            except Exception:
                return False

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(lambda _: validate(), range(10)))

        assert all(results)

    def test_configuration_timestamp(self):
        """Test configuration timestamp handling."""
        config = SecureTestConfig()
        assert isinstance(config.created_at, datetime)

        # Test timestamp immutability
        with pytest.raises(Exception):
            config.created_at = datetime.utcnow()

    def test_merge_restrictions(self):
        """Test merge restrictions in production environment."""
        dev_config = SecureTestConfig(environment="development")
        prod_config = SecureTestConfig(environment="production")

        # Development merging should work
        dev_config.merge(SecureTestConfig(rate_limit=500))
        assert dev_config.rate_limit == 500

        # Production merging should fail
        with pytest.raises(ConfigurationError):
            prod_config.merge(SecureTestConfig(rate_limit=500))
