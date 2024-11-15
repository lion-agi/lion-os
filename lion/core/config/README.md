# Lion Configuration System

A secure, modular, and environment-aware configuration system for the Lion framework.

## Overview

The Lion configuration system provides a robust, secure, and flexible way to manage configurations across different environments. It features:

- Secure handling of sensitive data
- Environment-specific configurations
- Encryption support for sensitive values
- Validation and type checking
- Centralized configuration management
- Backward compatibility with existing systems

## Components

### BaseConfig

The foundation of the configuration system, providing:
- Secure field handling
- Environment awareness
- Configuration validation
- File operations (JSON/YAML)
- Encryption capabilities

```python
from lion.core.config import BaseConfig

class MyConfig(BaseConfig):
    field1: str
    field2: int = 42
```

### APIConfig

Manages API-related configurations:
- API key management
- Rate limiting
- CORS settings
- Security policies

```python
from lion.core.config import APIConfig

api_config = APIConfig.from_environment()
api_config.validate_security()
```

### ModelConfig

Handles AI model configurations:
- Provider settings
- Model parameters
- API authentication
- Rate limiting
- Fallback configurations

```python
from lion.core.config import ModelConfig

model_config = ModelConfig(
    provider="openai",
    model_name="gpt-4",
    api_key_env_var="OPENAI_API_KEY"
)
```

### ConfigurationManager

Central manager for all configurations:
- Singleton pattern
- Environment management
- Configuration loading/saving
- Cross-config validation

```python
from lion.core.config import ConfigurationManager

manager = ConfigurationManager.get_instance(
    config_dir="config",
    environment="development"
)
```

## Usage

### Basic Usage

```python
from lion.settings import config_manager

# Access configurations
api_config = config_manager.api_config
model_config = config_manager.model_config

# Use configurations
if config_manager.is_production:
    # Production-specific logic
    api_config.require_https = True
    model_config.require_encryption = True
```

### Environment Variables

Required environment variables:
- `LION_ENV`: Environment name (development/staging/production)
- `LION_API_KEY`: API key for authentication
- `LION_CONFIG_DIR`: Configuration directory (optional)
- `LION_ENCRYPTION_KEY`: Key for encrypting sensitive data (optional)

### Security Features

1. Encryption:
```python
config = MyConfig(sensitive_field="secret")
config.configure_encryption(key="encryption_key")
config._secure_fields.add("sensitive_field")
```

2. Environment-specific validation:
```python
config.environment = "production"
config.validate_security()  # Enforces stricter rules
```

### File Operations

1. Save configuration:
```python
config.save_to_file("config.yaml")
```

2. Load configuration:
```python
config = MyConfig.from_file("config.yaml")
```

## Best Practices

1. Always use environment variables for sensitive data:
```python
api_key_env_var = "MY_API_KEY"  # Store name of env var, not actual key
```

2. Enable encryption in production:
```python
if config.is_production:
    config.require_encryption = True
```

3. Use the configuration manager for centralized access:
```python
from lion.settings import config_manager
config_manager.validate_all()
```

4. Implement environment-specific validation:
```python
def validate_security(self) -> None:
    super().validate_security()
    if self.is_production:
        # Add production-specific validations
        pass
```

## Testing

Run the test suite:
```bash
pytest tests/test_core/test_config/
```

Test files:
- `test_base.py`: Base configuration tests
- `test_api.py`: API configuration tests
- `test_model.py`: Model configuration tests
- `test_manager.py`: Configuration manager tests

## Migration Guide

If you're using the old configuration system:

1. Update imports:
```python
from lion.core.config import ConfigurationManager
```

2. Use the configuration manager:
```python
config_manager = ConfigurationManager.get_instance()
```

3. Access configurations:
```python
# Old way
from lion.settings import Settings
settings = Settings()

# New way
from lion.settings import config_manager
api_config = config_manager.api_config
model_config = config_manager.model_config
```

The `settings.py` module maintains backward compatibility while providing access to the new configuration system.
