# Models Module API Reference

## Module Overview

The models module provides the core data model classes for the Lion-OS framework. It builds on Pydantic with additional functionality for field management, schema handling, and nested data structures.

## Class Hierarchy

```
BaseModel
    └── BaseAutoModel
        ├── SchemaModel
        │   ├── FieldModel
        │   └── NewModelParams
        └── OperableModel
```

## Core Components

### BaseAutoModel

Base class extending Pydantic's BaseModel with additional serialization capabilities.

```python
class BaseAutoModel(BaseModel):
    def clean_dump(self) -> dict[str, Any]:
        """Remove undefined values from model dump."""

    def to_dict(self) -> dict:
        """Convert model to dictionary."""

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """Create model from dictionary."""
```

### SchemaModel

Base class for models with schema validation and unique hash support.

```python
class SchemaModel(BaseAutoModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_default=False,
        populate_by_name=True,
        arbitrary_types_allowed=True,
        use_enum_values=True,
    )
    _unique_hash: str = PrivateAttr(lambda: unique_hash(32))
```

Key Features:
- Forbids extra fields
- Provides unique hash support
- Maintains field keys list

### FieldModel

Represents a Pydantic field with extended validation and configuration.

```python
class FieldModel(SchemaModel):
    default: Any = UNDEFINED
    default_factory: Callable = UNDEFINED
    title: str = UNDEFINED
    description: str = UNDEFINED
    examples: list = UNDEFINED
    validators: list = UNDEFINED
    exclude: bool = UNDEFINED
    deprecated: bool = UNDEFINED
    frozen: bool = UNDEFINED
    alias: str = UNDEFINED
    alias_priority: int = UNDEFINED

    name: str = Field(..., exclude=True)
    annotation: type | Any = Field(UNDEFINED, exclude=True)
    validator: Callable | Any = Field(UNDEFINED, exclude=True)
    validator_kwargs: dict | Any = Field(default_factory=dict, exclude=True)
```

Properties:
- `field_info -> FieldInfo`: Creates Pydantic FieldInfo instance
- `field_validator -> dict[str, Callable]`: Gets field validator if defined

### OperableModel

Model supporting dynamic field management.

```python
class OperableModel(BaseAutoModel):
    extra_fields: dict[str, Any] = Field(default_factory=dict)
```

Key Methods:

```python
def add_field(
    self,
    field_name: FIELD_NAME,
    /,
    value: Any = UNDEFINED,
    annotation: type = UNDEFINED,
    field_obj: FieldInfo = UNDEFINED,
    field_model: FieldModel = UNDEFINED,
    **kwargs,
) -> None:
    """Add new field to extra_fields."""

def update_field(
    self,
    field_name: FIELD_NAME,
    /,
    value: Any = UNDEFINED,
    annotation: type = None,
    field_obj: FieldInfo = None,
    field_model: FieldModel = None,
    **kwargs,
) -> None:
    """Update existing field or create new one."""

def field_setattr(self, field_name: FIELD_NAME, attr: str, value: Any, /) -> None:
    """Set field attribute."""

def field_hasattr(self, field_name: FIELD_NAME, attr: str, /) -> bool:
    """Check field attribute existence."""

def field_getattr(self, field_name: FIELD_NAME, attr: str, default: Any = UNDEFINED, /) -> Any:
    """Get field attribute."""
```

Validation Rules:
- Cannot have both default and default_factory
- Cannot provide both field_obj and field_model
- field_obj must be FieldInfo instance
- field_model must be FieldModel instance

### Note

Container for nested dictionary data structures with advanced access patterns.

```python
class Note(BaseAutoModel):
    content: dict[str, Any] = Field(default_factory=dict)
```

Key Methods:

```python
def pop(self, indices: INDICE_TYPE, /, default: Any = UNDEFINED) -> Any:
    """Remove and return nested item."""

def insert(self, indices: INDICE_TYPE, value: Any, /) -> None:
    """Insert value at nested location."""

def set(self, indices: INDICE_TYPE, value: Any, /) -> None:
    """Set value at nested location."""

def get(self, indices: INDICE_TYPE, /, default: Any = UNDEFINED) -> Any:
    """Get value from nested location."""

def update(self, indices: INDICE_TYPE, value: Any) -> None:
    """Update nested structure."""
```

Features:
- Nested dictionary management
- Flattened key/value access
- Dict-like operations
- Type-safe updates

### NewModelParams

Configuration for creating new Pydantic models dynamically.

```python
class NewModelParams(SchemaModel):
    name: str | None = None
    parameter_fields: dict[str, FieldInfo] = Field(default_factory=dict)
    base_type: type[BaseModel] = Field(default=BaseModel)
    field_models: list[FieldModel] = Field(default_factory=list)
    exclude_fields: list = Field(default_factory=list)
    field_descriptions: dict = Field(default_factory=dict)
    inherit_base: bool = Field(default=True)
    use_base_kwargs: bool = False
    config_dict: dict | None = Field(default=None)
    doc: str | None = Field(default=None)
    frozen: bool = False
```

Validation Rules:
- Field models must be FieldModel instances
- Parameter fields must be FieldInfo instances
- Field names must be strings
- Base type must be BaseModel subclass/instance
- Descriptions must be strings

Key Method:
```python
def create_new_model(self) -> type[BaseModel]:
    """Create new Pydantic model type."""
```

## Common Configuration

All models use common configuration:
```python
common_config = {
    "populate_by_name": True,    # Allow population by field name
    "arbitrary_types_allowed": True,  # Allow arbitrary types
    "use_enum_values": True,     # Use enum values in serialization
}
```

## Type Definitions

```python
INDICE_TYPE = str | list[str | int]  # For nested access
FIELD_NAME = TypeVar("FIELD_NAME", bound=str)  # For field names
```

## Notes

1. Field Management:
   - Dynamic field addition/update
   - Type annotation support
   - Validation preservation
   - Schema integration

2. Serialization:
   - Clean serialization of undefined values
   - Proper nested structure handling
   - Type-safe conversions
   - Schema preservation

3. Validation:
   - Field type checking
   - Value validation
   - Schema validation
   - Configuration validation

4. Model Creation:
   - Dynamic model generation
   - Inheritance support
   - Validator integration
   - Configuration preservation
