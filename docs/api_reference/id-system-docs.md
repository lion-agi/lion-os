# Lion-OS ID System

## Overview

The Lion-OS ID system provides type-safe unique identifiers for framework objects. It includes utilities for ID generation, validation, and type-safe references.

## Core Types

```python
# Base ID type
LnID: TypeAlias = Annotated[str, Field(description="A unique identifier.")]

# Generic type for Observable objects
T = TypeVar("T", bound=Observable)
```

## ID Class

The `ID` class provides type-safe ID references and utilities:

```python
class ID(Generic[T]):
    # Reference types
    Ref: TypeAlias = LnID | T             # ID string or object
    ID: TypeAlias = LnID                  # ID string only
    Item: TypeAlias = T                   # Object only

    # Collection types
    IDSeq: TypeAlias = Sequence[LnID] | Ordering[LnID]
    ItemSeq: TypeAlias = Sequence[T] | Mapping[LnID, T] | Container[LnID | T]
    RefSeq: TypeAlias = IDSeq | ItemSeq

    # Communication types
    SenderRecipient: TypeAlias = LnID | T
```

## ID Generation

```python
@staticmethod
def id(
    config: LionIDConfig = Settings.Config.ID,
    n: int = None,              # ID length
    prefix: str = None,         # Prefix string
    postfix: str = None,        # Postfix string
    random_hyphen: bool = None, # Add random hyphens
    num_hyphens: int = None,    # Number of hyphens
    hyphen_start_index: int = None,  # Hyphen start position
    hyphen_end_index: int = None,    # Hyphen end position
) -> LnID:
    """Generate a unique identifier."""
```

## ID Utilities

```python
@staticmethod
def get_id(item, config: LionIDConfig = Settings.Config.ID) -> str:
    """Extract Lion ID from an item."""

@staticmethod
def is_id(item, config: LionIDConfig = Settings.Config.ID) -> bool:
    """Check if an item is a valid Lion ID."""
```

## Error Types

```python
class ItemError(Exception):
    """Base error for item operations."""

class IDError(ItemError):
    """Invalid or missing ID error."""

class ItemNotFoundError(ItemError):
    """Item not found error."""

class ItemExistsError(ItemError):
    """Item already exists error."""
```

## Observable Protocol

Items that can have IDs must implement the Observable protocol:

```python
class Observable(ABC):
    ln_id: str
```

## Usage Example

```python
# Generate an ID
new_id = ID.id(prefix="user-", n=16)  # e.g., "user-a1b2c3d4e5f6g7h8"

# Check ID validity
is_valid = ID.is_id(new_id)  # True

# Get ID from item
item_id = ID.get_id(some_item)
```

## Notes

1. ID Generation:
   - Configurable through LionIDConfig
   - Supports prefixes and postfixes
   - Optional hyphenation

2. ID Validation:
   - Built-in validation rules
   - Configuration-based validation
   - Type-safe references

3. Error Handling:
   - Specific error types
   - Clear error messages
   - ID-aware exceptions
