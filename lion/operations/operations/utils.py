from enum import Enum
from typing import Any, Union


def parse_to_representation(
    choices: list[str] | type[Enum] | dict[str, Any]
) -> tuple[list[str], list[Any]]:
    """Convert choices into a consistent representation."""
    if isinstance(choices, type) and issubclass(choices, Enum):
        return [e.name for e in choices], [e.value for e in choices]
    elif isinstance(choices, dict):
        return list(choices.keys()), list(choices.values())
    return choices, choices


def parse_selection(
    selection: Any, choices: list[str] | type[Enum] | dict[str, Any]
) -> Any:
    """Parse a selection back to its original form."""
    if isinstance(choices, type) and issubclass(choices, Enum):
        return choices[selection].value
    elif isinstance(choices, dict):
        return choices.get(selection, selection)
    return selection
