import inspect
import json
import re
from collections import deque
from collections.abc import Callable, Generator, Iterable, Mapping, Sequence
from decimal import Decimal
from numbers import Complex
from typing import Any, Literal, TypeVar, overload
from xml.etree import ElementTree as ET

from pydantic import BaseModel
from pydantic_core import PydanticUndefined, PydanticUndefinedType

from .constants import (
    FALSE_VALUES,
    NUM_TYPES,
    PATTERNS,
    TRUE_VALUES,
    TYPE_MAP,
    UNDEFINED,
    KeysDict,
    UndefinedType,
    py_json_msp,
)
from .string_similarity import SIMILARITY_ALGO_MAP, SIMILARITY_TYPE, string_similarity

T = TypeVar("T")


def as_readable_json(input_: Any, /, **kwargs) -> str:
    """Convert input to a human-readable JSON string.

    Args:
        input_: Object to convert to readable JSON
        **kwargs: Additional arguments passed to json.dumps()

    Returns:
        A formatted, human-readable JSON string

    Raises:
        ValueError: If conversion to JSON fails
    """
    # Extract to_dict kwargs
    to_dict_kwargs = {
        "use_model_dump": True,
        "fuzzy_parse": True,
        "recursive": True,
        "recursive_python_only": True,
        "max_recursive_depth": 5,
    }
    to_dict_kwargs.update(kwargs)

    # Handle empty input
    if not input_:
        if isinstance(input_, list):
            return ""
        return "{}"

    try:
        if isinstance(input_, list):
            # For lists, convert and format each item separately
            items = []
            for item in input_:
                dict_ = to_dict(item, **to_dict_kwargs)
                items.append(
                    json.dumps(
                        dict_,
                        indent=4,
                        ensure_ascii=False,
                        default=lambda o: to_dict(o),
                    )
                )
            return "\n\n".join(items)

        # Handle single items
        dict_ = to_dict(input_, **to_dict_kwargs)

        # Extract json.dumps kwargs from input kwargs
        json_kwargs = {
            "indent": 4,
            "ensure_ascii": kwargs.get("ensure_ascii", False),
            "default": lambda o: to_dict(o),
        }

        # Add any other JSON-specific kwargs
        for k in ["indent", "separators", "cls"]:
            if k in kwargs:
                json_kwargs[k] = kwargs[k]

        # Convert to JSON string
        if kwargs.get("ensure_ascii", False):
            # Force ASCII encoding for special characters
            return json.dumps(
                dict_,
                ensure_ascii=True,
                **{k: v for k, v in json_kwargs.items() if k != "ensure_ascii"},
            )

        return json.dumps(dict_, **json_kwargs)

    except Exception as e:
        raise ValueError(f"Failed to convert input to readable JSON: {e}") from e


def as_readable(input_: Any, /, *, md: bool = False, **kwargs) -> str:
    """Convert input to readable string with optional markdown formatting.

    Args:
        input_: Object to convert
        md: Whether to wrap in markdown block
        **kwargs: Additional arguments for as_readable_json()

    Returns:
        Formatted string representation
    """
    try:
        result = as_readable_json(input_, **kwargs)
        if md:
            return f"```json\n{result}\n```"
        return result

    except Exception:
        return str(input_)


def extract_code_block(
    str_to_parse: str,
    return_as_list: bool = False,
    languages: list[str] | None = None,
    categorize: bool = False,
) -> str | list[str] | dict[str, list[str]]:
    """
    Extract code blocks from a given string containing
    Markdown-formatted text.

    This function identifies code blocks enclosed by triple
    backticks (```) or
    tildes (~~~), extracts their content, and can filter
    them based on specified
    programming languages. It provides options to return
    the extracted code
    blocks as a single concatenated string, a list, or a
    dictionary categorized
    by language.

    Args:
        str_to_parse: The input string containing Markdown
        code blocks.
        return_as_list: If True, returns a list of code
        blocks; otherwise, returns
            them as a single concatenated string separated
            by two newlines.
        languages: A list of languages to filter the code
        blocks. If None,
            extracts code blocks of all languages.
        categorize: If True, returns a dictionary mapping
        languages to lists of
            code blocks.

    Returns:
        Depending on the parameters:
            - A concatenated string of code blocks.
            - A list of code blocks.
            - A dictionary mapping languages to lists of
            code blocks.
    """
    code_blocks = []
    code_dict = {}

    pattern = re.compile(
        r"""
        ^(?P<fence>```|~~~)[ \t]*     # Opening fence ``` or ~~~
        (?P<lang>[\w+-]*)[ \t]*\n     # Optional language identifier
        (?P<code>.*?)(?<=\n)          # Code content
        ^(?P=fence)[ \t]*$            # Closing fence matching the opening
        """,
        re.MULTILINE | re.DOTALL | re.VERBOSE,
    )

    for match in pattern.finditer(str_to_parse):
        lang = match.group("lang") or "plain"
        code = match.group("code")

        if languages is None or lang in languages:
            if categorize:
                code_dict.setdefault(lang, []).append(code)
            else:
                code_blocks.append(code)

    if categorize:
        return code_dict
    elif return_as_list:
        return code_blocks
    else:
        return "\n\n".join(code_blocks)


def extract_docstring(
    func: Callable, style: Literal["google", "rest"] = "google"
) -> tuple[str | None, dict[str, str]]:
    """
    Extract function description and parameter descriptions from docstring.

    Args:
        func: The function from which to extract docstring details.
        style: The style of docstring to parse ('google' or 'rest').

    Returns:
        A tuple containing the function description and a dictionary with
        parameter names as keys and their descriptions as values.

    Raises:
        ValueError: If an unsupported style is provided.

    Examples:
        >>> def example_function(param1: int, param2: str):
        ...     '''Example function.
        ...
        ...     Args:
        ...         param1 (int): The first parameter.
        ...         param2 (str): The second parameter.
        ...     '''
        ...     pass
        >>> description, params = extract_docstring_details(example_function)
        >>> description
        'Example function.'
        >>> params == {'param1': 'The first parameter.',
        ...            'param2': 'The second parameter.'}
        True
    """
    style = str(style).strip().lower()

    if style == "google":
        func_description, params_description = _extract_docstring_details_google(func)
    elif style == "rest":
        func_description, params_description = _extract_docstring_details_rest(func)
    else:
        raise ValueError(
            f'{style} is not supported. Please choose either "google" or' ' "reST".'
        )
    return func_description, params_description


def _extract_docstring_details_google(
    func: Callable,
) -> tuple[str | None, dict[str, str]]:
    """
    Extract details from Google-style docstring.

    Args:
        func: The function from which to extract docstring details.

    Returns:
        A tuple containing the function description and a dictionary with
        parameter names as keys and their descriptions as values.

    Examples:
        >>> def example_function(param1: int, param2: str):
        ...     '''Example function.
        ...
        ...     Args:
        ...         param1 (int): The first parameter.
        ...         param2 (str): The second parameter.
        ...     '''
        ...     pass
        >>> description, params = _extract_docstring_details_google(
        ...     example_function)
        >>> description
        'Example function.'
        >>> params == {'param1': 'The first parameter.',
        ...            'param2': 'The second parameter.'}
        True
    """
    docstring = inspect.getdoc(func)
    if not docstring:
        return None, {}
    lines = docstring.split("\n")
    func_description = lines[0].strip()

    lines_len = len(lines)

    params_description = {}
    param_start_pos = next(
        (
            i + 1
            for i in range(1, lines_len)
            if (
                (a := str(lines[i]).strip().lower()).startswith("args")
                or a.startswith("parameters")
                or a.startswith("params")
                or a.startswith("arguments")
            )
        ),
        0,
    )
    current_param = None
    for i in range(param_start_pos, lines_len):
        if lines[i] == "":
            continue
        elif lines[i].startswith(" "):
            param_desc = lines[i].split(":", 1)
            if len(param_desc) == 1:
                params_description[current_param] += f" {param_desc[0].strip()}"
                continue
            param, desc = param_desc
            param = param.split("(")[0].strip()
            params_description[param] = desc.strip()
            current_param = param
        else:
            break
    return func_description, params_description


def _extract_docstring_details_rest(
    func: Callable,
) -> tuple[str | None, dict[str, str]]:
    """
    Extract details from reStructuredText-style docstring.

    Args:
        func: The function from which to extract docstring details.

    Returns:
        A tuple containing the function description and a dictionary with
        parameter names as keys and their descriptions as values.

    Examples:
        >>> def example_function(param1: int, param2: str):
        ...     '''Example function.
        ...
        ...     :param param1: The first parameter.
        ...     :type param1: int
        ...     :param param2: The second parameter.
        ...     :type param2: str
        ...     '''
        ...     pass
        >>> description, params = _extract_docstring_details_rest(
        ...     example_function)
        >>> description
        'Example function.'
        >>> params == {'param1': 'The first parameter.',
        ...            'param2': 'The second parameter.'}
        True
    """
    docstring = inspect.getdoc(func)
    if not docstring:
        return None, {}
    lines = docstring.split("\n")
    func_description = lines[0].strip()

    params_description = {}
    current_param = None
    for line in lines[1:]:
        line = line.strip()
        if line.startswith(":param"):
            param_desc = line.split(":", 2)
            _, param, desc = param_desc
            param = param.split()[-1].strip()
            params_description[param] = desc.strip()
            current_param = param
        elif line.startswith(" "):
            params_description[current_param] += f" {line}"

    return func_description, params_description


def extract_json_blocks(
    str_to_parse: str,
    suppress: bool = True,
    fuzzy_parse: bool = True,
    dropna: bool = True,
) -> list[dict[str, Any]]:
    """
    Extract and parse JSON blocks from the given text.

    This function searches for JSON blocks enclosed in triple backticks
    within the input text, parses them, and returns a list of parsed
    dictionaries.

    Args:
        text: The input text containing JSON blocks.
        suppress: If True, suppress errors during parsing. Default is True.
        fuzzy_parse: If True, use fuzzy parsing for JSON. Default is True.
        dropna: If True, remove None values from the result. Default is True.

    Returns:
        A list of parsed JSON blocks as dictionaries.

    Example:
        >>> text = "```json\n{\"key\": \"value\"}\n```"
        >>> extract_json_blocks(text)
        [{'key': 'value'}]
    """
    pattern = r"```json\s*(.*?)\s*```"
    matches = re.findall(pattern, str_to_parse, re.DOTALL)

    json_blocks = [
        to_dict(match, fuzzy_parse=fuzzy_parse, suppress=suppress) for match in matches
    ]

    return [block for block in json_blocks if block] if dropna else json_blocks


def extract_block(
    str_to_parse: str,
    language: str = "json",
    regex_pattern: str | None = None,
    *,
    parser: Callable[[str], Any] | None = None,
    suppress: bool = False,
) -> dict[str, Any] | None:
    """
    Extract and parse a code block from the given string.

    This function searches for a code block in the input string using a
    regular expression pattern, extracts it, and parses it using the
    provided parser function.

    Args:
        str_to_parse: The input string containing the code block.
        language: The language of the code block. Default is "json".
        regex_pattern: Custom regex pattern to find the code block.
            If provided, overrides the default pattern.
        parser: A function to parse the extracted code string.
            If not provided, uses `to_dict` with fuzzy parsing.
        suppress: If True, return None instead of raising an error
            when no code block is found. Default is False.

    Returns:
        The parsed content of the code block as a dictionary,
        or None if no block is found and suppress is True.

    Raises:
        ValueError: If no code block is found and suppress is False.

    Example:
        >>> text = "```json\n{\"key\": \"value\"}\n```"
        >>> extract_block(text)
        {'key': 'value'}
    """
    if not regex_pattern:
        regex_pattern = rf"```{language}\n?(.*?)\n?```"
    if not language:
        regex_pattern = r"```\n?(.*?)\n?```"

    match = re.search(regex_pattern, str_to_parse, re.DOTALL)

    if match:
        code_str = match.group(1).strip()
    elif str_to_parse.startswith(f"```{language}\n") and str_to_parse.endswith("\n```"):
        code_str = str_to_parse[4 + len(language) : -4].strip()
    elif suppress:
        return None
    else:
        raise ValueError("No code block found in the input string.")

    parser = parser or (lambda x: to_dict(x, fuzzy_parse=True, suppress=True))
    return parser(code_str)


def extract_json_schema(
    data: Any,
    *,
    sep: str = "|",
    coerce_keys: bool = True,
    dynamic: bool = True,
    coerce_sequence: Literal["dict", "list"] | None = None,
    max_depth: int | None = None,
) -> dict[str, Any]:
    """
    Extract a JSON schema from JSON data.

    This function uses the flatten function to create a flat representation
    of the JSON data, then builds a schema based on the flattened structure.

    Args:
        data: The JSON data to extract the schema from.
        sep: Separator used in flattened keys.
        coerce_keys: Whether to coerce keys to strings.
        dynamic: Whether to use dynamic flattening.
        coerce_sequence: How to coerce sequences ("dict", "list", or None).
        max_depth: Maximum depth to flatten.

    Returns:
        A dictionary representing the JSON schema.
    """
    flattened = flatten(
        data,
        sep=sep,
        coerce_keys=coerce_keys,
        dynamic=dynamic,
        coerce_sequence=coerce_sequence,
        max_depth=max_depth,
    )

    schema = {}
    for key, value in flattened.items():
        key_parts = key.split(sep) if isinstance(key, str) else key
        current = schema
        for part in key_parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        current[key_parts[-1]] = _get_type(value)

    return {"type": "object", "properties": _consolidate_schema(schema)}


def _get_type(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        return {"type": "string"}
    elif isinstance(value, bool):
        return {"type": "boolean"}
    elif isinstance(value, int):
        return {"type": "integer"}
    elif isinstance(value, float):
        return {"type": "number"}
    elif isinstance(value, list):
        if not value:
            return {"type": "array", "items": {}}
        item_types = [_get_type(item) for item in value]
        if all(item_type == item_types[0] for item_type in item_types):
            return {"type": "array", "items": item_types[0]}
        else:
            return {"type": "array", "items": {"oneOf": item_types}}
    elif isinstance(value, dict):
        return {
            "type": "object",
            "properties": _consolidate_schema(
                {k: _get_type(v) for k, v in value.items()}
            ),
        }
    elif value is None:
        return {"type": "null"}
    else:
        return {"type": "any"}


def _consolidate_schema(schema: dict) -> dict:
    """
    Consolidate the schema to handle lists and nested structures.
    """
    consolidated = {}
    for key, value in schema.items():
        if isinstance(value, dict) and all(k.isdigit() for k in value.keys()):
            # This is likely a list
            item_types = list(value.values())
            if all(item_type == item_types[0] for item_type in item_types):
                consolidated[key] = {"type": "array", "items": item_types[0]}
            else:
                consolidated[key] = {
                    "type": "array",
                    "items": {"oneOf": item_types},
                }
        elif isinstance(value, dict) and "type" in value:
            consolidated[key] = value
        else:
            consolidated[key] = _consolidate_schema(value)
    return consolidated


def json_schema_to_cfg(
    schema: dict[str, Any], start_symbol: str = "S"
) -> list[tuple[str, list[str]]]:
    productions = []
    visited = set()
    symbol_counter = 0

    def generate_symbol(base: str) -> str:
        nonlocal symbol_counter
        symbol = f"{base}@{symbol_counter}"
        symbol_counter += 1
        return symbol

    def generate_rules(s: dict[str, Any], symbol: str):
        if symbol in visited:
            return
        visited.add(symbol)

        if s.get("type") == "object":
            properties = s.get("properties", {})
            if properties:
                props_symbol = generate_symbol("PROPS")
                productions.append((symbol, ["{", props_symbol, "}"]))

                productions.append((props_symbol, []))  # Empty object
                for i, prop in enumerate(properties):
                    prop_symbol = generate_symbol(prop)
                    if i == 0:
                        productions.append((props_symbol, [prop_symbol]))
                    else:
                        productions.append(
                            (props_symbol, [props_symbol, ",", prop_symbol])
                        )

                for prop, prop_schema in properties.items():
                    prop_symbol = generate_symbol(prop)
                    value_symbol = generate_symbol("VALUE")
                    productions.append((prop_symbol, [f'"{prop}"', ":", value_symbol]))
                    generate_rules(prop_schema, value_symbol)
            else:
                productions.append((symbol, ["{", "}"]))

        elif s.get("type") == "array":
            items = s.get("items", {})
            items_symbol = generate_symbol("ITEMS")
            value_symbol = generate_symbol("VALUE")
            productions.append((symbol, ["[", "]"]))
            productions.append((symbol, ["[", items_symbol, "]"]))
            productions.append((items_symbol, [value_symbol]))
            productions.append((items_symbol, [value_symbol, ",", items_symbol]))
            generate_rules(items, value_symbol)

        elif s.get("type") == "string":
            productions.append((symbol, ["STRING"]))

        elif s.get("type") == "number":
            productions.append((symbol, ["NUMBER"]))

        elif s.get("type") == "integer":
            productions.append((symbol, ["INTEGER"]))

        elif s.get("type") == "boolean":
            productions.append((symbol, ["BOOLEAN"]))

        elif s.get("type") == "null":
            productions.append((symbol, ["NULL"]))

    generate_rules(schema, start_symbol)
    return productions


def json_schema_to_regex(schema: dict[str, Any]) -> str:
    def schema_to_regex(s):
        if s.get("type") == "object":
            properties = s.get("properties", {})
            prop_patterns = [
                rf'"{prop}"\s*:\s*{schema_to_regex(prop_schema)}'
                for prop, prop_schema in properties.items()
            ]
            return (
                r"\{"
                + r"\s*("
                + r"|".join(prop_patterns)
                + r")"
                + r"(\s*,\s*("
                + r"|".join(prop_patterns)
                + r"))*\s*\}"
            )
        elif s.get("type") == "array":
            items = s.get("items", {})
            return (
                r"\[\s*("
                + schema_to_regex(items)
                + r"(\s*,\s*"
                + schema_to_regex(items)
                + r")*)?\s*\]"
            )
        elif s.get("type") == "string":
            return r'"[^"]*"'
        elif s.get("type") == "integer":
            return r"-?\d+"
        elif s.get("type") == "number":
            return r"-?\d+(\.\d+)?"
        elif s.get("type") == "boolean":
            return r"(true|false)"
        elif s.get("type") == "null":
            return r"null"
        else:
            return r".*"

    return "^" + schema_to_regex(schema) + "$"


def print_cfg(productions: list[tuple[str, list[str]]]):
    for lhs, rhs in productions:
        print(f"{lhs} -> {' '.join(rhs)}")


def function_to_schema(
    f_,
    style: Literal["google", "rest"] = "google",
    *,
    f_description=None,
    p_description=None,
) -> dict:
    """
    Generate a schema description for a given function. in openai format

    This function generates a schema description for the given function
    using typing hints and docstrings. The schema includes the function's
    name, description, and parameter details.

    Args:
        func (Callable): The function to generate a schema for.
        style (str): The docstring format. Can be 'google' (default) or
            'reST'.
        func_description (str, optional): A custom description for the
            function. If not provided, the description will be extracted
            from the function's docstring.
        params_description (dict, optional): A dictionary mapping
            parameter names to their descriptions. If not provided, the
            parameter descriptions will be extracted from the function's
            docstring.

    Returns:
        dict: A schema describing the function, including its name,
        description, and parameter details.

    Example:
        >>> def example_func(param1: int, param2: str) -> bool:
        ...     '''Example function.
        ...
        ...     Args:
        ...         param1 (int): The first parameter.
        ...         param2 (str): The second parameter.
        ...     '''
        ...     return True
        >>> schema = function_to_schema(example_func)
        >>> schema['function']['name']
        'example_func'
    """
    # Extract function name
    func_name = f_.__name__

    # Extract function description and parameter descriptions
    if not f_description or not p_description:
        func_desc, params_desc = extract_docstring(f_, style)
        f_description = f_description or func_desc
        p_description = p_description or params_desc

    # Extract parameter details using typing hints
    sig = inspect.signature(f_)
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    for name, param in sig.parameters.items():
        # Default type to string and update if type hint is available
        param_type = "string"
        if param.annotation is not inspect.Parameter.empty:
            param_type = py_json_msp[param.annotation.__name__]

        # Extract parameter description from docstring, if available
        param_description = p_description.get(name)

        # Assuming all parameters are required for simplicity
        parameters["required"].append(name)
        parameters["properties"][name] = {
            "type": param_type,
            "description": param_description,
        }

    return {
        "type": "function",
        "function": {
            "name": func_name,
            "description": f_description,
            "parameters": parameters,
        },
    }


def fuzzy_parse_json(str_to_parse: str, /) -> dict[str, Any] | list[dict[str, Any]]:
    """Parse a JSON string with automatic fixing of common formatting issues.

    Args:
        str_to_parse: The JSON string to parse

    Returns:
        The parsed JSON object as a dictionary

    Raises:
        ValueError: If the string cannot be parsed as valid JSON
        TypeError: If the input is not a string or the result is not a dict
    """
    if not isinstance(str_to_parse, str):
        raise TypeError("Input must be a string")

    if not str_to_parse.strip():
        raise ValueError("Input string is empty")

    try:
        return json.loads(str_to_parse)
    except Exception:
        pass

    cleaned = _clean_json_string(str_to_parse)
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    try:
        fixed = fix_json_string(cleaned)
        return json.loads(fixed)
    except Exception as e:
        raise ValueError(
            f"Failed to parse JSON string after all fixing attempts: {e}"
        ) from e


def _clean_json_string(s: str) -> str:
    """Clean and standardize a JSON string."""
    s = re.sub(r"(?<!\\)'", '"', s)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r'([{,])\s*([^"\s]+):', r'\1"\2":', s)
    return s.strip()


def fix_json_string(str_to_parse: str, /) -> str:
    """Fix a JSON string by ensuring all brackets are properly closed.

    Args:
        str_to_parse: JSON string to fix

    Returns:
        Fixed JSON string with proper bracket closure

    Raises:
        ValueError: If mismatched or extra closing brackets are found
    """
    if not str_to_parse:
        raise ValueError("Input string is empty")

    brackets = {"{": "}", "[": "]"}
    open_brackets = []
    pos = 0
    length = len(str_to_parse)

    while pos < length:
        char = str_to_parse[pos]

        # Handle escape sequences
        if char == "\\":
            pos += 2  # Skip escape sequence
            continue

        # Handle string content
        if char == '"':
            pos += 1
            # Skip until closing quote, accounting for escapes
            while pos < length:
                if str_to_parse[pos] == "\\":
                    pos += 2  # Skip escape sequence
                    continue
                if str_to_parse[pos] == '"':
                    break
                pos += 1
            pos += 1
            continue

        # Handle brackets
        if char in brackets:
            open_brackets.append(brackets[char])
        elif char in brackets.values():
            if not open_brackets:
                raise ValueError(f"Extra closing bracket '{char}' at position {pos}")
            if open_brackets[-1] != char:
                raise ValueError(f"Mismatched bracket '{char}' at position {pos}")
            open_brackets.pop()

        pos += 1

    # Add missing closing brackets
    closing_brackets = "".join(reversed(open_brackets))
    return str_to_parse + closing_brackets


@overload
def to_dict(
    input_: type[None] | UndefinedType | PydanticUndefinedType, /
) -> dict[str, Any]: ...


@overload
def to_dict(input_: Mapping, /) -> dict[str, Any]: ...


@overload
def to_dict(input_: set, /) -> dict[Any, Any]: ...


@overload
def to_dict(input_: Sequence, /) -> dict[str, Any]: ...


@overload
def to_dict(
    input_: Any,
    /,
    *,
    use_model_dump: bool = True,
    fuzzy_parse: bool = False,
    suppress: bool = False,
    str_type: Literal["json", "xml"] | None = "json",
    parser: Callable[[str], dict[str, Any]] | None = None,
    recursive: bool = False,
    max_recursive_depth: int = None,
    exclude_types: tuple = (),
    recursive_python_only: bool = True,
    **kwargs: Any,
) -> dict[str, Any]: ...


def to_dict(
    input_: Any,
    /,
    *,
    use_model_dump: bool = True,
    fuzzy_parse: bool = False,
    suppress: bool = False,
    str_type: Literal["json", "xml"] | None = "json",
    parser: Callable[[str], dict[str, Any]] | None = None,
    recursive: bool = False,
    max_recursive_depth: int = None,
    exclude_types: tuple = (),
    recursive_python_only: bool = True,
    **kwargs: Any,
):
    """
    Convert various input types to a dictionary, with optional recursive processing.

    Args:
        input_: The input to convert.
        use_model_dump: Use model_dump() for Pydantic models if available.
        fuzzy_parse: Use fuzzy parsing for string inputs.
        suppress: Return empty dict on errors if True.
        str_type: Input string type ("json" or "xml").
        parser: Custom parser function for string inputs.
        recursive: Enable recursive conversion of nested structures.
        max_recursive_depth: Maximum recursion depth (default 5, max 10).
        exclude_types: Tuple of types to exclude from conversion.
        recursive_python_only: If False, attempts to convert custom types recursively.
        **kwargs: Additional arguments for parsing functions.

    Returns:
        dict[str, Any]: A dictionary derived from the input.

    Raises:
        ValueError: If parsing fails and suppress is False.

    Examples:
        >>> to_dict({"a": 1, "b": [2, 3]})
        {'a': 1, 'b': [2, 3]}
        >>> to_dict('{"x": 10}', str_type="json")
        {'x': 10}
        >>> to_dict({"a": {"b": {"c": 1}}}, recursive=True, max_recursive_depth=2)
        {'a': {'b': {'c': 1}}}
    """
    try:
        if recursive:
            return recursive_to_dict(
                input_,
                use_model_dump=use_model_dump,
                fuzzy_parse=fuzzy_parse,
                str_type=str_type,
                parser=parser,
                max_recursive_depth=max_recursive_depth,
                exclude_types=exclude_types,
                recursive_custom_types=not recursive_python_only,
                **kwargs,
            )

        return _to_dict(
            input_,
            fuzzy_parse=fuzzy_parse,
            parser=parser,
            str_type=str_type,
            use_model_dump=use_model_dump,
            exclude_types=exclude_types,
            **kwargs,
        )
    except Exception as e:
        if suppress:
            return {}
        raise e


def _to_dict(
    input_: Any,
    /,
    *,
    use_model_dump: bool = True,
    fuzzy_parse: bool = False,
    str_type: Literal["json", "xml"] | None = "json",
    parser: Callable[[str], dict[str, Any]] | None = None,
    exclude_types: tuple = (),
    **kwargs: Any,
) -> dict[str, Any]:
    """Convert various input types to a dictionary.

    Handles multiple input types, including None, Mappings, strings, and more.

    Args:
        input_: The input to convert to a dictionary.
        use_model_dump: Use model_dump() for Pydantic models if available.
        fuzzy_parse: Use fuzzy parsing for string inputs.
        suppress: Return empty dict on parsing errors if True.
        str_type: Input string type, either "json" or "xml".
        parser: Custom parser function for string inputs.
        **kwargs: Additional arguments passed to parsing functions.

    Returns:
        A dictionary derived from the input.

    Raises:
        ValueError: If string parsing fails and suppress is False.

    Examples:
        >>> to_dict({"a": 1, "b": 2})
        {'a': 1, 'b': 2}
        >>> to_dict('{"x": 10}', str_type="json")
        {'x': 10}
        >>> to_dict("<root><a>1</a></root>", str_type="xml")
        {'a': '1'}
    """
    if isinstance(exclude_types, tuple) and len(exclude_types) > 0:
        if isinstance(input_, exclude_types):
            return input_

    if isinstance(input_, dict):
        return input_

    if use_model_dump and hasattr(input_, "model_dump"):
        return input_.model_dump(**kwargs)

    if isinstance(input_, type(None) | UndefinedType | PydanticUndefinedType):
        return _undefined_to_dict(input_)
    if isinstance(input_, Mapping):
        return _mapping_to_dict(input_)

    if isinstance(input_, str):
        if fuzzy_parse:
            parser = fuzzy_parse_json
        try:
            a = _str_to_dict(
                input_,
                str_type=str_type,
                parser=parser,
                **kwargs,
            )
            if isinstance(a, dict):
                return a
        except Exception as e:
            raise ValueError("Failed to convert string to dictionary") from e

    if isinstance(input_, set):
        return _set_to_dict(input_)
    if isinstance(input_, Iterable):
        return _iterable_to_dict(input_)

    return _generic_type_to_dict(input_, **kwargs)


def _recursive_to_dict(
    input_: Any,
    /,
    *,
    max_recursive_depth: int,
    current_depth: int = 0,
    recursive_custom_types: bool = False,
    exclude_types: tuple = (),
    **kwargs: Any,
) -> Any:

    if current_depth >= max_recursive_depth:
        return input_

    if isinstance(input_, str):
        try:
            # Attempt to parse the string
            parsed = _to_dict(input_, **kwargs)
            # Recursively process the parsed result
            return _recursive_to_dict(
                parsed,
                max_recursive_depth=max_recursive_depth,
                current_depth=current_depth + 1,
                recursive_custom_types=recursive_custom_types,
                exclude_types=exclude_types,
                **kwargs,
            )
        except Exception:
            # Return the original string if parsing fails
            return input_

    elif isinstance(input_, dict):
        # Recursively process dictionary values
        return {
            key: _recursive_to_dict(
                value,
                max_recursive_depth=max_recursive_depth,
                current_depth=current_depth + 1,
                recursive_custom_types=recursive_custom_types,
                exclude_types=exclude_types,
                **kwargs,
            )
            for key, value in input_.items()
        }

    elif isinstance(input_, (list, tuple)):
        # Recursively process list or tuple elements
        processed = [
            _recursive_to_dict(
                element,
                max_recursive_depth=max_recursive_depth,
                current_depth=current_depth + 1,
                recursive_custom_types=recursive_custom_types,
                exclude_types=exclude_types,
                **kwargs,
            )
            for element in input_
        ]
        return type(input_)(processed)

    elif recursive_custom_types:
        # Process custom classes if enabled
        try:
            obj_dict = to_dict(input_, **kwargs)
            return _recursive_to_dict(
                obj_dict,
                max_recursive_depth=max_recursive_depth,
                current_depth=current_depth + 1,
                recursive_custom_types=recursive_custom_types,
                exclude_types=exclude_types,
                **kwargs,
            )
        except Exception:
            return input_

    else:
        # Return the input as is for other data types
        return input_


def recursive_to_dict(
    input_: Any,
    /,
    *,
    max_recursive_depth: int = None,
    exclude_types: tuple = (),
    recursive_custom_types: bool = False,
    **kwargs: Any,
) -> Any:

    if not isinstance(max_recursive_depth, int):
        max_recursive_depth = 5
    else:
        if max_recursive_depth < 0:
            raise ValueError("max_recursive_depth must be a non-negative integer")
        if max_recursive_depth == 0:
            return input_
        if max_recursive_depth > 10:
            raise ValueError("max_recursive_depth must be less than or equal to 10")

    return _recursive_to_dict(
        input_,
        max_recursive_depth=max_recursive_depth,
        current_depth=0,
        recursive_custom_types=recursive_custom_types,
        exclude_types=exclude_types,
        **kwargs,
    )


def _undefined_to_dict(
    input_: type[None] | UndefinedType | PydanticUndefinedType,
    /,
) -> dict:
    return {}


def _mapping_to_dict(input_: Mapping, /) -> dict:
    return dict(input_)


def _str_to_dict(
    input_: str,
    /,
    *,
    str_type: Literal["json", "xml"] | None = "json",
    parser: Callable[[str], dict[str, Any]] | None = None,
    **kwargs: Any,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Handle string inputs."""
    if not input_:
        return {}

    if str_type == "json":
        try:
            return (
                json.loads(input_, **kwargs)
                if parser is None
                else parser(input_, **kwargs)
            )
        except json.JSONDecodeError as e:
            raise ValueError("Failed to parse JSON string") from e

    if str_type == "xml":
        try:
            if parser is None:
                return xml_to_dict(input_, **kwargs)
            return parser(input_, **kwargs)
        except Exception as e:
            raise ValueError("Failed to parse XML string") from e

    raise ValueError(
        f"Unsupported string type for `to_dict`: {str_type}, it should "
        "be 'json' or 'xml'."
    )


def _set_to_dict(input_: set, /) -> dict:
    return {value: value for value in input_}


def _iterable_to_dict(input_: Iterable, /) -> dict:
    return {idx: v for idx, v in enumerate(input_)}


def _generic_type_to_dict(
    input_,
    /,
    **kwargs: Any,
) -> dict[str, Any]:

    try:
        for method in ["to_dict", "dict", "json", "to_json"]:
            if hasattr(input_, method):
                result = getattr(input_, method)(**kwargs)
                return json.loads(result) if isinstance(result, str) else result
    except Exception:
        pass

    if hasattr(input_, "__dict__"):
        return input_.__dict__

    try:
        return dict(input_)
    except Exception as e:
        raise ValueError(f"Unable to convert input to dictionary: {e}")


def to_json(
    string: str | list[str], /, fuzzy_parse: bool = False
) -> list[dict[str, Any]] | dict:
    """Extract and parse JSON content from a string or markdown code blocks.

    This function attempts to parse JSON directly from the input string first.
    If that fails, it looks for JSON content within markdown code blocks
    (denoted by ```json).

    Args:
        string: Input string or list of strings to parse. If a list is provided,
               it will be joined with newlines.

    Returns:
        - A dictionary if a single JSON object is found
        - A list of dictionaries if multiple JSON objects are found
        - An empty list if no valid JSON is found

    Examples:
        >>> to_json('{"key": "value"}')
        {'key': 'value'}

        >>> to_json('''
        ... ```json
        ... {"key": "value"}
        ... ```
        ... ''')
        {'key': 'value'}

        >>> to_json('''
        ... ```json
        ... {"key1": "value1"}
        ... ```
        ... ```json
        ... {"key2": "value2"}
        ... ```
        ... ''')
        [{'key1': 'value1'}, {'key2': 'value2'}]
    """

    if isinstance(string, list):
        string = "\n".join(string)

    # Try direct JSON parsing first
    try:
        if fuzzy_parse:
            return fuzzy_parse_json(string)
        return json.loads(string)
    except Exception:
        pass

    # Look for JSON in markdown code blocks
    pattern = r"```json\s*(.*?)\s*```"
    matches = re.findall(pattern, string, re.DOTALL)

    if not matches:
        return []

    if len(matches) == 1:
        return json.loads(matches[0])

    if fuzzy_parse:
        return [fuzzy_parse_json(match) for match in matches]
    return [json.loads(match) for match in matches]


@overload
def to_list(input_: None | UndefinedType | PydanticUndefinedType, /) -> list: ...


@overload
def to_list(
    input_: str | bytes | bytearray, /, use_values: bool = False
) -> list[str | int]: ...


@overload
def to_list(input_: Mapping, /, use_values: bool = False) -> list[Any]: ...


@overload
def to_list(
    input_: Any,
    /,
    *,
    flatten: bool = False,
    dropna: bool = False,
    unique: bool = False,
) -> list: ...


def to_list(
    input_: Any,
    /,
    *,
    flatten: bool = False,
    dropna: bool = False,
    unique: bool = False,
    use_values: bool = False,
) -> list:
    """Convert various input types to a list.

    Handles different input types and converts them to a list, with options
    for flattening nested structures and removing None values.

    Args:
        input_: The input to be converted to a list.
        flatten: If True, flattens nested list structures.
        dropna: If True, removes None values from the result.
        unique: If True, returns only unique values (requires flatten=True).
        use_values: If True, uses .values() for dict-like inputs.

    Returns:
        A list derived from the input, processed as specified.

    Raises:
        ValueError: If unique=True and flatten=False.

    Examples:
        >>> to_list(1)
        [1]
        >>> to_list([1, [2, 3]], flatten=True)
        [1, 2, 3]
        >>> to_list([1, None, 2], dropna=True)
        [1, 2]
        >>> to_list({'a': 1, 'b': 2}, use_values=True)
        [1, 2]
    """
    if unique and not flatten:
        raise ValueError("unique=True requires flatten=True")

    lst_ = _to_list_type(input_, use_values=use_values)

    if any((flatten, dropna)):
        lst_ = _process_list(
            lst=lst_,
            flatten=flatten,
            dropna=dropna,
        )

    return list(set(lst_)) if unique else lst_


def _undefined_to_list(input_: None | UndefinedType | PydanticUndefinedType, /) -> list:
    return []


def _str_to_list(
    input_: str | bytes | bytearray, /, use_values: bool = False
) -> list[str | int]:
    if use_values:
        return list(input_)
    return [input_]


def _dict_to_list(input_: Mapping, /, use_values: bool = False) -> list[Any]:
    if use_values:
        return list(input_.values())
    return [input_]


def _to_list_type(input_: Any, /, use_values: bool = False) -> Any | None:

    if isinstance(input_, BaseModel):
        return [input_]

    if use_values and hasattr(input_, "values"):
        return list(input_.values())

    if isinstance(input_, list):
        return input_

    if isinstance(input_, type(None) | UndefinedType | PydanticUndefinedType):
        return _undefined_to_list(input_)

    if isinstance(input_, str | bytes | bytearray):
        return _str_to_list(input_, use_values=use_values)

    if isinstance(input_, dict):
        return _dict_to_list(input_, use_values=use_values)

    if isinstance(input_, Iterable):
        return list(input_)

    return [input_]


def _process_list(lst: list[Any], flatten: bool, dropna: bool) -> list[Any]:
    """Process a list by optionally flattening and removing None values.

    Args:
        lst: The list to process.
        flatten: If True, flattens nested list structures.
        dropna: If True, removes None values.

    Returns:
        The processed list.
    """
    result = []
    for item in lst:
        if isinstance(item, Iterable) and not isinstance(
            item, (str, bytes, bytearray, Mapping)
        ):
            if flatten:
                result.extend(
                    _process_list(
                        lst=list(item),
                        flatten=flatten,
                        dropna=dropna,
                    )
                )
            else:
                result.append(
                    _process_list(
                        lst=list(item),
                        flatten=flatten,
                        dropna=dropna,
                    )
                )
        elif not dropna or item is not None:
            result.append(item)

    return result


def extract_numbers(text: str) -> list[tuple[str, str]]:
    """Extract numeric values from text using ordered regex patterns.

    Args:
        text: The text to extract numbers from.

    Returns:
        List of tuples containing (pattern_type, matched_value).
    """
    combined_pattern = "|".join(PATTERNS.values())
    matches = re.finditer(combined_pattern, text, re.IGNORECASE)
    numbers = []

    for match in matches:
        value = match.group()
        # Check which pattern matched
        for pattern_name, pattern in PATTERNS.items():
            if re.fullmatch(pattern, value, re.IGNORECASE):
                numbers.append((pattern_name, value))
                break

    return numbers


def validate_num_type(num_type: NUM_TYPES) -> type:
    """Validate and normalize numeric type specification.

    Args:
        num_type: The numeric type to validate.

    Returns:
        The normalized Python type object.

    Raises:
        ValueError: If the type specification is invalid.
    """
    if isinstance(num_type, str):
        if num_type not in TYPE_MAP:
            raise ValueError(f"Invalid number type: {num_type}")
        return TYPE_MAP[num_type]

    if num_type not in (int, float, complex):
        raise ValueError(f"Invalid number type: {num_type}")
    return num_type


def infer_type(value: tuple[str, str]) -> type:
    """Infer appropriate numeric type from value.

    Args:
        value: Tuple of (pattern_type, matched_value).

    Returns:
        The inferred Python type.
    """
    pattern_type, _ = value
    if pattern_type in ("complex", "complex_sci", "pure_imaginary"):
        return complex
    return float


def convert_special(value: str) -> float:
    """Convert special float values (inf, -inf, nan).

    Args:
        value: The string value to convert.

    Returns:
        The converted float value.
    """
    value = value.lower()
    if "infinity" in value or "inf" in value:
        return float("-inf") if value.startswith("-") else float("inf")
    return float("nan")


def convert_percentage(value: str) -> float:
    """Convert percentage string to float.

    Args:
        value: The percentage string to convert.

    Returns:
        The converted float value.

    Raises:
        ValueError: If the percentage value is invalid.
    """
    try:
        return float(value.rstrip("%")) / 100
    except ValueError as e:
        raise ValueError(f"Invalid percentage value: {value}") from e


def convert_complex(value: str) -> complex:
    """Convert complex number string to complex.

    Args:
        value: The complex number string to convert.

    Returns:
        The converted complex value.

    Raises:
        ValueError: If the complex number is invalid.
    """
    try:
        # Handle pure imaginary numbers
        if value.endswith("j") or value.endswith("J"):
            if value in ("j", "J"):
                return complex(0, 1)
            if value in ("+j", "+J"):
                return complex(0, 1)
            if value in ("-j", "-J"):
                return complex(0, -1)
            if "+" not in value and "-" not in value[1:]:
                # Pure imaginary number
                imag = float(value[:-1] or "1")
                return complex(0, imag)

        return complex(value.replace(" ", ""))
    except ValueError as e:
        raise ValueError(f"Invalid complex number: {value}") from e


def convert_type(
    value: float | complex,
    target_type: type,
    inferred_type: type,
) -> int | float | complex:
    """Convert value to target type if specified, otherwise use inferred type.

    Args:
        value: The value to convert.
        target_type: The requested target type.
        inferred_type: The inferred type from the value.

    Returns:
        The converted value.

    Raises:
        TypeError: If the conversion is not possible.
    """
    try:
        # If no specific type requested, use inferred type
        if target_type is float and inferred_type is complex:
            return value

        # Handle explicit type conversions
        if target_type is int and isinstance(value, complex):
            raise TypeError("Cannot convert complex number to int")
        return target_type(value)
    except (ValueError, TypeError) as e:
        raise TypeError(f"Cannot convert {value} to {target_type.__name__}") from e


def apply_bounds(
    value: float | complex,
    upper_bound: float | None = None,
    lower_bound: float | None = None,
) -> float | complex:
    """Apply bounds checking to numeric value.

    Args:
        value: The value to check.
        upper_bound: Maximum allowed value (inclusive).
        lower_bound: Minimum allowed value (inclusive).

    Returns:
        The validated value.

    Raises:
        ValueError: If the value is outside bounds.
    """
    if isinstance(value, complex):
        return value

    if upper_bound is not None and value > upper_bound:
        raise ValueError(f"Value {value} exceeds upper bound {upper_bound}")
    if lower_bound is not None and value < lower_bound:
        raise ValueError(f"Value {value} below lower bound {lower_bound}")
    return value


def apply_precision(
    value: float | complex,
    precision: int | None,
) -> float | complex:
    """Apply precision rounding to numeric value.

    Args:
        value: The value to round.
        precision: Number of decimal places.

    Returns:
        The rounded value.
    """
    if precision is None or isinstance(value, complex):
        return value
    if isinstance(value, float):
        return round(value, precision)
    return value


def parse_number(type_and_value: tuple[str, str]) -> float | complex:
    """Parse string to numeric value based on pattern type.

    Args:
        type_and_value: Tuple of (pattern_type, matched_value).

    Returns:
        The parsed numeric value.

    Raises:
        ValueError: If parsing fails.
    """
    num_type, value = type_and_value
    value = value.strip()

    try:
        if num_type == "special":
            return convert_special(value)

        if num_type == "percentage":
            return convert_percentage(value)

        if num_type == "fraction":
            if "/" not in value:
                raise ValueError(f"Invalid fraction: {value}")
            if value.count("/") > 1:
                raise ValueError(f"Invalid fraction: {value}")
            num, denom = value.split("/")
            if not (num.strip("-").isdigit() and denom.isdigit()):
                raise ValueError(f"Invalid fraction: {value}")
            denom_val = float(denom)
            if denom_val == 0:
                raise ValueError("Division by zero")
            return float(num) / denom_val
        if num_type in ("complex", "complex_sci", "pure_imaginary"):

            return convert_complex(value)
        if num_type == "scientific":

            if "e" not in value.lower():
                raise ValueError(f"Invalid scientific notation: {value}")
            parts = value.lower().split("e")
            if len(parts) != 2:
                raise ValueError(f"Invalid scientific notation: {value}")
            if not (parts[1].lstrip("+-").isdigit()):
                raise ValueError(f"Invalid scientific notation: {value}")
            return float(value)
        if num_type == "decimal":

            return float(value)

        raise ValueError(f"Unknown number type: {num_type}")
    except Exception as e:
        # Preserve the specific error type but wrap with more context
        raise type(e)(f"Failed to parse {value} as {num_type}: {str(e)}")


def to_num(
    input_: Any,
    /,
    *,
    upper_bound: int | float | None = None,
    lower_bound: int | float | None = None,
    num_type: NUM_TYPES = float,
    precision: int | None = None,
    num_count: int = 1,
) -> int | float | complex | list[int | float | complex]:
    """Convert input to numeric type(s) with validation and bounds checking.

    Args:
        input_value: The input to convert to number(s).
        upper_bound: Maximum allowed value (inclusive).
        lower_bound: Minimum allowed value (inclusive).
        num_type: Target numeric type ('int', 'float', 'complex' or type objects).
        precision: Number of decimal places for rounding (float only).
        num_count: Number of numeric values to extract.

    Returns:
        Converted number(s). Single value if num_count=1, else list.

    Raises:
        ValueError: For invalid input or out of bounds values.
        TypeError: For invalid input types or invalid type conversions.
    """
    # Validate input
    if isinstance(input_, (list, tuple)):
        raise TypeError("Input cannot be a sequence")

    # Handle boolean input
    if isinstance(input_, bool):
        return validate_num_type(num_type)(input_)

    # Handle direct numeric input
    if isinstance(input_, (int, float, complex, Decimal)):
        inferred_type = type(input_)
        if isinstance(input_, Decimal):
            inferred_type = float
        value = float(input_) if not isinstance(input_, complex) else input_
        value = apply_bounds(value, upper_bound, lower_bound)
        value = apply_precision(value, precision)
        return convert_type(value, validate_num_type(num_type), inferred_type)

    # Convert input to string and extract numbers
    input_str = str(input_)
    number_matches = extract_numbers(input_str)

    if not number_matches:
        raise ValueError(f"No valid numbers found in: {input_str}")

    # Process numbers
    results = []
    target_type = validate_num_type(num_type)

    number_matches = (
        number_matches[:num_count]
        if num_count < len(number_matches)
        else number_matches
    )

    for type_and_value in number_matches:
        try:
            # Infer appropriate type
            inferred_type = infer_type(type_and_value)

            # Parse to numeric value
            value = parse_number(type_and_value)

            # Apply bounds if not complex
            value = apply_bounds(value, upper_bound, lower_bound)

            # Apply precision
            value = apply_precision(value, precision)

            # Convert to target type if different from inferred
            value = convert_type(value, target_type, inferred_type)

            results.append(value)

        except Exception as e:
            if len(type_and_value) == 2:
                raise type(e)(f"Error processing {type_and_value[1]}: {str(e)}")
            raise type(e)(f"Error processing {type_and_value}: {str(e)}")

    if results and num_count == 1:
        return results[0]
    return results


def _serialize_as(
    input_,
    /,
    *,
    serialize_as: Literal["json", "xml"],
    strip_lower: bool = False,
    chars: str | None = None,
    str_type: Literal["json", "xml"] | None = None,
    use_model_dump: bool = False,
    str_parser: Callable[[str], dict[str, Any]] | None = None,
    parser_kwargs: dict = {},
    **kwargs: Any,
) -> str:
    try:
        dict_ = to_dict(
            input_,
            use_model_dump=use_model_dump,
            str_type=str_type,
            suppress=True,
            parser=str_parser,
            **parser_kwargs,
        )
        if any((str_type, chars)):
            str_ = json.dumps(dict_)
            str_ = _process_string(str_, strip_lower=strip_lower, chars=chars)
            dict_ = json.loads(str_)

        if serialize_as == "json":
            return json.dumps(dict_, **kwargs)

        if serialize_as == "xml":

            return dict_to_xml(dict_, **kwargs)
    except Exception as e:
        raise ValueError(
            f"Failed to serialize input of {type(input_).__name__} "
            f"into <{str_type}>"
        ) from e


def _to_str_type(input_: Any, /) -> str:
    if input_ in [set(), [], {}]:
        return ""

    if isinstance(input_, type(None) | UndefinedType | PydanticUndefinedType):
        return ""

    if isinstance(input_, bytes | bytearray):
        return input_.decode("utf-8", errors="replace")

    if isinstance(input_, str):
        return input_

    if isinstance(input_, Mapping):
        return json.dumps(dict(input_))

    try:
        return str(input_)
    except Exception as e:
        raise ValueError(
            f"Could not convert input of type <{type(input_).__name__}> " "to string"
        ) from e


def to_str(
    input_: Any,
    /,
    *,
    strip_lower: bool = False,
    chars: str | None = None,
    str_type: Literal["json", "xml"] | None = None,
    serialize_as: Literal["json", "xml"] | None = None,
    use_model_dump: bool = False,
    str_parser: Callable[[str], dict[str, Any]] | None = None,
    parser_kwargs: dict = {},
    **kwargs: Any,
) -> str:
    """Convert any input to its string representation.

    Handles various input types, with options for serialization and formatting.

    Args:
        input_: The input to convert to a string.
        strip_lower: If True, strip whitespace and convert to lowercase.
        chars: Specific characters to strip from the result.
        str_type: Type of string input ("json" or "xml") if applicable.
        serialize_as: Output serialization format ("json" or "xml").
        use_model_dump: Use model_dump for Pydantic models if available.
        str_parser: Custom parser function for string inputs.
        parser_kwargs: Additional keyword arguments for the parser.
        **kwargs: Additional arguments passed to json.dumps or serialization.

    Returns:
        str: The string representation of the input.

    Raises:
        ValueError: If serialization or conversion fails.

    Examples:
        >>> to_str(123)
        '123'
        >>> to_str("  HELLO  ", strip_lower=True)
        'hello'
        >>> to_str({"a": 1}, serialize_as="json")
        '{"a": 1}'
        >>> to_str({"a": 1}, serialize_as="xml")
        '<root><a>1</a></root>'
    """

    if serialize_as:
        return _serialize_as(
            input_,
            serialize_as=serialize_as,
            strip_lower=strip_lower,
            chars=chars,
            str_type=str_type,
            use_model_dump=use_model_dump,
            str_parser=str_parser,
            parser_kwargs=parser_kwargs,
            **kwargs,
        )

    str_ = _to_str_type(input_, **kwargs)
    if any((strip_lower, chars)):
        str_ = _process_string(str_, strip_lower=strip_lower, chars=chars)
    return str_


def _process_string(s: str, strip_lower: bool, chars: str | None) -> str:
    if s in [UNDEFINED, PydanticUndefined, None, [], {}]:
        return ""

    if strip_lower:
        s = s.lower()
        s = s.strip(chars) if chars is not None else s.strip()
    return s


def strip_lower(
    input_: Any,
    /,
    *,
    chars: str | None = None,
    str_type: Literal["json", "xml"] | None = None,
    serialize_as: Literal["json", "xml"] | None = None,
    use_model_dump: bool = False,
    str_parser: Callable[[str], dict[str, Any]] | None = None,
    parser_kwargs: dict = {},
    **kwargs: Any,
) -> str:
    """
    Convert input to stripped and lowercase string representation.

    This function is a convenience wrapper around to_str that always
    applies stripping and lowercasing.

    Args:
        input_: The input to convert to a string.
        use_model_dump: If True, use model_dump for Pydantic models.
        chars: Characters to strip from the result.
        **kwargs: Additional arguments to pass to to_str.

    Returns:
        Stripped and lowercase string representation of the input.

    Raises:
        ValueError: If conversion fails.

    Example:
        >>> strip_lower("  HELLO WORLD  ")
        'hello world'
    """
    return to_str(
        input_,
        strip_lower=True,
        chars=chars,
        str_type=str_type,
        serialize_as=serialize_as,
        use_model_dump=use_model_dump,
        str_parser=str_parser,
        parser_kwargs=parser_kwargs,
        **kwargs,
    )


def validate_boolean(x: Any, /) -> bool:
    """
    Forcefully validate and convert the input into a boolean value.

    This function attempts to convert various input types to a boolean value.
    It recognizes common string representations of true and false, as well
    as numeric values. The conversion is case-insensitive.

    Args:
        x: The input to be converted to boolean. Can be:
           - Boolean: returned as-is
           - Number (including complex): converted using Python's bool rules
           - String: converted based on common boolean representations
           - None: raises TypeError
           - Other types: converted to string and then evaluated

    Returns:
        bool: The boolean representation of the input.

    Raises:
        ValueError: If the input cannot be unambiguously converted to a boolean value.
        TypeError: If the input type is unsupported or None.

    Examples:
        >>> validate_boolean(True)
        True
        >>> validate_boolean("yes")
        True
        >>> validate_boolean("OFF")
        False
        >>> validate_boolean(1)
        True
        >>> validate_boolean(0j)
        False
        >>> validate_boolean(1 + 1j)
        True

    Notes:
        - String matching is case-insensitive
        - Leading/trailing whitespace is stripped
        - Numeric values follow Python's bool() rules
        - Complex numbers: bool(0j) is False, bool(any other complex) is True
        - None values raise TypeError
        - Empty strings raise ValueError
    """
    if x is None:
        raise TypeError("Cannot convert None to boolean")

    if isinstance(x, bool):
        return x

    # Handle all numeric types (including complex) using Python's bool
    if isinstance(x, (int, float, Complex)):
        return bool(x)

    # Convert to string if not already a string
    if not isinstance(x, str):
        try:
            x = str(x)
        except Exception as e:
            raise TypeError(f"Cannot convert {type(x)} to boolean: {str(e)}")

    # Handle string inputs
    x_cleaned = str(x).strip().lower()

    if not x_cleaned:
        raise ValueError("Cannot convert empty string to boolean")

    if x_cleaned in TRUE_VALUES:
        return True

    if x_cleaned in FALSE_VALUES:
        return False

    # Try numeric conversion as a last resort
    try:
        # Try to evaluate as a literal if it looks like a complex number
        if "j" in x_cleaned:
            try:
                return bool(complex(x_cleaned))
            except ValueError:
                pass
        return bool(float(x_cleaned))
    except ValueError:
        pass

    raise ValueError(
        f"Cannot convert '{x}' to boolean. Valid true values are: {sorted(TRUE_VALUES)}, "
        f"valid false values are: {sorted(FALSE_VALUES)}"
    )


def validate_keys(
    d_: dict[str, Any],
    keys: Sequence[str] | KeysDict,
    /,
    *,
    similarity_algo: SIMILARITY_TYPE | Callable[[str, str], float] = "jaro_winkler",
    similarity_threshold: float = 0.85,
    fuzzy_match: bool = True,
    handle_unmatched: Literal["ignore", "raise", "remove", "fill", "force"] = "ignore",
    fill_value: Any = None,
    fill_mapping: dict[str, Any] | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    """
    Validate and correct dictionary keys based on expected keys using string similarity.

    Args:
        d_: The dictionary to validate and correct keys for.
        keys: List of expected keys or dictionary mapping keys to types.
        similarity_algo: String similarity algorithm to use or custom function.
        similarity_threshold: Minimum similarity score for fuzzy matching.
        fuzzy_match: If True, use fuzzy matching for key correction.
        handle_unmatched: Specifies how to handle unmatched keys:
            - "ignore": Keep unmatched keys in output.
            - "raise": Raise ValueError if unmatched keys exist.
            - "remove": Remove unmatched keys from output.
            - "fill": Fill unmatched keys with default value/mapping.
            - "force": Combine "fill" and "remove" behaviors.
        fill_value: Default value for filling unmatched keys.
        fill_mapping: Dictionary mapping unmatched keys to default values.
        strict: If True, raise ValueError if any expected key is missing.

    Returns:
        A new dictionary with validated and corrected keys.

    Raises:
        ValueError: If validation fails based on specified parameters.
        TypeError: If input types are invalid.
        AttributeError: If key validation fails.
    """
    # Input validation
    if not isinstance(d_, dict):
        raise TypeError("First argument must be a dictionary")
    if keys is None:
        raise TypeError("Keys argument cannot be None")
    if not 0.0 <= similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold must be between 0.0 and 1.0")

    # Extract expected keys
    fields_set = set(keys) if isinstance(keys, list) else set(keys.keys())
    if not fields_set:
        return d_.copy()  # Return copy of original if no expected keys

    # Initialize output dictionary and tracking sets
    corrected_out = {}
    matched_expected = set()
    matched_input = set()

    # Get similarity function
    if isinstance(similarity_algo, str):
        if similarity_algo not in SIMILARITY_ALGO_MAP:
            raise ValueError(f"Unknown similarity algorithm: {similarity_algo}")
        similarity_func = SIMILARITY_ALGO_MAP[similarity_algo]
    else:
        similarity_func = similarity_algo

    # First pass: exact matches
    for key in d_:
        if key in fields_set:
            corrected_out[key] = d_[key]
            matched_expected.add(key)
            matched_input.add(key)

    # Second pass: fuzzy matching if enabled
    if fuzzy_match:
        remaining_input = set(d_.keys()) - matched_input
        remaining_expected = fields_set - matched_expected

        for key in remaining_input:
            if not remaining_expected:
                break

            matches = string_similarity(
                key,
                list(remaining_expected),
                algorithm=similarity_func,
                threshold=similarity_threshold,
                return_most_similar=True,
            )

            if matches:
                match = matches
                corrected_out[match] = d_[key]
                matched_expected.add(match)
                matched_input.add(key)
                remaining_expected.remove(match)
            elif handle_unmatched == "ignore":
                corrected_out[key] = d_[key]

    # Handle unmatched keys based on handle_unmatched parameter
    unmatched_input = set(d_.keys()) - matched_input
    unmatched_expected = fields_set - matched_expected

    if handle_unmatched == "raise" and unmatched_input:
        raise ValueError(f"Unmatched keys found: {unmatched_input}")

    elif handle_unmatched == "ignore":
        for key in unmatched_input:
            corrected_out[key] = d_[key]

    elif handle_unmatched in ("fill", "force"):
        # Fill missing expected keys
        for key in unmatched_expected:
            if fill_mapping and key in fill_mapping:
                corrected_out[key] = fill_mapping[key]
            else:
                corrected_out[key] = fill_value

        # For "fill" mode, also keep unmatched original keys
        if handle_unmatched == "fill":
            for key in unmatched_input:
                corrected_out[key] = d_[key]

    # Check strict mode
    if strict and unmatched_expected:
        raise ValueError(f"Missing required keys: {unmatched_expected}")

    return corrected_out


def validate_mapping(
    d: Any,
    keys: Sequence[str] | KeysDict,
    /,
    *,
    similarity_algo: SIMILARITY_TYPE | Callable[[str, str], float] = "jaro_winkler",
    similarity_threshold: float = 0.85,
    fuzzy_match: bool = True,
    handle_unmatched: Literal["ignore", "raise", "remove", "fill", "force"] = "ignore",
    fill_value: Any = None,
    fill_mapping: dict[str, Any] | None = None,
    strict: bool = False,
    suppress_conversion_errors: bool = False,
) -> dict[str, Any]:
    """
    Validate and correct any input into a dictionary with expected keys.

    Args:
        d: Input to validate. Can be:
            - Dictionary
            - JSON string or markdown code block
            - XML string
            - Object with to_dict/model_dump method
            - Any type convertible to dictionary
        keys: List of expected keys or dictionary mapping keys to types.
        similarity_algo: String similarity algorithm or custom function.
        similarity_threshold: Minimum similarity score for fuzzy matching.
        fuzzy_match: If True, use fuzzy matching for key correction.
        handle_unmatched: How to handle unmatched keys:
            - "ignore": Keep unmatched keys
            - "raise": Raise error for unmatched keys
            - "remove": Remove unmatched keys
            - "fill": Fill missing keys with default values
            - "force": Combine "fill" and "remove" behaviors
        fill_value: Default value for filling unmatched keys.
        fill_mapping: Dictionary mapping keys to default values.
        strict: Raise error if any expected key is missing.
        suppress_conversion_errors: Return empty dict on conversion errors.

    Returns:
        Validated and corrected dictionary.

    Raises:
        ValueError: If input cannot be converted or validation fails.
        TypeError: If input types are invalid.
    """
    if d is None:
        raise TypeError("Input cannot be None")

    # Try converting to dictionary
    try:
        if isinstance(d, str):
            # First try to_json for JSON strings and code blocks
            try:
                json_result = to_json(d)
                dict_input = (
                    json_result[0] if isinstance(json_result, list) else json_result
                )
            except Exception:
                # Fall back to to_dict for other string formats
                dict_input = to_dict(
                    d, str_type="json", fuzzy_parse=True, suppress=True
                )
        else:
            dict_input = to_dict(
                d, use_model_dump=True, fuzzy_parse=True, suppress=True
            )

        if not isinstance(dict_input, dict):
            if suppress_conversion_errors:
                dict_input = {}
            else:
                raise ValueError(
                    f"Failed to convert input to dictionary: {type(dict_input)}"
                )

    except Exception as e:
        if suppress_conversion_errors:
            dict_input = {}
        else:
            raise ValueError(f"Failed to convert input to dictionary: {e}")

    # Validate the dictionary
    return validate_keys(
        dict_input,
        keys,
        similarity_algo=similarity_algo,
        similarity_threshold=similarity_threshold,
        fuzzy_match=fuzzy_match,
        handle_unmatched=handle_unmatched,
        fill_value=fill_value,
        fill_mapping=fill_mapping,
        strict=strict,
    )


def xml_to_dict(
    xml_string: str,
    /,
    suppress=False,
    remove_root: bool = True,
    root_tag: str = None,
) -> dict[str, Any]:
    """
    Parse an XML string into a nested dictionary structure.

    This function converts an XML string into a dictionary where:
    - Element tags become dictionary keys
    - Text content is assigned directly to the tag key if there are no children
    - Attributes are stored in a '@attributes' key
    - Multiple child elements with the same tag are stored as lists

    Args:
        xml_string: The XML string to parse.

    Returns:
        A dictionary representation of the XML structure.

    Raises:
        ValueError: If the XML is malformed or parsing fails.
    """
    try:
        a = XMLParser(xml_string).parse()
        if remove_root and (root_tag or "root") in a:
            a = a[root_tag or "root"]
        return a
    except ValueError as e:
        if not suppress:
            raise e


def dict_to_xml(data: dict, /, root_tag: str = "root") -> str:

    root = ET.Element(root_tag)

    def convert(dict_obj: dict, parent: Any) -> None:
        for key, val in dict_obj.items():
            if isinstance(val, dict):
                element = ET.SubElement(parent, key)
                convert(dict_obj=val, parent=element)
            else:
                element = ET.SubElement(parent, key)
                element.text = str(object=val)

    convert(dict_obj=data, parent=root)
    return ET.tostring(root, encoding="unicode")


class XMLParser:
    def __init__(self, xml_string: str):
        self.xml_string = xml_string.strip()
        self.index = 0

    def parse(self) -> dict[str, Any]:
        """Parse the XML string and return the root element as a dictionary."""
        return self._parse_element()

    def _parse_element(self) -> dict[str, Any]:
        """Parse a single XML element and its children."""
        self._skip_whitespace()
        if self.xml_string[self.index] != "<":
            raise ValueError(f"Expected '<', found '{self.xml_string[self.index]}'")

        tag, attributes = self._parse_opening_tag()
        children: dict[str, str | list | dict] = {}
        text = ""

        while self.index < len(self.xml_string):
            self._skip_whitespace()
            if self.xml_string.startswith("</", self.index):
                closing_tag = self._parse_closing_tag()
                if closing_tag != tag:
                    raise ValueError(f"Mismatched tags: '{tag}' and '{closing_tag}'")
                break
            elif self.xml_string.startswith("<", self.index):
                child = self._parse_element()
                child_tag, child_data = next(iter(child.items()))
                if child_tag in children:
                    if not isinstance(children[child_tag], list):
                        children[child_tag] = [children[child_tag]]
                    children[child_tag].append(child_data)
                else:
                    children[child_tag] = child_data
            else:
                text += self._parse_text()

        result: dict[str, Any] = {}
        if attributes:
            result["@attributes"] = attributes
        if children:
            result.update(children)
        elif text.strip():
            result = text.strip()

        return {tag: result}

    def _parse_opening_tag(self) -> tuple[str, dict[str, str]]:
        """Parse an opening XML tag and its attributes."""
        match = re.match(
            r'<(\w+)((?:\s+\w+="[^"]*")*)\s*/?>',
            self.xml_string[self.index :],  # noqa
        )
        if not match:
            raise ValueError("Invalid opening tag")
        self.index += match.end()
        tag = match.group(1)
        attributes = dict(re.findall(r'(\w+)="([^"]*)"', match.group(2)))
        return tag, attributes

    def _parse_closing_tag(self) -> str:
        """Parse a closing XML tag."""
        match = re.match(r"</(\w+)>", self.xml_string[self.index :])  # noqa
        if not match:
            raise ValueError("Invalid closing tag")
        self.index += match.end()
        return match.group(1)

    def _parse_text(self) -> str:
        """Parse text content between XML tags."""
        start = self.index
        while self.index < len(self.xml_string) and self.xml_string[self.index] != "<":
            self.index += 1
        return self.xml_string[start : self.index]  # noqa

    def _skip_whitespace(self) -> None:
        """Skip any whitespace characters at the current parsing position."""
        p_ = len(self.xml_string[self.index :])  # noqa
        m_ = len(self.xml_string[self.index :].lstrip())  # noqa

        self.index += p_ - m_


@overload
def flatten(
    nested_structure: T,
    /,
    *,
    parent_key: tuple = (),
    sep: str = "|",
    coerce_keys: Literal[True] = True,
    dynamic: bool = True,
    coerce_sequence: Literal["dict", None] = None,
    max_depth: int | None = None,
) -> dict[str, Any] | None: ...


@overload
def flatten(
    nested_structure: T,
    /,
    *,
    parent_key: tuple = (),
    sep: str = "|",
    coerce_keys: Literal[False],
    dynamic: bool = True,
    coerce_sequence: Literal["dict", "list", None] = None,
    max_depth: int | None = None,
) -> dict[tuple, Any] | None: ...


def flatten(
    nested_structure: Any,
    /,
    *,
    parent_key: tuple = (),
    sep: str = "|",
    coerce_keys: bool = True,
    dynamic: bool = True,
    coerce_sequence: Literal["dict", "list"] | None = None,
    max_depth: int | None = None,
) -> dict[tuple | str, Any] | None:
    """Flatten a nested structure into a single-level dictionary.

    Recursively traverses the input, creating keys that represent the path
    to each value in the flattened result.

    Args:
        nested_structure: The nested structure to flatten.
        parent_key: Base key for the current recursion level. Default: ().
        sep: Separator for joining keys. Default: "|".
        coerce_keys: Join keys into strings if True, keep as tuples if False.
            Default: True.
        dynamic: Handle sequences (except strings) dynamically if True.
            Default: True.
        coerce_sequence: Force sequences to be treated as dicts or lists.
            Options: "dict", "list", or None. Default: None.
        max_depth: Maximum depth to flatten. None for complete flattening.
            Default: None.

    Returns:
        A flattened dictionary with keys as tuples or strings (based on
        coerce_keys) representing the path to each value.

    Raises:
        ValueError: If coerce_sequence is "list" and coerce_keys is True.

    Example:
        >>> nested = {"a": 1, "b": {"c": 2, "d": [3, 4]}}
        >>> flatten(nested)
        {'a': 1, 'b|c': 2, 'b|d|0': 3, 'b|d|1': 4}

    Note:
        - Preserves order of keys in dicts and indices in sequences.
        - With dynamic=True, treats sequences (except strings) as nestable.
        - coerce_sequence allows forcing sequence handling for homogeneity.
    """

    if coerce_keys and coerce_sequence == "list":
        raise ValueError("coerce_sequence cannot be 'list' when coerce_keys is True")

    coerce_sequence_to_list = None
    coerce_sequence_to_dict = None

    if dynamic and coerce_sequence:
        if coerce_sequence == "dict":
            coerce_sequence_to_dict = True
        elif coerce_sequence == "list":
            coerce_sequence_to_list = True

    return _flatten_iterative(
        obj=nested_structure,
        parent_key=parent_key,
        sep=sep,
        coerce_keys=coerce_keys,
        dynamic=dynamic,
        coerce_sequence_to_list=coerce_sequence_to_list,
        coerce_sequence_to_dict=coerce_sequence_to_dict,
        max_depth=max_depth,
    )


def _flatten_iterative(
    obj: Any,
    parent_key: tuple,
    sep: str,
    coerce_keys: bool,
    dynamic: bool,
    coerce_sequence_to_list: bool = False,
    coerce_sequence_to_dict: bool = False,
    max_depth: int | None = None,
) -> dict[tuple | str, Any]:
    stack = deque([(obj, parent_key, 0)])
    result = {}

    while stack:
        current_obj, current_key, depth = stack.pop()

        if max_depth is not None and depth >= max_depth:
            result[_format_key(current_key, sep, coerce_keys)] = current_obj
            continue

        if isinstance(current_obj, Mapping):
            for k, v in current_obj.items():
                new_key = current_key + (k,)
                if (
                    v
                    and isinstance(v, (Mapping, Sequence))
                    and not isinstance(v, (str, bytes, bytearray))
                ):
                    stack.appendleft((v, new_key, depth + 1))
                else:
                    result[_format_key(new_key, sep, coerce_keys)] = v

        elif (
            dynamic
            and isinstance(current_obj, Sequence)
            and not isinstance(current_obj, (str, bytes, bytearray))
        ):
            if coerce_sequence_to_dict:
                dict_obj = {str(i): v for i, v in enumerate(current_obj)}
                for k, v in dict_obj.items():
                    new_key = current_key + (k,)
                    stack.appendleft((v, new_key, depth + 1))
            elif coerce_sequence_to_list:
                for i, v in enumerate(current_obj):
                    new_key = current_key + (i,)
                    stack.appendleft((v, new_key, depth + 1))
            else:
                for i, v in enumerate(current_obj):
                    new_key = current_key + (str(i),)
                    stack.appendleft((v, new_key, depth + 1))
        else:
            result[_format_key(current_key, sep, coerce_keys)] = current_obj

    return result


def _format_key(key: tuple, sep: str, coerce_keys: bool, /) -> tuple | str:
    if not key:
        return key
    return sep.join(map(str, key)) if coerce_keys else key


def nfilter(
    nested_structure: dict[Any, Any] | list[Any],
    /,
    condition: Callable[[Any], bool],
) -> dict[Any, Any] | list[Any]:
    """Filter elements in a nested structure based on a condition.

    Args:
        nested_structure: The nested structure (dict or list) to filter.
        condition: Function returning True for elements to keep, False to
            discard.

    Returns:
        The filtered nested structure.

    Raises:
        TypeError: If nested_structure is not a dict or list.

    Example:
        >>> data = {"a": 1, "b": {"c": 2, "d": 3}, "e": [4, 5, 6]}
        >>> nfilter(data, lambda x: isinstance(x, int) and x > 2)
        {'b': {'d': 3}, 'e': [4, 5, 6]}
    """
    if isinstance(nested_structure, dict):
        return _filter_dict(nested_structure, condition)
    elif isinstance(nested_structure, list):
        return _filter_list(nested_structure, condition)
    else:
        raise TypeError("The nested_structure must be either a dict or a list.")


def _filter_dict(
    dictionary: dict[Any, Any], condition: Callable[[tuple[Any, Any]], bool]
) -> dict[Any, Any]:
    return {
        k: nfilter(v, condition) if isinstance(v, dict | list) else v
        for k, v in dictionary.items()
        if condition(v) or isinstance(v, dict | list)
    }


def _filter_list(lst: list[Any], condition: Callable[[Any], bool]) -> list[Any]:
    return [
        nfilter(item, condition) if isinstance(item, dict | list) else item
        for item in lst
        if condition(item) or isinstance(item, dict | list)
    ]


def nget(
    nested_structure: dict[Any, Any] | list[Any],
    /,
    indices: list[int | str],
    default: Any = UNDEFINED,
) -> Any:
    try:
        target_container = get_target_container(nested_structure, indices[:-1])
        last_index = indices[-1]

        if (
            isinstance(target_container, list)
            and isinstance(last_index, int)
            and last_index < len(target_container)
        ):
            return target_container[last_index]
        elif isinstance(target_container, dict) and last_index in target_container:
            return target_container[last_index]
        elif default is not UNDEFINED:
            return default
        else:
            raise LookupError("Target not found and no default value provided.")
    except (IndexError, KeyError, TypeError):
        if default is not UNDEFINED:
            return default
        else:
            raise LookupError("Target not found and no default value provided.")


def nget(
    nested_structure: dict[Any, Any] | list[Any],
    /,
    indices: list[int | str],
    default: Any = UNDEFINED,
) -> Any:
    try:
        target_container = get_target_container(nested_structure, indices[:-1])
        last_index = indices[-1]

        if (
            isinstance(target_container, list)
            and isinstance(last_index, int)
            and last_index < len(target_container)
        ):
            return target_container[last_index]
        elif isinstance(target_container, dict) and last_index in target_container:
            return target_container[last_index]
        elif default is not UNDEFINED:
            return default
        else:
            raise LookupError("Target not found and no default value provided.")
    except (IndexError, KeyError, TypeError):
        if default is not UNDEFINED:
            return default
        else:
            raise LookupError("Target not found and no default value provided.")


def ninsert(
    nested_structure: dict[Any, Any] | list[Any],
    /,
    indices: list[str | int],
    value: Any,
    *,
    current_depth: int = 0,
) -> None:
    """
    Inserts a value into a nested structure at a specified path.

    Navigates a nested dictionary or list based on a sequence of indices or
    keys and inserts `value` at the final location. This method can create
    intermediate dictionaries or lists as needed.

    Args:
        nested_structure: The nested structure to modify.
        indices: The sequence of keys or indices defining the insertion path.
        value: The value to insert at the specified location.
        current_depth: Internal use only; tracks the current depth during
            recursive calls.

    Raises:
        ValueError: If the indices list is empty.
        TypeError: If an invalid key or container type is encountered.

    Examples:
        >>> subject_ = {'a': {'b': [1, 2]}}
        >>> ninsert(subject_, ['a', 'b', 2], 3)
        >>> assert subject_ == {'a': {'b': [1, 2, 3]}}

        >>> subject_ = []
        >>> ninsert(subject_, [0, 'a'], 1)
        >>> assert subject_ == [{'a': 1}]
    """
    if not indices:
        raise ValueError("Indices list cannot be empty")

    indices = to_list(indices)
    for i, part in enumerate(indices[:-1]):
        if isinstance(part, int):
            if isinstance(nested_structure, dict):
                raise TypeError(
                    f"Unsupported key type: {type(part).__name__}."
                    "Only string keys are acceptable.",
                )
            while len(nested_structure) <= part:
                nested_structure.append(None)
            if nested_structure[part] is None or not isinstance(
                nested_structure[part], (dict, list)
            ):
                next_part = indices[i + 1]
                nested_structure[part] = [] if isinstance(next_part, int) else {}
        elif isinstance(nested_structure, dict):
            if part is None:
                raise TypeError("Cannot use NoneType as a key in a dictionary")
            if isinstance(part, (float, complex)):
                raise TypeError(
                    f"Unsupported key type: {type(part).__name__}."
                    "Only string keys are acceptable.",
                )
            if part not in nested_structure:
                next_part = indices[i + 1]
                nested_structure[part] = [] if isinstance(next_part, int) else {}
        else:
            raise TypeError(
                f"Invalid container type: {type(nested_structure)} "
                "encountered during insertion"
            )

        nested_structure = nested_structure[part]
        current_depth += 1

    last_part = indices[-1]
    if isinstance(last_part, int):
        if isinstance(nested_structure, dict):
            raise TypeError(
                f"Unsupported key type: {type(last_part).__name__}."
                "Only string keys are acceptable.",
            )
        while len(nested_structure) <= last_part:
            nested_structure.append(None)
        nested_structure[last_part] = value
    elif isinstance(nested_structure, list):
        raise TypeError("Cannot use non-integer index on a list")
    else:
        if last_part is None:
            raise TypeError("Cannot use NoneType as a key in a dictionary")
        if isinstance(last_part, (float, complex)):
            raise TypeError(
                f"Unsupported key type: {type(last_part).__name__}."
                "Only string keys are acceptable.",
            )
        nested_structure[last_part] = value


from collections import defaultdict
from collections.abc import Callable, Sequence
from itertools import chain
from typing import Any


def nmerge(
    nested_structure: Sequence[dict[str, Any] | list[Any]],
    /,
    *,
    overwrite: bool = False,
    dict_sequence: bool = False,
    sort_list: bool = False,
    custom_sort: Callable[[Any], Any] | None = None,
) -> dict[str, Any] | list[Any]:
    """
    Merge multiple dictionaries, lists, or sequences into a unified structure.

    Args:
        nested_structure: A sequence containing dictionaries, lists, or other
            iterable objects to merge.
        overwrite: If True, overwrite existing keys in dictionaries with
            those from subsequent dictionaries.
        dict_sequence: Enables unique key generation for duplicate keys by
            appending a sequence number. Applicable only if `overwrite` is
            False.
        sort_list: When True, sort the resulting list after merging. It does
            not affect dictionaries.
        custom_sort: An optional callable that defines custom sorting logic
            for the merged list.

    Returns:
        A merged dictionary or list, depending on the types present in
        `nested_structure`.

    Raises:
        TypeError: If `nested_structure` contains objects of incompatible
            types that cannot be merged.
    """
    if not isinstance(nested_structure, list):
        raise TypeError("Please input a list")
    if is_homogeneous(nested_structure, dict):
        return _merge_dicts(nested_structure, overwrite, dict_sequence)
    elif is_homogeneous(nested_structure, list):
        return _merge_sequences(nested_structure, sort_list, custom_sort)
    else:
        raise TypeError(
            "All items in the input list must be of the same type, "
            "either dict, list, or Iterable."
        )


def _deep_merge_dicts(dict1: dict[str, Any], dict2: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively merges two dictionaries, combining values where keys overlap.

    Args:
        dict1: The first dictionary.
        dict2: The second dictionary.

    Returns:
        The merged dictionary.
    """
    for key in dict2:
        if key in dict1:
            if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                _deep_merge_dicts(dict1[key], dict2[key])
            else:
                if not isinstance(dict1[key], list):
                    dict1[key] = [dict1[key]]
                dict1[key].append(dict2[key])
        else:
            dict1[key] = dict2[key]
    return dict1


def _merge_dicts(
    iterables: list[dict[str, Any]],
    dict_update: bool,
    dict_sequence: bool,
) -> dict[str, Any]:
    """
    Merges a list of dictionaries into a single dictionary, with options for
    handling duplicate keys and sequences.

    Args:
        iterables: A list of dictionaries to merge.
        dict_update: If True, overwrite existing keys in dictionaries
            with those from subsequent dictionaries.
        dict_sequence: Enables unique key generation for duplicate keys
            by appending a sequence number

    Returns:
        The merged dictionary.
    """
    merged_dict = {}  # {'a': [1, 2]}
    sequence_counters = defaultdict(int)
    list_values = {}

    for d in iterables:  # [{'a': [1, 2]}, {'a': [3, 4]}]
        for key, value in d.items():  # {'a': [3, 4]}
            if key not in merged_dict or dict_update:
                if (
                    key in merged_dict
                    and isinstance(merged_dict[key], dict)
                    and isinstance(value, dict)
                ):
                    _deep_merge_dicts(merged_dict[key], value)
                else:
                    merged_dict[key] = value  # {'a': [1, 2]}
                    if isinstance(value, list):
                        list_values[key] = True
            elif dict_sequence:
                sequence_counters[key] += 1
                new_key = f"{key}{sequence_counters[key]}"
                merged_dict[new_key] = value
            else:
                if not isinstance(merged_dict[key], list) or list_values.get(
                    key, False
                ):
                    merged_dict[key] = [merged_dict[key]]
                merged_dict[key].append(value)

    return merged_dict


def _merge_sequences(
    iterables: list[list[Any]],
    sort_list: bool,
    custom_sort: Callable[[Any], Any] | None = None,
) -> list[Any]:
    """
    Merges a list of lists into a single list, with options for sorting and
    custom sorting logic.

    Args:
        iterables: A list of lists to merge.
        sort_list: When True, sort the resulting list after merging.
        custom_sort: An optional callable that defines custom sorting logic
            for the merged list.

    Returns:
        The merged list.
    """
    merged_list = list(chain(*iterables))
    if sort_list:
        if custom_sort:
            return sorted(merged_list, key=custom_sort)
        else:
            return sorted(merged_list, key=lambda x: (isinstance(x, str), x))
    return merged_list


def npop(
    input_: dict[str, Any] | list[Any],
    /,
    indices: str | int | Sequence[str | int],
    default: Any = UNDEFINED,
) -> Any:
    """
    Perform a nested pop operation on the input structure.

    This function navigates through the nested structure using the provided
    indices and removes and returns the value at the final location.

    Args:
        input_: The input nested structure (dict or list) to pop from.
        indices: A single index or a sequence of indices to navigate the
            nested structure.
        default: The value to return if the key is not found. If not
            provided, a KeyError will be raised.

    Returns:
        The value at the specified nested location.

    Raises:
        ValueError: If the indices list is empty.
        KeyError: If a key is not found in a dictionary.
        IndexError: If an index is out of range for a list.
        TypeError: If an operation is not supported on the current data type.
    """
    if not indices:
        raise ValueError("Indices list cannot be empty")

    indices = to_list(indices)

    current = input_
    for key in indices[:-1]:
        if isinstance(current, dict):
            if current.get(key):
                current = current[key]
            else:
                raise KeyError(f"{key} is not found in {current}")
        elif isinstance(current, list) and isinstance(key, int):
            if key >= len(current):
                raise KeyError(f"{key} exceeds the length of the list {current}")
            elif key < 0:
                raise ValueError("list index cannot be negative")
            current = current[key]

    last_key = indices[-1]
    try:
        return current.pop(
            last_key,
        )
    except Exception as e:
        if default is not UNDEFINED:
            return default
        else:
            raise KeyError(f"Invalid npop. Error: {e}")


def nset(
    nested_structure: dict[str, Any] | list[Any],
    /,
    indices: str | int | Sequence[str | int],
    value: Any,
) -> None:
    """Set a value within a nested structure at the specified path.

    This method allows setting a value deep within a nested dictionary or list
    by specifying a path to the target location using a sequence of indices.
    Each index in the sequence represents a level in the nested structure,
    with integers used for list indices and strings for dictionary keys.

    Args:
        nested_structure: The nested structure to modify.
        indices: The path of indices leading to the target location.
        value: The value to set at the specified location.

    Raises:
        ValueError: If the indices sequence is empty.
        TypeError: If the target container is not a list or dictionary,
                   or if the index type is incorrect.

    Examples:
        >>> data = {'a': {'b': [10, 20]}}
        >>> nset(data, ['a', 'b', 1], 99)
        >>> assert data == {'a': {'b': [10, 99]}}

        >>> data = [0, [1, 2], 3]
        >>> nset(data, [1, 1], 99)
        >>> assert data == [0, [1, 99], 3]
    """

    if not indices:
        raise ValueError("Indices list is empty, cannot determine target container")

    _indices = to_list(indices)
    target_container = nested_structure

    for i, index in enumerate(_indices[:-1]):
        if isinstance(target_container, list):
            if not isinstance(index, int):
                raise TypeError("Cannot use non-integer index on a list")
            ensure_list_index(target_container, index)
            if target_container[index] is None:
                next_index = _indices[i + 1]
                target_container[index] = [] if isinstance(next_index, int) else {}
        elif isinstance(target_container, dict):
            if isinstance(index, int):
                raise TypeError(
                    f"Unsupported key type: {type(index).__name__}. "
                    "Only string keys are acceptable."
                )
            if index not in target_container:
                next_index = _indices[i + 1]
                target_container[index] = [] if isinstance(next_index, int) else {}
        else:
            raise TypeError("Target container is not a list or dictionary")

        target_container = target_container[index]

    last_index = _indices[-1]
    if isinstance(target_container, list):
        if not isinstance(last_index, int):
            raise TypeError("Cannot use non-integer index on a list")
        ensure_list_index(target_container, last_index)
        target_container[last_index] = value
    elif isinstance(target_container, dict):
        if not isinstance(last_index, str):
            raise TypeError(
                f"Unsupported key type: {type(last_index).__name__}. "
                "Only string keys are acceptable."
            )
        target_container[last_index] = value
    else:
        raise TypeError("Cannot set value on non-list/dict element")


def ensure_list_index(lst: list[Any], index: int, default: Any = UNDEFINED) -> None:
    while len(lst) <= index:
        lst.append(default if default is not UNDEFINED else None)


def unflatten(
    flat_dict: dict[str, Any], sep: str = "|", inplace: bool = False
) -> dict[str, Any] | list[Any]:
    """
    Unflatten a single-level dictionary into a nested dictionary or list.

    Args:
        flat_dict: The flattened dictionary to unflatten.
        sep: The separator used for joining keys.
        inplace: Whether to modify the input dictionary in place.

    Returns:
        The unflattened nested dictionary or list.

    Examples:
        >>> unflatten({"a|b|c": 1, "a|b|d": 2})
        {'a': {'b': {'c': 1, 'd': 2}}}

        >>> unflatten({"0": "a", "1": "b", "2": "c"})
        ['a', 'b', 'c']
    """

    def _unflatten(data: dict) -> dict | list:
        result = {}
        for key, value in data.items():
            parts = key.split(sep)
            current = result
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            if isinstance(value, dict):
                current[parts[-1]] = _unflatten(value)
            else:
                current[parts[-1]] = value

        # Convert dictionary to list if keys are consecutive integers
        if result and all(isinstance(key, str) and key.isdigit() for key in result):
            return [result[str(i)] for i in range(len(result))]
        return result

    if inplace:
        unflattened_dict = {}
        for key, value in flat_dict.items():
            parts = key.split(sep)
            current = unflattened_dict
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value

        unflattened_result = _unflatten(unflattened_dict)
        flat_dict.clear()
        if isinstance(unflattened_result, list):
            flat_dict.update({str(i): v for i, v in enumerate(unflattened_result)})
        else:
            flat_dict.update(unflattened_result)
        return flat_dict

    else:
        unflattened_dict = {}
        for key, value in flat_dict.items():
            parts = key.split(sep)
            current = unflattened_dict
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value

        return _unflatten(unflattened_dict)


def is_homogeneous(
    iterables: list[Any] | dict[Any, Any], type_check: type | tuple[type, ...]
) -> bool:
    """
    Check if all elements in a list or all values in a dict are of same type.

    Args:
        iterables: The list or dictionary to check.
        type_check: The type to check against.

    Returns:
        True if all elements/values are of the same type, False otherwise.
    """
    if isinstance(iterables, list):
        return all(isinstance(it, type_check) for it in iterables)

    elif isinstance(iterables, dict):
        return all(isinstance(val, type_check) for val in iterables.values())

    else:
        return isinstance(iterables, type_check)


def is_same_dtype(
    input_: list[Any] | dict[Any, Any],
    dtype: type | None = None,
    return_dtype: bool = False,
) -> bool | tuple[bool, type | None]:
    """
    Check if all elements in a list or dict values are of the same data type.

    Args:
        input_: The input list or dictionary to check.
        dtype: The data type to check against. If None, uses the type of the
            first element.
        return_dtype: If True, return the data type with the check result.

    Returns:
        If return_dtype is False, returns True if all elements are of the
        same type (or if the input is empty), False otherwise.
        If return_dtype is True, returns a tuple (bool, type | None).
    """
    if not input_:
        return True

    iterable = input_.values() if isinstance(input_, dict) else input_
    first_element_type = type(next(iter(iterable), None))

    dtype = dtype or first_element_type

    result = all(isinstance(element, dtype) for element in iterable)
    return (result, dtype) if return_dtype else result


def is_structure_homogeneous(
    structure: Any, return_structure_type: bool = False
) -> bool | tuple[bool, type | None]:
    """
    Check if a nested structure is homogeneous (no mix of lists and dicts).

    Args:
        structure: The nested structure to check.
        return_structure_type: If True, return the type of the homogeneous
            structure.

    Returns:
        If return_structure_type is False, returns True if the structure is
        homogeneous, False otherwise.
        If True, returns a tuple (bool, type | None).

    Examples:
        >>> is_structure_homogeneous({'a': {'b': 1}, 'c': {'d': 2}})
        True
        >>> is_structure_homogeneous({'a': {'b': 1}, 'c': [1, 2]})
        False
    """

    def _check_structure(substructure):
        structure_type = None
        if isinstance(substructure, list):
            structure_type = list
            for item in substructure:
                if not isinstance(item, structure_type) and isinstance(
                    item, list | dict
                ):
                    return False, None
                result, _ = _check_structure(item)
                if not result:
                    return False, None
        elif isinstance(substructure, dict):
            structure_type = dict
            for item in substructure.values():
                if not isinstance(item, structure_type) and isinstance(
                    item, list | dict
                ):
                    return False, None
                result, _ = _check_structure(item)
                if not result:
                    return False, None
        return True, structure_type

    is_homogeneous, structure_type = _check_structure(structure)
    return (is_homogeneous, structure_type) if return_structure_type else is_homogeneous


def deep_update(original: dict[Any, Any], update: dict[Any, Any]) -> dict[Any, Any]:
    """
    Recursively merge two dicts, updating nested dicts instead of overwriting.

    Args:
        original: The dictionary to update.
        update: The dictionary containing updates to apply to `original`.

    Returns:
        The `original` dictionary after applying updates from `update`.

    Note:
        This method modifies the `original` dictionary in place.
    """
    for key, value in update.items():
        if isinstance(value, dict) and key in original:
            original[key] = deep_update(original.get(key, {}), value)
        else:
            original[key] = value
    return original


def get_target_container(
    nested: list[Any] | dict[Any, Any], indices: list[int | str]
) -> list[Any] | dict[Any, Any]:
    """
    Retrieve the target container in a nested structure using indices.

    Args:
        nested: The nested structure to navigate.
        indices: A list of indices to navigate through the nested structure.

    Returns:
        The target container at the specified path.

    Raises:
        IndexError: If a list index is out of range.
        KeyError: If a dictionary key is not found.
        TypeError: If the current element is neither a list nor a dictionary.
    """
    current_element = nested
    for index in indices:
        if isinstance(current_element, list):
            if isinstance(index, str) and index.isdigit():
                index = int(index)

            if isinstance(index, int) and 0 <= index < len(current_element):
                current_element = current_element[index]

            else:
                raise IndexError("List index is invalid or out of range")

        elif isinstance(current_element, dict):
            if index in current_element:
                current_element = current_element.get(index, None)
            else:
                raise KeyError("Key not found in dictionary")
        else:
            raise TypeError("Current element is neither a list nor a dictionary")
    return current_element


def _flatten_list_generator(
    lst_: list[Any], dropna: bool
) -> Generator[Any, None, None]:
    for i in lst_:
        if isinstance(i, list):
            yield from _flatten_list_generator(i, dropna)
        else:
            yield i


def to_flat_list(
    input_: Any, /, *, dropna: bool = False, unique: bool = True
) -> list[Any]:
    if isinstance(input_, type(None) | UndefinedType | PydanticUndefinedType):
        return []

    if not isinstance(input_, Iterable) or isinstance(
        input_, (str, bytes, bytearray, dict)
    ):
        return [input_]

    if isinstance(input_, list):
        return _flatten_list(input_, dropna, unique=unique)

    if isinstance(input_, tuple):
        return _flatten_list(list(input_), dropna, unique=unique)

    if isinstance(input_, set):
        return list(_dropna_iterator(list(input_))) if dropna else list(input_)

    try:
        iterable_list = list(input_)
        return _flatten_list(iterable_list, dropna, unique=unique)

    except Exception as e:
        raise ValueError(f"Could not convert {type(input_)} object to list: {e}") from e


def _dropna_iterator(lst_: list[Any]) -> iter:
    return (item for item in lst_ if item is not None)


def _flatten_list(
    lst_: list[Any], dropna: bool = False, unique: bool = False
) -> list[Any]:
    flattened_list = list(_flatten_list_generator(lst_, dropna))
    if dropna:
        flattened_list = list(_dropna_iterator(flattened_list))
    if unique:
        try:
            flattened_list = list(set(flattened_list))
        except Exception:
            pass
    return flattened_list
