# Lion-OS Action Module API Reference

## Type Definitions

```python
# From tool_manager.py
FUNCTOOL = Tool | Callable[..., Any]
FINDABLE_TOOL = FUNCTOOL | str
INPUTTABLE_TOOL = dict[str, Any] | bool | FINDABLE_TOOL
TOOL_TYPE = FINDABLE_TOOL | list[FINDABLE_TOOL] | INPUTTABLE_TOOL

# From base.py
class EventStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
```

## Core Components

### ObservableAction

Base class for actions that can be monitored and tracked.

```python
class ObservableAction(Element):
    status: EventStatus = EventStatus.PENDING
    execution_time: float | None = None
    execution_response: Any = None
    execution_error: str | None = None
    _timed_config: TimedFuncCallConfig | None = PrivateAttr(None)
    _content_fields: list = PrivateAttr(["execution_response"])

    def __init__(
        self,
        timed_config: dict | TimedFuncCallConfig | None,
        **kwargs: Any
    ) -> None
```

Key Methods:
- `to_log() -> Log` - Convert action to log entry
- `from_dict()` - Not implemented, raises NotImplementedError

See also: [Element API Reference](../generic/element.md) | [Log API Reference](../generic/log.md)

### Tool

```python
class Tool(Element):
    function: Callable[..., Any] = Field(
        ...,
        description="The callable function of the tool.",
    )
    schema_: dict[str, Any] | None = Field(
        default=None,
        description="Schema of the function in OpenAI format.",
    )
    pre_processor: Callable[..., dict[str, Any]] | None = Field(
        default=None,
        description="Function to preprocess input arguments.",
    )
    pre_processor_kwargs: dict[str, Any] | None = Field(
        default=None,
        description="Keyword arguments for the pre-processor.",
    )
    post_processor: Callable[..., Any] | None = Field(
        default=None,
        description="Function to post-process the result.",
    )
    post_processor_kwargs: dict[str, Any] | None = Field(
        default=None,
        description="Keyword arguments for the post-processor.",
    )
    parser: Callable[[Any], Any] | None = Field(
        default=None,
        description="Function to parse result to JSON serializable format.",
    )
```

Properties:
- `function_name() -> str` - Get name from schema

Helper Function:
```python
def func_to_tool(
    func_: Callable[..., Any] | list[Callable[..., Any]],
    parser: Callable[[Any], Any] | list[Callable[[Any], Any]] | None = None,
    docstring_style: Literal["google", "rest"] = "google",
    **kwargs,
) -> list[Tool]
```

See also: [Element API Reference](../generic/element.md)

### FunctionCalling

```python
class FunctionCalling(ObservableAction):
    func_tool: Tool | None = Field(default=None, exclude=True)
    _content_fields: list = PrivateAttr(
        default=["execution_response", "arguments", "function"]
    )
    arguments: dict[str, Any] | None = None
    function: str | None = None

    def __init__(
        self,
        func_tool: Tool,
        arguments: dict[str, Any],
        timed_config: dict | TimedFuncCallConfig = None,
        **kwargs: Any,
    ) -> None
```

Key Methods:
- `async invoke() -> Any` - Execute function with processing pipeline

### ToolManager

```python
class ToolManager:
    def __init__(
        self,
        registry: dict[str, Tool] | None = None,
        logger=None
    ) -> None
```

Key Methods:
```python
def register_tool(
    self,
    tool: FUNCTOOL,
    update: bool = False,
) -> None

def register_tools(
    self,
    tools: list[FUNCTOOL] | FUNCTOOL,
    update: bool = False,
) -> None

@singledispatchmethod
def match_tool(self, func_call: Any) -> FunctionCalling

async def invoke(
    self,
    func_call: dict | str | ActionRequest,
    log_manager: LogManager = None,
) -> Any
```

Properties:
- `schema_list -> list[dict[str, Any]]` - All registered tool schemas

See also: [LogManager API Reference](../generic/log_manager.md)

## External Dependencies

The following types/classes are used but defined elsewhere:

- [Element](../generic/element.md)
- [Log](../generic/log.md)
- [LogManager](../generic/log_manager.md)
- [TimedFuncCallConfig](../settings.md)
- [ActionRequest](../communication/action_request.md)
- [ActionRequestModel](../protocols/operatives/action.md)

## Notes

- Tool instances inherit from Element
- Function execution is asynchronous
- Tool schemas follow OpenAI function format
- Registration requires unique function names
- Tools support pre/post processing pipeline
