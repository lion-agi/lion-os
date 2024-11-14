import asyncio

from typing_extensions import override

from lion.core.typing import Any, Field, PrivateAttr
from lion.libs.func import CallDecorator as cd
from lion.libs.func import tcall
from lion.settings import TimedFuncCallConfig

from .base import EventStatus, ObservableAction
from .tool import Tool


class FunctionCalling(ObservableAction):
    """Represents an action that calls a function with specified arguments.

    Encapsulates a function call, including pre-processing, invocation,
    and post-processing steps. Designed to be executed asynchronously.

    Attributes:
        func_tool (Tool): Tool containing the function to be invoked.
        arguments (dict[str, Any]): Arguments for the function invocation.
        function (str | None): Name of the function to be called.
    """

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
    ) -> None:
        """Initialize a FunctionCalling instance.

        Args:
            func_tool: Tool containing the function to be invoked.
            arguments: Arguments for the function invocation.
            timed_config: Configuration for timing and retries.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(timed_config=timed_config, **kwargs)
        self.func_tool = func_tool
        self.arguments = arguments or {}
        self.function = self.func_tool.function_name

    @override
    async def invoke(self) -> Any:
        """Asynchronously invokes the function with stored arguments.

        Handles function invocation, applying pre/post-processing steps.
        If a parser is defined, it's applied to the result before returning.

        Returns:
            Any: Result of the function call, possibly processed.

        Raises:
            Exception: If function call or processing steps fail.
        """
        start = asyncio.get_event_loop().time()
        try:
            # Create inner function with pre/post processing
            @cd.pre_post_process(
                preprocess=self.func_tool.pre_processor,
                postprocess=self.func_tool.post_processor,
                preprocess_kwargs=self.func_tool.pre_processor_kwargs or {},
                postprocess_kwargs=self.func_tool.post_processor_kwargs or {},
            )
            async def _inner(**kwargs) -> Any:
                config = self._timed_config.to_dict()
                result = await tcall(self.func_tool.function, **kwargs, **config)
                # Handle tuple result from tcall when retry_timing is True
                if isinstance(result, tuple) and len(result) == 2:
                    return result[0]  # Return just the result, not timing info
                return result

            # Execute function with pre/post processing
            result = await _inner(**self.arguments)
            self.execution_response = result
            self.execution_time = asyncio.get_event_loop().time() - start
            self.status = EventStatus.COMPLETED

            # Apply parser if defined
            if self.func_tool.parser is not None:
                if asyncio.iscoroutinefunction(self.func_tool.parser):
                    result = await self.func_tool.parser(result)
                else:
                    result = self.func_tool.parser(result)
            return result

        except Exception as e:
            self.status = EventStatus.FAILED
            self.execution_error = str(e)
            self.execution_time = asyncio.get_event_loop().time() - start
            return None

    def __str__(self) -> str:
        """Returns a string representation of the function call."""
        return f"{self.func_tool.function_name}({self.arguments})"

    def __repr__(self) -> str:
        """Returns a detailed string representation of the function call."""
        return (
            f"FunctionCalling(function={self.func_tool.function_name}, "
            f"arguments={self.arguments})"
        )


__all__ = ["FunctionCalling"]
