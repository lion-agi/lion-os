import asyncio
import functools
import logging
import time
from collections.abc import AsyncGenerator, Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache, wraps
from typing import Any, TypeVar

from .constants import UNDEFINED
from .parse import to_list
from .utils import time as _t

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


async def bcall(
    input_: Any,
    func: Callable[..., T],
    /,
    batch_size: int,
    num_retries: int = 0,
    initial_delay: float = 0,
    retry_delay: float = 0,
    backoff_factor: float = 1,
    retry_default: Any = None,
    retry_timeout: float | None = None,
    retry_timing: bool = False,
    verbose_retry: bool = True,
    error_msg: str | None = None,
    error_map: dict[type, Callable[[Exception], Any]] | None = None,
    max_concurrent: int | None = None,
    throttle_period: float | None = None,
    **kwargs: Any,
) -> AsyncGenerator[list[T | tuple[T, float]], None]:
    """
    Asynchronously call a function in batches with retry and timing options.

    Args:
        input_: The input data to process.
        func: The function to call.
        batch_size: The size of each batch.
        retries: The number of retries.
        initial_delay: Initial delay before the first attempt in seconds.
        delay: The delay between retries in seconds.
        backoff_factor: Factor by which delay increases after each retry.
        default: Default value to return if an error occurs.
        timeout: The timeout for the function call in seconds.
        timing: If True, return execution time along with the result.
        verbose: If True, print retry attempts and exceptions.
        error_msg: Custom error message prefix.
        error_map: Mapping of errors to handle custom error responses.
        max_concurrent: Maximum number of concurrent calls.
        throttle_period: Throttle period in seconds.
        **kwargs: Additional keyword arguments to pass to the function.

    Yields:
        A list of results for each batch of inputs.

    Examples:
        >>> async def sample_func(x):
        ...     return x * 2
        >>> async for batch_results in bcall([1, 2, 3, 4, 5], sample_func, 2,
        ...                                  retries=3, delay=1):
        ...     print(batch_results)
    """
    input_ = to_list(input_, flatten=True, dropna=True)

    for i in range(0, len(input_), batch_size):
        batch = input_[i : i + batch_size]  # noqa: E203
        batch_results = await alcall(
            batch,
            func,
            num_retries=num_retries,
            initial_delay=initial_delay,
            retry_delay=retry_delay,
            backoff_factor=backoff_factor,
            retry_default=retry_default,
            retry_timeout=retry_timeout,
            retry_timing=retry_timing,
            verbose_retry=verbose_retry,
            error_msg=error_msg,
            error_map=error_map,
            max_concurrent=max_concurrent,
            throttle_period=throttle_period,
            **kwargs,
        )
        yield batch_results


class CallDecorator:
    """A collection of decorators to enhance function calls."""

    @staticmethod
    def retry(
        num_retries: int = 0,
        initial_delay: float = 0,
        retry_delay: float = 0,
        backoff_factor: float = 1,
        retry_default: Any = UNDEFINED,
        retry_timeout: float | None = None,
        retry_timing: bool = False,
        verbose_retry: bool = True,
        error_msg: str | None = None,
        error_map: dict[type, Callable[[Exception], None]] | None = None,
    ) -> Callable[[F], F]:
        """Decorator to automatically retry a function call on failure.

        Args:
            retries: Number of retry attempts.
            initial_delay: Initial delay before retrying.
            delay: Delay between retries.
            backoff_factor: Factor to increase delay after each retry.
            default: Default value to return on failure.
            timeout: Timeout for each function call.
            timing: If True, logs the time taken for each call.
            verbose: If True, logs the retries.
            error_msg: Custom error message on failure.
            error_map: A map of exception types to handler functions.

        Returns:
            The decorated function.
        """

        def decorator(func: F) -> F:
            @wraps(func)
            async def wrapper(*args, **kwargs) -> Any:
                return await rcall(
                    func,
                    *args,
                    num_retries=num_retries,
                    initial_delay=initial_delay,
                    retry_delay=retry_delay,
                    backoff_factor=backoff_factor,
                    retry_default=retry_default,
                    retry_timeout=retry_timeout,
                    retry_timing=retry_timing,
                    verbose_retry=verbose_retry,
                    error_msg=error_msg,
                    error_map=error_map,
                    **kwargs,
                )

            return wrapper

        return decorator

    @staticmethod
    def throttle(period: float) -> Callable[[F], F]:
        """Decorator to limit the execution frequency of a function.

        Args:
            period: Minimum time in seconds between function calls.

        Returns:
            The decorated function.
        """

        def decorator(func: F) -> F:
            if not is_coroutine_func(func):
                func = force_async(func)
            throttle_instance = Throttle(period)

            @wraps(func)
            async def wrapper(*args, **kwargs):
                await throttle_instance(func)(*args, **kwargs)
                return await func(*args, **kwargs)

            return wrapper

        return decorator

    @staticmethod
    def max_concurrent(limit: int) -> Callable[[F], F]:
        """Decorator to limit the maximum number of concurrent executions.

        Args:
            limit: Maximum number of concurrent executions.

        Returns:
            The decorated function.
        """

        def decorator(func: F) -> F:
            if not is_coroutine_func(func):
                func = force_async(func)
            semaphore = asyncio.Semaphore(limit)

            @wraps(func)
            async def wrapper(*args, **kwargs):
                async with semaphore:
                    return await func(*args, **kwargs)

            return wrapper

        return decorator

    @staticmethod
    def compose(*functions: Callable[[T], T]) -> Callable[[F], F]:
        """Decorator to compose multiple functions, applying in sequence.

        Args:
            functions: Functions to apply in sequence.

        Returns:
            The decorated function.
        """

        def decorator(func: F) -> F:
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                value = await ucall(func, *args, **kwargs)
                for function in functions:
                    try:
                        value = await ucall(function, value)
                    except Exception as e:
                        raise ValueError(f"Error in function {function.__name__}: {e}")
                return value

            return async_wrapper

        return decorator

    @staticmethod
    def pre_post_process(
        preprocess: Callable[..., Any] | None = None,
        postprocess: Callable[..., Any] | None = None,
        preprocess_args: Sequence[Any] = (),
        preprocess_kwargs: dict[str, Any] = {},
        postprocess_args: Sequence[Any] = (),
        postprocess_kwargs: dict[str, Any] = {},
    ) -> Callable[[F], F]:
        """Decorator to apply pre-processing and post-processing functions.

        Args:
            preprocess: Function to apply before the main function.
            postprocess: Function to apply after the main function.
            preprocess_args: Arguments for the preprocess function.
            preprocess_kwargs: Keyword arguments for preprocess function.
            postprocess_args: Arguments for the postprocess function.
            postprocess_kwargs: Keyword arguments for postprocess function.

        Returns:
            The decorated function.
        """

        def decorator(func: F) -> F:
            @wraps(func)
            async def async_wrapper(*args, **kwargs) -> Any:
                preprocessed_args = (
                    [
                        await ucall(
                            preprocess,
                            arg,
                            *preprocess_args,
                            **preprocess_kwargs,
                        )
                        for arg in args
                    ]
                    if preprocess
                    else args
                )
                preprocessed_kwargs = (
                    {
                        k: await ucall(
                            preprocess,
                            v,
                            *preprocess_args,
                            **preprocess_kwargs,
                        )
                        for k, v in kwargs.items()
                    }
                    if preprocess
                    else kwargs
                )
                result = await ucall(func, *preprocessed_args, **preprocessed_kwargs)

                return (
                    await ucall(
                        postprocess,
                        result,
                        *postprocess_args,
                        **postprocess_kwargs,
                    )
                    if postprocess
                    else result
                )

            return async_wrapper

        return decorator

    @staticmethod
    def map(function: Callable[[Any], Any]) -> Callable:
        """Decorator to map a function over async function results.

        Applies a mapping function to each element in the list returned
        by the decorated function. Useful for post-processing results of
        asynchronous operations, such as transforming data fetched from
        an API or processing items in a collection concurrently.

        Args:
            function: Mapping function to apply to each element.

        Returns:
            Decorated async function with transformed results.

        Examples:
            >>> @CallDecorator.map(lambda x: x.upper())
            ... async def get_names():
            ...     return ["alice", "bob", "charlie"]
            ... # `get_names` now returns ["ALICE", "BOB", "CHARLIE"]
        """

        def decorator(func: Callable[..., list[Any]]) -> Callable:
            if is_coroutine_func(func):

                @wraps(func)
                async def async_wrapper(*args, **kwargs) -> list[Any]:
                    values = await func(*args, **kwargs)
                    return [function(value) for value in values]

                return async_wrapper
            else:

                @wraps(func)
                def sync_wrapper(*args, **kwargs) -> list[Any]:
                    values = func(*args, **kwargs)
                    return [function(value) for value in values]

                return sync_wrapper

        return decorator


def lcall(
    input_: list[Any],
    func: Callable[..., T],
    /,
    *,
    flatten: bool = False,
    dropna: bool = False,
    unique: bool = False,
    **kwargs,
) -> list[Any]:
    """Apply a function to each element of a list synchronously.

    Args:
        input_: List of inputs to be processed.
        func: Function to apply to each input element.
        flatten: If True, flatten the resulting list.
        dropna: If True, remove None values from the result.
        unique: If True, return only unique values (requires flatten=True).
        **kwargs: Additional keyword arguments passed to func.

    Returns:
        list[Any]: List of results after applying func to each input element.

    Raises:
        ValueError: If more than one function is provided.

    Examples:
        >>> lcall([1, 2, 3], lambda x: x * 2)
        [2, 4, 6]
        >>> lcall([[1, 2], [3, 4]], sum, flatten=True)
        [3, 7]
        >>> lcall([1, 2, 2, 3], lambda x: x, unique=True, flatten=True)
        [1, 2, 3]

    Note:
        The function uses to_list internally, which allows for flexible input
        types beyond just lists.
    """
    lst = to_list(input_)
    if len(to_list(func, flatten=True, dropna=True)) != 1:
        raise ValueError("There must be one and only one function for list calling.")
    return to_list(
        [func(i, **kwargs) for i in lst],
        flatten=flatten,
        dropna=dropna,
        unique=unique,
    )


async def alcall(
    input_: list[Any],
    func: Callable[..., T],
    /,
    num_retries: int = 0,
    initial_delay: float = 0,
    retry_delay: float = 0,
    backoff_factor: float = 1,
    retry_default: Any = UNDEFINED,
    retry_timeout: float | None = None,
    retry_timing: bool = False,
    verbose_retry: bool = True,
    error_msg: str | None = None,
    error_map: dict[type, Callable[[Exception], None]] | None = None,
    max_concurrent: int | None = None,
    throttle_period: float | None = None,
    flatten: bool = False,
    dropna: bool = False,
    unique: bool = False,
    **kwargs: Any,
) -> list[T] | list[tuple[T, float]]:
    """Apply a function to each element of a list asynchronously with options.

    Args:
        input_: List of inputs to be processed.
        func: Async or sync function to apply to each input element.
        num_retries: Number of retry attempts for each function call.
        initial_delay: Initial delay before starting execution (seconds).
        retry_delay: Delay between retry attempts (seconds).
        backoff_factor: Factor by which delay increases after each attempt.
        retry_default: Default value to return if all attempts fail.
        retry_timeout: Timeout for each function execution (seconds).
        retry_timing: If True, return execution duration for each call.
        verbose_retry: If True, print retry messages.
        error_msg: Custom error message prefix for exceptions.
        error_map: Dict mapping exception types to error handlers.
        max_concurrent: Maximum number of concurrent executions.
        throttle_period: Minimum time between function executions (seconds).
        flatten: If True, flatten the resulting list.
        dropna: If True, remove None values from the result.
        **kwargs: Additional keyword arguments passed to func.

    Returns:
        list[T] | list[tuple[T, float]]: List of results, optionally with
        execution times if retry_timing is True.

    Raises:
        asyncio.TimeoutError: If execution exceeds retry_timeout.
        Exception: Any exception raised by func if not handled by error_map.

    Examples:
        >>> async def square(x):
        ...     return x * x
        >>> await alcall([1, 2, 3], square)
        [1, 4, 9]
        >>> await alcall([1, 2, 3], square, retry_timing=True)
        [(1, 0.001), (4, 0.001), (9, 0.001)]

    Note:
        - Uses semaphores for concurrency control if max_concurrent is set.
        - Supports both synchronous and asynchronous functions for `func`.
        - Results are returned in the original input order.
    """
    if initial_delay:
        await asyncio.sleep(initial_delay)

    semaphore = asyncio.Semaphore(max_concurrent) if max_concurrent else None
    throttle_delay = throttle_period if throttle_period else 0

    async def _task(i: Any, index: int) -> Any:
        if semaphore:
            async with semaphore:
                return await _execute_task(i, index)
        else:
            return await _execute_task(i, index)

    async def _execute_task(i: Any, index: int) -> Any:
        attempts = 0
        current_delay = retry_delay
        while True:
            try:
                if retry_timing:
                    start_time = asyncio.get_event_loop().time()
                    result = await asyncio.wait_for(
                        ucall(func, i, **kwargs), retry_timeout
                    )
                    end_time = asyncio.get_event_loop().time()
                    return index, result, end_time - start_time
                else:
                    result = await asyncio.wait_for(
                        ucall(func, i, **kwargs), retry_timeout
                    )
                    return index, result
            except TimeoutError as e:
                raise TimeoutError(
                    f"{error_msg or ''} Timeout {retry_timeout} seconds " "exceeded"
                ) from e
            except Exception as e:
                if error_map and type(e) in error_map:
                    handler = error_map[type(e)]
                    if asyncio.iscoroutinefunction(handler):
                        return index, await handler(e)
                    else:
                        return index, handler(e)
                attempts += 1
                if attempts <= num_retries:
                    if verbose_retry:
                        print(
                            f"Attempt {attempts}/{num_retries} failed: {e}"
                            ", retrying..."
                        )
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff_factor
                else:
                    if retry_default is not UNDEFINED:
                        return index, retry_default
                    raise e

    tasks = [_task(i, index) for index, i in enumerate(input_)]
    results = []
    for coro in asyncio.as_completed(tasks):
        result = await coro
        results.append(result)
        await asyncio.sleep(throttle_delay)

    results.sort(key=lambda x: x[0])  # Sort results based on the original index

    if retry_timing:
        if dropna:
            return [
                (result[1], result[2]) for result in results if result[1] is not None
            ]
        else:
            return [(result[1], result[2]) for result in results]
    else:
        return to_list(
            [result[1] for result in results],
            flatten=flatten,
            dropna=dropna,
            unique=unique,
        )


async def mcall(
    input_: Any,
    func: Callable[..., T] | Sequence[Callable[..., T]],
    /,
    *,
    explode: bool = False,
    num_retries: int = 0,
    initial_delay: float = 0,
    retry_delay: float = 0,
    backoff_factor: float = 1,
    retry_default: Any = UNDEFINED,
    retry_timeout: float | None = None,
    retry_timing: bool = False,
    verbose_retry: bool = True,
    error_msg: str | None = None,
    error_map: dict[type, Callable[[Exception], None]] | None = None,
    max_concurrent: int | None = None,
    throttle_period: float | None = None,
    dropna: bool = False,
    **kwargs: Any,
) -> list[T] | list[tuple[T, float]]:
    """
    Apply functions over inputs asynchronously with customizable options.

    Args:
        input_: The input data to be processed.
        func: The function or sequence of functions to be applied.
        explode: Whether to apply each function to all inputs.
        retries: Number of retry attempts for each function call.
        initial_delay: Initial delay before starting execution.
        delay: Delay between retry attempts.
        backoff_factor: Factor by which delay increases after each attempt.
        default: Default value to return if all attempts fail.
        timeout: Timeout for each function execution.
        timing: Whether to return the execution duration.
        verbose: Whether to print retry messages.
        error_msg: Custom error message.
        error_map: Dictionary mapping exception types to error handlers.
        max_concurrent: Maximum number of concurrent executions.
        throttle_period: Minimum time period between function executions.
        dropna: Whether to drop None values from the output list.
        **kwargs: Additional keyword arguments for the functions.

    Returns:
        List of results, optionally including execution durations if timing
        is True.

    Raises:
        ValueError: If the length of inputs and functions don't match when
            not exploding the function calls.
    """
    input_ = to_list(input_, flatten=False, dropna=False)
    func = to_list(func, flatten=False, dropna=False)

    if explode:
        tasks = [
            alcall(
                input_,
                f,
                num_retries=num_retries,
                initial_delay=initial_delay,
                retry_delay=retry_delay,
                backoff_factor=backoff_factor,
                retry_default=retry_default,
                retry_timeout=retry_timeout,
                retry_timing=retry_timing,
                verbose_retry=verbose_retry,
                error_msg=error_msg,
                error_map=error_map,
                max_concurrent=max_concurrent,
                throttle_period=throttle_period,
                dropna=dropna,
                **kwargs,
            )
            for f in func
        ]
        return await asyncio.gather(*tasks)
    elif len(func) == 1:
        tasks = [
            rcall(
                func[0],
                inp,
                num_retries=num_retries,
                initial_delay=initial_delay,
                retry_delay=retry_delay,
                backoff_factor=backoff_factor,
                retry_default=retry_default,
                retry_timeout=retry_timeout,
                retry_timing=retry_timing,
                verbose_retry=verbose_retry,
                error_msg=error_msg,
                error_map=error_map,
                **kwargs,
            )
            for inp in input_
        ]
        return await asyncio.gather(*tasks)

    elif len(input_) == len(func):
        tasks = [
            rcall(
                f,
                inp,
                num_retries=num_retries,
                initial_delay=initial_delay,
                retry_delay=retry_delay,
                backoff_factor=backoff_factor,
                retry_default=retry_default,
                retry_timeout=retry_timeout,
                retry_timing=retry_timing,
                verbose_retry=verbose_retry,
                error_msg=error_msg,
                error_map=error_map,
                **kwargs,
            )
            for inp, f in zip(input_, func)
        ]
        return await asyncio.gather(*tasks)
    else:
        raise ValueError(
            "Inputs and functions must be the same length for map calling."
        )


async def pcall(
    funcs: Sequence[Callable[..., T]],
    /,
    num_retries: int = 0,
    initial_delay: float = 0,
    retry_delay: float = 0,
    backoff_factor: float = 1,
    retry_default: Any = UNDEFINED,
    retry_timeout: float | None = None,
    retry_timing: bool = False,
    verbose_retry: bool = True,
    error_msg: str | None = None,
    error_map: dict[type, Callable[[Exception], None]] | None = None,
    max_concurrent: int | None = None,
    throttle_period: float | None = None,
    **kwargs: Any,
) -> list[T] | list[tuple[T, float]]:
    """Execute multiple functions asynchronously in parallel with options.

    Manages parallel execution of functions with retry logic, timing, and
    error handling. Supports concurrency control and throttling.

    Args:
        funcs: Sequence of functions to execute in parallel.
        num_retries: Number of retry attempts for each function (default: 0).
        initial_delay: Delay before starting execution (seconds).
        retry_delay: Initial delay between retry attempts (seconds).
        backoff_factor: Factor to increase delay after each retry.
        retry_default: Value to return if all attempts for a function fail.
        retry_timeout: Timeout for each function execution (seconds).
        retry_timing: If True, return execution duration for each function.
        verbose_retry: If True, print retry messages.
        error_msg: Custom error message prefix.
        error_map: Dict mapping exception types to error handlers.
        max_concurrent: Maximum number of functions to run concurrently.
        throttle_period: Minimum time between function starts (seconds).
        **kwargs: Additional keyword arguments passed to each function.

    Returns:
        list[T] | list[tuple[T, float]]: List of results, optionally with
        execution times if retry_timing is True.

    Raises:
        asyncio.TimeoutError: If any function execution exceeds retry_timeout.
        Exception: Any unhandled exception from function executions.

    Examples:
        >>> async def func1(x):
        ...     await asyncio.sleep(1)
        ...     return x * 2
        >>> async def func2(x):
        ...     await asyncio.sleep(0.5)
        ...     return x + 10
        >>> results = await pcall([func1, func2], retry_timing=True, x=5)
        >>> for result, duration in results:
        ...     print(f"Result: {result}, Duration: {duration:.2f}s")
        Result: 10, Duration: 1.00s
        Result: 15, Duration: 0.50s

    Note:
        - Executes functions in parallel, respecting max_concurrent limit.
        - Implements exponential backoff for retries.
        - Can return execution timing for performance analysis.
        - Supports both coroutine and regular functions via ucall.
        - Results are returned in the original order of input functions.
    """
    if initial_delay:
        await asyncio.sleep(initial_delay)

    semaphore = asyncio.Semaphore(max_concurrent) if max_concurrent else None
    throttle_delay = throttle_period if throttle_period else 0

    async def _task(func: Callable[..., Any], index: int) -> Any:
        if semaphore:
            async with semaphore:
                return await _execute_task(func, index)
        else:
            return await _execute_task(func, index)

    async def _execute_task(func: Callable[..., Any], index: int) -> Any:
        attempts = 0
        current_delay = retry_delay
        while True:
            try:
                if retry_timing:
                    start_time = asyncio.get_event_loop().time()
                    result = await asyncio.wait_for(
                        ucall(func, **kwargs), retry_timeout
                    )
                    end_time = asyncio.get_event_loop().time()
                    return index, result, end_time - start_time
                else:
                    result = await asyncio.wait_for(
                        ucall(func, **kwargs), retry_timeout
                    )
                    return index, result
            except TimeoutError as e:
                raise TimeoutError(
                    f"{error_msg or ''} Timeout {retry_timeout} seconds " "exceeded"
                ) from e
            except Exception as e:
                if error_map and type(e) in error_map:
                    handler = error_map[type(e)]
                    if is_coroutine_func(handler):
                        return index, await handler(e)
                    else:
                        return index, handler(e)
                attempts += 1
                if attempts <= num_retries:
                    if verbose_retry:
                        print(
                            f"Attempt {attempts}/{num_retries + 1} failed: {e}"
                            ", retrying..."
                        )
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff_factor
                else:
                    if retry_default is not UNDEFINED:
                        return index, retry_default
                    raise e

    tasks = [_task(func, index) for index, func in enumerate(funcs)]
    results = []
    for coro in asyncio.as_completed(tasks):
        result = await coro
        results.append(result)
        await asyncio.sleep(throttle_delay)

    results.sort(key=lambda x: x[0])  # Sort results based on the original index

    if retry_timing:
        return [(result[1], result[2]) for result in results]
    else:
        return [result[1] for result in results]


async def rcall(
    func: Callable[..., T],
    /,
    *args: Any,
    num_retries: int = 0,
    initial_delay: float = 0,
    retry_delay: float = 0,
    backoff_factor: float = 1,
    retry_default: Any = UNDEFINED,
    retry_timeout: float | None = None,
    retry_timing: bool = False,
    verbose_retry: bool = True,
    error_msg: str | None = None,
    error_map: dict[type, Callable[[Exception], None]] | None = None,
    **kwargs: Any,
) -> T | tuple[T, float]:
    """Retry a function asynchronously with customizable options.

    Executes a function with specified retry logic, timing, and error handling.

    Args:
        func: The function to execute (coroutine or regular).
        *args: Positional arguments for the function.
        num_retries: Number of retry attempts (default: 0).
        initial_delay: Delay before first attempt (seconds).
        retry_delay: Delay between retry attempts (seconds).
        backoff_factor: Factor to increase delay after each retry.
        retry_default: Value to return if all attempts fail.
        retry_timeout: Timeout for each function execution (seconds).
        retry_timing: If True, return execution duration.
        verbose_retry: If True, print retry messages.
        error_msg: Custom error message prefix.
        error_map: Dict mapping exception types to error handlers.
        **kwargs: Additional keyword arguments for the function.

    Returns:
        T | tuple[T, float]: Function result, optionally with duration.

    Raises:
        RuntimeError: If function fails after all retries.
        asyncio.TimeoutError: If execution exceeds retry_timeout.

    Examples:
        >>> async def flaky_func(x):
        ...     if random.random() < 0.5:
        ...         raise ValueError("Random failure")
        ...     return x * 2
        >>> result = await rcall(flaky_func, 5, num_retries=3)
        >>> print(result)
        10

    Note:
        - Supports both coroutine and regular functions.
        - Implements exponential backoff for retries.
        - Can return execution timing for performance analysis.
    """
    last_exception = None
    result = None

    await asyncio.sleep(initial_delay)
    for attempt in range(num_retries + 1):
        try:
            if num_retries == 0:
                if retry_timing:
                    result, duration = await _rcall(
                        func,
                        *args,
                        retry_timeout=retry_timeout,
                        retry_timing=True,
                        **kwargs,
                    )
                    return result, duration
                result = await _rcall(
                    func,
                    *args,
                    retry_timeout=retry_timeout,
                    **kwargs,
                )
                return result
            err_msg = f"Attempt {attempt + 1}/{num_retries + 1}: {error_msg or ''}"
            if retry_timing:
                result, duration = await _rcall(
                    func,
                    *args,
                    error_msg=err_msg,
                    retry_timeout=retry_timeout,
                    retry_timing=True,
                    **kwargs,
                )
                return result, duration

            result = await _rcall(
                func,
                *args,
                error_msg=err_msg,
                retry_timeout=retry_timeout,
                **kwargs,
            )
            return result
        except Exception as e:
            last_exception = e
            if error_map and type(e) in error_map:
                error_map[type(e)](e)
            if attempt < num_retries:
                if verbose_retry:
                    print(
                        f"Attempt {attempt + 1}/{num_retries + 1} failed: {e},"
                        " retrying..."
                    )
                await asyncio.sleep(retry_delay)
                retry_delay *= backoff_factor
            else:
                break

    if retry_default is not UNDEFINED:
        return retry_default

    if last_exception is not None:
        if error_map and type(last_exception) in error_map:
            handler = error_map[type(last_exception)]
            if asyncio.iscoroutinefunction(handler):
                return await handler(last_exception)
            else:
                return handler(last_exception)
        raise RuntimeError(
            f"{error_msg or ''} Operation failed after {num_retries + 1} "
            f"attempts: {last_exception}"
        ) from last_exception

    raise RuntimeError(
        f"{error_msg or ''} Operation failed after {num_retries + 1} attempts"
    )


async def _rcall(
    func: Callable[..., T],
    *args: Any,
    retry_delay: float = 0,
    error_msg: str | None = None,
    ignore_err: bool = False,
    retry_timing: bool = False,
    retry_default: Any = None,
    retry_timeout: float | None = None,
    **kwargs: Any,
) -> T | tuple[T, float]:
    start_time = _t()

    try:
        await asyncio.sleep(retry_delay)
        if retry_timeout is not None:
            result = await asyncio.wait_for(
                ucall(func, *args, **kwargs), timeout=retry_timeout
            )
        else:
            result = await ucall(func, *args, **kwargs)
        duration = _t() - start_time
        return (result, duration) if retry_timing else result
    except TimeoutError as e:
        error_msg = f"{error_msg or ''} Timeout {retry_timeout} seconds exceeded"
        if ignore_err:
            duration = _t() - start_time
            return (retry_default, duration) if retry_timing else retry_default
        else:
            raise TimeoutError(error_msg) from e
    except Exception:
        if ignore_err:
            duration = _t() - start_time
            return (retry_default, duration) if retry_timing else retry_default
        else:
            raise


async def tcall(
    func: Callable[..., T],
    /,
    *args: Any,
    initial_delay: float = 0,
    error_msg: str | None = None,
    suppress_err: bool = False,
    retry_timing: bool = False,
    retry_timeout: float | None = None,
    retry_default: Any = None,
    error_map: dict[type, Callable[[Exception], None]] | None = None,
    **kwargs: Any,
) -> T | tuple[T, float]:
    """Execute a function asynchronously with timing and error handling.

    Handles both coroutine and regular functions, supporting timing,
    timeout, and custom error handling.

    Args:
        func: The function to execute (coroutine or regular).
        *args: Positional arguments for the function.
        initial_delay: Delay before execution (seconds).
        error_msg: Custom error message prefix.
        suppress_err: If True, return default on error instead of raising.
        retry_timing: If True, return execution duration.
        retry_timeout: Timeout for function execution (seconds).
        retry_default: Value to return if an error occurs and suppress_err
        is True.
        error_map: Dict mapping exception types to error handlers.
        **kwargs: Additional keyword arguments for the function.

    Returns:
        T | tuple[T, float]: Function result, optionally with duration.

    Raises:
        asyncio.TimeoutError: If execution exceeds the timeout.
        RuntimeError: If an error occurs and suppress_err is False.

    Examples:
        >>> async def slow_func(x):
        ...     await asyncio.sleep(1)
        ...     return x * 2
        >>> result, duration = await tcall(slow_func, 5, retry_timing=True)
        >>> print(f"Result: {result}, Duration: {duration:.2f}s")
        Result: 10, Duration: 1.00s

    Note:
        - Automatically handles both coroutine and regular functions.
        - Provides timing information for performance analysis.
        - Supports custom error handling and suppression.
    """
    start = asyncio.get_event_loop().time()

    try:
        await asyncio.sleep(initial_delay)
        result = None

        if asyncio.iscoroutinefunction(func):
            # Asynchronous function
            if retry_timeout is None:
                result = await func(*args, **kwargs)
            else:
                result = await asyncio.wait_for(
                    func(*args, **kwargs), timeout=retry_timeout
                )
        else:
            # Synchronous function
            if retry_timeout is None:
                result = func(*args, **kwargs)
            else:
                result = await asyncio.wait_for(
                    asyncio.shield(asyncio.to_thread(func, *args, **kwargs)),
                    timeout=retry_timeout,
                )

        duration = asyncio.get_event_loop().time() - start
        return (result, duration) if retry_timing else result

    except TimeoutError as e:
        error_msg = f"{error_msg or ''} Timeout {retry_timeout} seconds exceeded"
        if suppress_err:
            duration = asyncio.get_event_loop().time() - start
            return (retry_default, duration) if retry_timing else retry_default
        else:
            raise TimeoutError(error_msg) from e

    except Exception as e:
        if error_map and type(e) in error_map:
            error_map[type(e)](e)
            duration = asyncio.get_event_loop().time() - start
            return (None, duration) if retry_timing else None
        error_msg = (
            f"{error_msg} Error: {e}"
            if error_msg
            else f"An error occurred in async execution: {e}"
        )
        if suppress_err:
            duration = asyncio.get_event_loop().time() - start
            return (retry_default, duration) if retry_timing else retry_default
        else:
            raise RuntimeError(error_msg) from e


class Throttle:
    """
    Provide a throttling mechanism for function calls.

    When used as a decorator, it ensures that the decorated function can only
    be called once per specified period. Subsequent calls within this period
    are delayed to enforce this constraint.

    Attributes:
        period: The minimum time period (in seconds) between successive calls.
    """

    def __init__(self, period: float) -> None:
        """
        Initialize a new instance of Throttle.

        Args:
            period: The minimum time period (in seconds) between
                successive calls.
        """
        self.period = period
        self.last_called = 0

    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """
        Decorate a synchronous function with the throttling mechanism.

        Args:
            func: The synchronous function to be throttled.

        Returns:
            The throttled synchronous function.
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            elapsed = _t() - self.last_called
            if elapsed < self.period:
                time.sleep(self.period - elapsed)
            self.last_called = _t()
            return func(*args, **kwargs)

        return wrapper

    def __call_async__(
        self, func: Callable[..., Callable[..., T]]
    ) -> Callable[..., Callable[..., T]]:
        """
        Decorate an asynchronous function with the throttling mechanism.

        Args:
            func: The asynchronous function to be throttled.

        Returns:
            The throttled asynchronous function.
        """

        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            elapsed = _t() - self.last_called
            if elapsed < self.period:
                await asyncio.sleep(self.period - elapsed)
            self.last_called = _t()
            return await func(*args, **kwargs)

        return wrapper


async def ucall(
    func: Callable[..., T],
    /,
    *args: Any,
    error_map: dict[type, Callable[[Exception], None]] | None = None,
    **kwargs: Any,
) -> T:
    """Execute a function asynchronously with error handling.

    Ensures asynchronous execution of both coroutine and regular functions,
    managing event loops and applying custom error handling.

    Args:
        func: The function to be executed (coroutine or regular).
        *args: Positional arguments for the function.
        error_map: Dict mapping exception types to error handlers.
        **kwargs: Keyword arguments for the function.

    Returns:
        T: The result of the function call.

    Raises:
        Exception: Any unhandled exception from the function execution.

    Examples:
        >>> async def example_func(x):
        ...     return x * 2
        >>> await ucall(example_func, 5)
        10
        >>> await ucall(lambda x: x + 1, 5)  # Non-coroutine function
        6

    Note:
        - Automatically wraps non-coroutine functions for async execution.
        - Manages event loop creation and closure when necessary.
        - Applies custom error handling based on the provided error_map.
    """
    try:
        if not is_coroutine_func(func):
            func = force_async(func)

        # Checking for a running event loop
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                return await func(*args, **kwargs)
            else:
                return await asyncio.run(func(*args, **kwargs))

        except RuntimeError:  # No running event loop
            loop = asyncio.new_event_loop()
            result = await func(*args, **kwargs)
            loop.close()
            return result

    except Exception as e:
        if error_map:

            return await custom_error_handler(e, error_map)
        raise e


def force_async(fn: Callable[..., T]) -> Callable[..., Callable[..., T]]:
    """
    Convert a synchronous function to an asynchronous function
    using a thread pool.

    Args:
        fn: The synchronous function to convert.

    Returns:
        The asynchronous version of the function.
    """
    pool = ThreadPoolExecutor()

    @wraps(fn)
    def wrapper(*args, **kwargs):
        future = pool.submit(fn, *args, **kwargs)
        return asyncio.wrap_future(future)  # Make it awaitable

    return wrapper


@lru_cache(maxsize=None)
def is_coroutine_func(func: Callable[..., Any]) -> bool:
    """
    Check if a function is a coroutine function.

    Args:
        func: The function to check.

    Returns:
        True if the function is a coroutine function, False otherwise.
    """
    return asyncio.iscoroutinefunction(func)


async def custom_error_handler(
    error: Exception, error_map: dict[type, Callable[[Exception], None]]
) -> None:
    for error_type, handler in error_map.items():
        if isinstance(error, error_type):
            if is_coroutine_func(handler):
                return await handler(error)
            return handler(error)
    logging.error(f"Unhandled error: {error}")


def max_concurrent(
    func: Callable[..., T], limit: int
) -> Callable[..., Callable[..., T]]:
    """
    Limit the concurrency of function execution using a semaphore.

    Args:
        func: The function to limit concurrency for.
        limit: The maximum number of concurrent executions.

    Returns:
        The function wrapped with concurrency control.
    """
    if not is_coroutine_func(func):
        func = force_async(func)
    semaphore = asyncio.Semaphore(limit)

    @wraps(func)
    async def wrapper(*args, **kwargs):
        async with semaphore:
            return await func(*args, **kwargs)

    return wrapper


def throttle(func: Callable[..., T], period: float) -> Callable[..., Callable[..., T]]:
    """
    Throttle function execution to limit the rate of calls.

    Args:
        func: The function to throttle.
        period: The minimum time interval between consecutive calls.

    Returns:
        The throttled function.
    """
    if not is_coroutine_func(func):
        func = force_async(func)
    throttle_instance = Throttle(period)

    @wraps(func)
    async def wrapper(*args, **kwargs):
        await throttle_instance(func)(*args, **kwargs)
        return await func(*args, **kwargs)

    return wrapper
