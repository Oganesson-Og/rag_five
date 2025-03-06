"""
Async utilities for handling mixed sync/async code.

This module provides utilities for working with both synchronous and
asynchronous functions in a unified way, making it easier to write
code that can handle both types of functions without errors.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from typing import Any, Callable, TypeVar, Union, Awaitable, cast

T = TypeVar('T')

async def await_if_coro(obj: Union[Awaitable[T], T]) -> T:
    """
    Await the object if it's a coroutine, otherwise return it directly.
    
    Args:
        obj: Either an awaitable object or a direct result
        
    Returns:
        The result value, awaited if necessary
    """
    if asyncio.iscoroutine(obj):
        return await obj
    return obj

async def execute_function(
    func: Callable[..., Union[Awaitable[T], T]], 
    *args: Any, 
    **kwargs: Any
) -> T:
    """
    Execute a function, handling both async and sync functions appropriately.
    
    Args:
        func: The function to execute, can be either sync or async
        *args: Positional arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        The result of the function, properly awaited if needed
    """
    if asyncio.iscoroutinefunction(func):
        # Function is async, await the result
        result = func(*args, **kwargs)
        if asyncio.iscoroutine(result):
            return await result
        return result
    else:
        # Function is sync, run in thread pool
        with ThreadPoolExecutor() as executor:
            return await asyncio.get_event_loop().run_in_executor(
                executor, 
                lambda: func(*args, **kwargs)
            )

def ensure_async(func: Callable[..., Union[Awaitable[T], T]]) -> Callable[..., Awaitable[T]]:
    """
    Decorator to ensure a function is async, wrapping sync functions if needed.
    
    Args:
        func: The function to wrap
        
    Returns:
        An async function that wraps the original
    """
    if asyncio.iscoroutinefunction(func):
        return func
        
    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> T:
        with ThreadPoolExecutor() as executor:
            return await asyncio.get_event_loop().run_in_executor(
                executor,
                lambda: func(*args, **kwargs)
            )
    
    return wrapper

def is_awaitable(obj: Any) -> bool:
    """
    Check if an object is awaitable (coroutine or has __await__).
    
    Args:
        obj: The object to check
        
    Returns:
        True if the object is awaitable, False otherwise
    """
    return asyncio.iscoroutine(obj) or hasattr(obj, "__await__") 