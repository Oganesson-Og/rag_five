"""
Debug Utilities
--------------

Utilities for debugging async/await and other common issues.

Contains functions to help identify and debug:
- Async/await mismatches
- Dict awaiting issues
- Object awaiting issues
- Coroutine leaks
"""

import inspect
import asyncio
import logging
import traceback
import functools
from typing import Any, Callable, Dict, List, Optional, Union, Coroutine

logger = logging.getLogger(__name__)

def is_awaitable(obj: Any) -> bool:
    """Check if an object is awaitable."""
    return inspect.isawaitable(obj)

def is_coroutine_function(func: Callable) -> bool:
    """Check if a function is a coroutine function (defined with async def)."""
    return asyncio.iscoroutinefunction(func)

async def safe_await(obj: Any, fallback: Any = None) -> Any:
    """
    Safely await an object, handling cases where it might not be awaitable.
    
    Args:
        obj: Object to await
        fallback: Value to return if obj is not awaitable
        
    Returns:
        Result of awaiting obj if awaitable, otherwise fallback
    """
    if inspect.isawaitable(obj):
        return await obj
    logger.warning(f"Object of type {type(obj)} is not awaitable, returning fallback")
    return fallback

def debug_async_call(func_name: str, *args, **kwargs) -> None:
    """
    Debug an async function call.
    
    Args:
        func_name: Name of the function being called
        *args: Positional arguments
        **kwargs: Keyword arguments
    """
    call_stack = traceback.format_stack()[:-1]
    logger.debug(f"Async call to {func_name}")
    logger.debug(f"Args: {args}")
    logger.debug(f"Kwargs: {kwargs}")
    logger.debug(f"Call stack:\n{''.join(call_stack)}")

async def check_awaitable_properties(obj: Any) -> Dict[str, bool]:
    """
    Check properties of an object to see which are awaitable.
    
    Args:
        obj: Object to check
        
    Returns:
        Dict mapping property names to whether they are awaitable
    """
    results = {}
    for name in dir(obj):
        if name.startswith('_'):
            continue
        try:
            attr = getattr(obj, name)
            if callable(attr):
                results[f"{name}()"] = asyncio.iscoroutinefunction(attr)
            else:
                results[name] = inspect.isawaitable(attr)
        except Exception:
            results[name] = False
    return results

def debug_dict_operations(dict_obj: Dict[str, Any]) -> None:
    """
    Debug dict operations to catch common issues.
    
    Args:
        dict_obj: Dictionary to debug
    """
    for key, value in dict_obj.items():
        logger.debug(f"Key: {key}, Value type: {type(value)}")
        if inspect.isawaitable(value):
            logger.warning(f"Value for key '{key}' is awaitable but not being awaited")

class AsyncDebugWrapper:
    """
    Wrapper for objects to debug async/await issues.
    
    Usage:
        obj = AsyncDebugWrapper(real_obj)
        result = await obj.some_method()  # This will log debug info
    """
    
    def __init__(self, wrapped_obj: Any):
        self.wrapped_obj = wrapped_obj
        self.logger = logging.getLogger(f"AsyncDebug:{type(wrapped_obj).__name__}")
        
    def __getattr__(self, name: str) -> Any:
        attr = getattr(self.wrapped_obj, name)
        
        # If it's a callable, wrap it to provide debugging
        if callable(attr):
            @functools.wraps(attr)
            async def wrapped_method(*args, **kwargs):
                self.logger.debug(f"Calling {name} with args={args}, kwargs={kwargs}")
                try:
                    result = attr(*args, **kwargs)
                    # Check if result is awaitable
                    if inspect.isawaitable(result):
                        self.logger.debug(f"{name} returned awaitable object")
                        return await result
                    else:
                        self.logger.debug(f"{name} returned non-awaitable: {type(result)}")
                        return result
                except Exception as e:
                    self.logger.error(f"Error in {name}: {str(e)}", exc_info=True)
                    raise
            
            return wrapped_method
        
        # For non-callables, just return the attribute
        return attr 