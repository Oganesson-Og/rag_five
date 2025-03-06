#!/usr/bin/env python3
"""
Async Call Tracer
----------------

Utility script to trace async calls during pipeline execution and diagnose 
await-related issues in real-time.

Usage:
    python3 trace_async_calls.py <path_to_document>

Features:
- Trace all await expressions
- Monitor coroutine state changes
- Detect dictionary awaits
- Log exceptions with context
"""

import os
import sys
import asyncio
import inspect
import logging
import traceback
from typing import Dict, Any, Optional, Union, Callable
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.rag.pipeline import Pipeline
from src.rag.models import ContentModality, Document
from src.utils.debug_utils import is_awaitable, is_coroutine_function, safe_await


# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("async_trace.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("async_tracer")


class AsyncCallTracer:
    """Traces async calls during execution."""
    
    def __init__(self):
        self.call_stack = []
        self.active_coroutines = {}
        self.traced_methods = set()
        
    def _log_call(self, func_name: str, args: tuple, kwargs: dict):
        """Log function call."""
        args_str = ", ".join([str(arg) for arg in args])
        kwargs_str = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
        logger.debug(f"CALL: {func_name}({args_str}{', ' if args_str and kwargs_str else ''}{kwargs_str})")
        
    def _log_return(self, func_name: str, result: Any):
        """Log function return value."""
        result_type = type(result).__name__
        result_str = str(result)
        if len(result_str) > 100:
            result_str = result_str[:97] + "..."
        logger.debug(f"RETURN: {func_name} -> {result_type}: {result_str}")
        
    async def _wrap_coroutine(self, coro, func_name: str):
        """Wrap a coroutine to trace its execution."""
        coro_id = id(coro)
        self.active_coroutines[coro_id] = func_name
        logger.debug(f"START COROUTINE: {func_name} (id={coro_id})")
        
        try:
            result = await coro
            logger.debug(f"END COROUTINE: {func_name} (id={coro_id})")
            self._log_return(func_name, result)
            return result
        except Exception as e:
            logger.error(f"EXCEPTION IN COROUTINE: {func_name} - {str(e)}", exc_info=True)
            raise
        finally:
            if coro_id in self.active_coroutines:
                del self.active_coroutines[coro_id]
    
    def patch_method(self, obj: Any, method_name: str):
        """Patch a method to trace its calls."""
        if not hasattr(obj, method_name):
            logger.warning(f"Object {obj} has no method {method_name}")
            return
            
        original_method = getattr(obj, method_name)
        func_path = f"{obj.__class__.__name__}.{method_name}"
        
        if func_path in self.traced_methods:
            return
            
        self.traced_methods.add(func_path)
        
        # Check if it's an async method
        if asyncio.iscoroutinefunction(original_method):
            async def traced_async_method(*args, **kwargs):
                self._log_call(func_path, args, kwargs)
                try:
                    coro = original_method(*args, **kwargs)
                    return await self._wrap_coroutine(coro, func_path)
                except Exception as e:
                    logger.error(f"EXCEPTION: {func_path} - {str(e)}", exc_info=True)
                    raise
            
            setattr(obj, method_name, traced_async_method)
        else:
            # Regular method
            def traced_method(*args, **kwargs):
                self._log_call(func_path, args, kwargs)
                try:
                    result = original_method(*args, **kwargs)
                    self._log_return(func_path, result)
                    
                    # Check if returned a coroutine that needs to be awaited
                    if inspect.iscoroutine(result):
                        logger.warning(f"WARNING: {func_path} returned a coroutine but is not an async function")
                    
                    return result
                except Exception as e:
                    logger.error(f"EXCEPTION: {func_path} - {str(e)}", exc_info=True)
                    raise
            
            setattr(obj, method_name, traced_method)
    
    def patch_object(self, obj: Any, method_names: Optional[list] = None):
        """Patch all methods of an object."""
        if method_names is None:
            # Find all methods
            method_names = []
            for attr_name in dir(obj):
                if not attr_name.startswith("_"):  # Skip private methods
                    attr = getattr(obj, attr_name)
                    if callable(attr):
                        method_names.append(attr_name)
        
        for method_name in method_names:
            self.patch_method(obj, method_name)
    
    def patch_pipeline(self, pipeline: Pipeline):
        """Patch a pipeline and all its stages for tracing."""
        # Patch pipeline methods
        self.patch_object(pipeline, ["process_document", "generate_response", "_cache_document", 
                                   "_retrieve_chunks", "_get_embedding"])
        
        # Patch all stages
        for stage_key, stage in pipeline.stages.items():
            logger.info(f"Patching stage: {stage_key}")
            self.patch_object(stage, ["process"])
            
            # If it's an extraction stage, patch all extractors
            if hasattr(stage, "extractors"):
                for modality, extractor in stage.extractors.items():
                    logger.info(f"Patching extractor: {modality}")
                    self.patch_object(extractor, ["extract", "_flush_batch", "_process_request_batch"])


async def trace_document_processing(document_path: str, tracer: AsyncCallTracer):
    """Process a document with tracing enabled."""
    logger.info(f"Processing document: {document_path}")
    
    try:
        # Load the pipeline
        pipeline = Pipeline("config.yaml")
        
        # Apply tracing
        tracer.patch_pipeline(pipeline)
        
        # Process document
        doc_path = Path(document_path)
        if not doc_path.exists():
            logger.error(f"Document not found: {document_path}")
            return
            
        # Detect modality
        ext = doc_path.suffix.lower()
        modality = None
        if ext == ".pdf":
            modality = ContentModality.PDF
        elif ext in [".txt", ".md"]:
            modality = ContentModality.TEXT
        elif ext in [".png", ".jpg", ".jpeg"]:
            modality = ContentModality.IMAGE
        else:
            modality = ContentModality.BINARY
            
        logger.info(f"Detected modality: {modality}")
        
        # Process the document
        result = await pipeline.process_document(
            source=str(doc_path),
            modality=modality
        )
        
        logger.info(f"Processing completed: {result.id if hasattr(result, 'id') else 'unknown'}")
        return result
        
    except Exception as e:
        logger.error(f"Error in document processing: {str(e)}", exc_info=True)
        raise


def main():
    """Run the async tracer on a document."""
    if len(sys.argv) < 2:
        print("Usage: python3 trace_async_calls.py <path_to_document>")
        return
        
    document_path = sys.argv[1]
    tracer = AsyncCallTracer()
    
    # Patch asyncio.iscoroutinefunction to add more debugging
    original_iscoro = asyncio.iscoroutinefunction
    
    def debug_iscoroutinefunction(func):
        is_coro = original_iscoro(func)
        if not is_coro and hasattr(func, "__name__") and "analyze" in func.__name__:
            logger.warning(f"CHECK: {func.__name__} is NOT a coroutine function")
        return is_coro
    
    asyncio.iscoroutinefunction = debug_iscoroutinefunction
    
    try:
        # Run the tracer
        asyncio.run(trace_document_processing(document_path, tracer))
    except Exception as e:
        logger.error(f"Trace failed: {str(e)}", exc_info=True)
    finally:
        # Restore original function
        asyncio.iscoroutinefunction = original_iscoro


if __name__ == "__main__":
    main() 