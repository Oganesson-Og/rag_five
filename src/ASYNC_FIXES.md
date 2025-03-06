# Async/Await Fixes in RAG Pipeline

## Key Issues Fixed

1. **Awaiting Dictionary Methods**: Fixed issue where dictionary methods were being awaited incorrectly
   ```python
   # Before - Error: "object dict can't be used in 'await' expression"
   await document.doc_info['metrics']
   
   # After
   document.doc_info['metrics']  # No await needed for dict access
   ```

2. **Awaiting Non-Async Functions**: Fixed issues where non-async functions were being awaited
   ```python
   # Before - Error when process_diagram is not async
   await self.diagram_analyzer.process_diagram(document)
   
   # After - Using execute_function utility
   await execute_function(self.diagram_analyzer.process_diagram, document)
   ```

3. **Syntax Errors in Async Code**: Fixed various syntax errors in async code blocks

4. **Missing Properties in Document Class**: Added missing properties to Document class
   ```python
   # Added to Document class
   content_hash: Optional[str] = None
   processing_time: Optional[float] = None
   ```

5. **Redis Connection Error Handling**: Added robust error handling for Redis connection issues
   ```python
   # Before - No error handling for Redis connection issues
   self.cache.set(cache_key, document, 'document')
   
   # After - With proper error handling
   try:
       await execute_function(self.cache.set, cache_key, document, 'document')
   except Exception as e:
       self.logger.warning(f"Error caching document: {str(e)}")
       # If it's a Redis connection error, disable cache for future operations
       if "Connection refused" in str(e):
           self.logger.warning("Redis connection failed. Caching will be disabled.")
           if hasattr(self, 'config') and isinstance(self.config, dict):
               if 'cache' in self.config:
                   self.config['cache']['enabled'] = False
   ```

## Improvements Made

1. **Robust Async Utilities**: Added utilities for handling mixed sync/async code
   ```python
   # New utility functions
   async def execute_function(func, *args, **kwargs)
   async def await_if_coro(obj)
   def ensure_async(func)
   ```

2. **Enhanced Pipeline Stage Processing**: Improved handling of async/sync stages
   ```python
   # Proper handling of both async and sync stage processing
   if asyncio.iscoroutinefunction(stage.process):
       document = await stage.process(document)
   else:
       # Run sync function in thread pool
       with ThreadPoolExecutor() as executor:
           document = await asyncio.get_event_loop().run_in_executor(
               executor, stage.process, document
           )
   ```

3. **Improved Error Handling**: Added comprehensive error handling throughout the pipeline
   ```python
   try:
       # Process code
   except Exception as e:
       self.logger.error(f"Error in stage {stage_name}: {str(e)}", exc_info=True)
       # Don't stop the pipeline for non-critical stages
       if stage_name in [ProcessingStage.ANALYZED, ProcessingStage.EDUCATIONAL_PROCESSED, 
                        ProcessingStage.FEEDBACK_PROCESSED]:
           self.logger.warning(f"Continuing pipeline despite error in {stage_name}")
       else:
           raise Exception(f"Document processing failed: {str(e)}")
   ```

4. **Added Document Properties**: Added necessary properties to Document class
   - `content_hash` for caching
   - `processing_time` for metrics

5. **Graceful Degradation**: Added ability to disable features when services are unavailable
   ```python
   # Disable caching if Redis is unavailable
   if "Connection refused" in str(e):
       self.logger.warning("Redis connection failed. Caching will be disabled.")
       if hasattr(self, 'config') and isinstance(self.config, dict):
           if 'cache' in self.config:
               self.config['cache']['enabled'] = False
   ```

## Testing Scripts

1. **End-to-End Pipeline Test**:
   ```bash
   python src/scripts/test_pipeline.py
   ```

2. **Verify content_hash Property**:
   ```python
   # Check if content_hash is properly generated
   document = Document(content="Test content")
   print(f"Content hash: {document.content_hash}")
   ```

3. **Debug Async Issues**:
   ```bash
   python src/scripts/debug_async_issues.py
   ```

4. **Trace Async Calls**:
   ```bash
   python src/scripts/trace_async_calls.py path/to/document.pdf
   ```

## Best Practices for Async Code

1. **Check if Function is Async Before Awaiting**:
   ```python
   if asyncio.iscoroutinefunction(func):
       result = await func()
   else:
       result = func()
   ```

2. **Use ThreadPoolExecutor for CPU-bound Sync Functions**:
   ```python
   with ThreadPoolExecutor() as executor:
       result = await asyncio.get_event_loop().run_in_executor(
           executor, cpu_bound_function, *args
       )
   ```

3. **Handle Errors in Cache Operations Robustly**:
   ```python
   try:
       # Cache operation
       await execute_function(self.cache.set, key, value)
   except Exception as e:
       # Log but continue
       logger.warning(f"Cache operation failed: {str(e)}")
   ```

4. **Gracefully Degrade When Services are Unavailable**:
   ```python
   # Example: Disable caching if Redis is unavailable
   if "Connection refused" in str(e):
       self.logger.warning("Redis connection failed. Caching will be disabled.")
       self.config['cache']['enabled'] = False
   ```

5. **Use Proper Async Debugging Tools**:
   - Use `src/utils/debug_utils.py` for debugging async issues
   - Use `src/scripts/trace_async_calls.py` for tracing async execution

## Conclusion

The RAG pipeline now handles async/await operations correctly, with proper error handling and graceful degradation when services are unavailable. The pipeline can process documents efficiently, with comprehensive logging and metrics collection at each stage. 