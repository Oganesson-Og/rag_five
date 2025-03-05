"""
Base Document Extractor Module
---------------------------

Base classes and utilities for document content extraction.

Key Features:
- Common extraction interfaces
- Shared utilities
- Error handling
- Type definitions
- Metadata management
- Validation tools

Technical Details:
- Abstract base classes
- Type annotations
- Error management
- Metadata handling
- Content validation
- Logging integration

Dependencies:
- abc (standard library)
- typing (standard library)
- logging (standard library)
- datetime (standard library)
- numpy>=1.24.0

Author: Keith Satuku
Version: 2.0.0
Created: 2025
License: MIT
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
import numpy as np
from numpy.typing import NDArray
import logging
from datetime import datetime
from .models import Document
from .metrics import MetricsCollector

# Type aliases
ImageArray = NDArray[np.uint8]
ExtractorResult = Dict[str, Any]
DocumentContent = Union[str, bytes, NDArray[np.uint8]]

class BaseExtractor(ABC):
    """Base class for all document extractors."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize base extractor.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.metrics = MetricsCollector()
        
        # Initialize batching and rate limiting
        self.batch_config = self.config.get('request_batching', {
            'enabled': True,
            'batch_size': 5,
            'buffer_time': 5,
            'max_retries': 3
        })
        self.last_request_time = 0
        self.current_batch = []

    @abstractmethod
    async def extract(self, document: 'Document') -> 'Document':
        """Extract content from document.
        
        Args:
            document: Document to process
            
        Returns:
            Processed document
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement extract()")
    
    async def _process_batch(self, batch: List[Any], processor_func: callable) -> List[Any]:
        """Process a batch of items with rate limiting.
        
        Args:
            batch: List of items to process
            processor_func: Function to process each item
            
        Returns:
            List of processed results
        """
        import asyncio
        import time
        from concurrent.futures import ThreadPoolExecutor
        
        results = []
        
        for item in batch:
            # Apply rate limiting
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            
            if time_since_last < self.batch_config['buffer_time']:
                await asyncio.sleep(self.batch_config['buffer_time'] - time_since_last)
            
            # Process with retries
            retry_count = 0
            while retry_count < self.batch_config['max_retries']:
                try:
                    if asyncio.iscoroutinefunction(processor_func):
                        result = await processor_func(item)
                    else:
                        with ThreadPoolExecutor() as executor:
                            result = await asyncio.get_event_loop().run_in_executor(
                                executor, processor_func, item
                            )
                    
                    results.append(result)
                    self.last_request_time = time.time()
                    break
                    
                except Exception as e:
                    retry_count += 1
                    if retry_count == self.batch_config['max_retries']:
                        self.logger.error(f"Processing failed after {retry_count} retries: {str(e)}")
                        raise
                    await asyncio.sleep(self.batch_config['buffer_time'] * (2 ** retry_count))
            
        return results

    async def _add_to_batch(self, item: Any, processor_func: callable) -> Optional[Any]:
        """Add an item to the current batch.
        
        Args:
            item: Item to process
            processor_func: Function to process the item
            
        Returns:
            Processed result if batch is full, None otherwise
        """
        self.current_batch.append(item)
        
        if len(self.current_batch) >= self.batch_config['batch_size']:
            results = await self._process_batch(self.current_batch, processor_func)
            self.current_batch = []
            return results
        
        return None

    async def _flush_batch(self, processor_func: callable) -> List[Any]:
        """Process any remaining items in the current batch.
        
        Args:
            processor_func: Function to process each item
            
        Returns:
            List of processed results
        """
        if self.current_batch:
            results = await self._process_batch(self.current_batch, processor_func)
            self.current_batch = []
            return results
        return []

    def _clean_text(self, text: str) -> str:
        """Basic text cleaning shared by all extractors."""
        if not text:
            return ""
        return " ".join(text.split())

    def get_metadata(self) -> Dict[str, Any]:
        """Get extractor metadata."""
        return {
            'extractor': self.__class__.__name__,
            'timestamp': datetime.now().isoformat(),
            'batch_config': self.batch_config
        }

    def validate_content(self, content: DocumentContent) -> bool:
        """Validate input content."""
        if content is None:
            return False
        if isinstance(content, (str, bytes)):
            return len(content) > 0
        if isinstance(content, np.ndarray):
            return content.size > 0
        return False

    def _record_metrics(self, document: Document):
        """Record extraction metrics."""
        self.metrics.record(
            document_id=document.id,
            extractor=self.__class__.__name__,
            timestamp=datetime.now(),
            batch_size=len(self.current_batch),
            request_count=self.metrics.get('request_count', 0) + 1
        ) 