"""
Document Cache Module
--------------------

This module provides a specialized cache for document objects in the RAG pipeline.
It extends the functionality of the advanced cache to handle document-specific caching needs.

Key Features:
- Document-specific caching
- Configurable cache size
- Efficient document retrieval
- Memory optimization for document storage
- Cache invalidation strategies

Technical Details:
- LRU cache implementation for documents
- Document metadata indexing
- Optimized memory usage
- Thread-safe operations

Dependencies:
- collections
- typing
- src.rag.models

Example Usage:
    # Create a document cache
    cache = DocumentCache(max_size=1000)
    
    # Cache a document
    cache.add(document_id, document)
    
    # Retrieve a document
    document = cache.get(document_id)
    
    # Check if document exists in cache
    if cache.contains(document_id):
        # Document exists
        pass

Performance Considerations:
- Memory usage scales with document size and count
- Efficient retrieval with O(1) complexity
- Automatic eviction of least recently used documents

Author: Keith Satuku
Version: 1.0.0
Created: 2023
License: MIT
"""

import logging
from collections import OrderedDict
from typing import Dict, Optional, Any, List, Set, Tuple
from threading import RLock

# Import Document model
try:
    from src.rag.models import Document
except ImportError:
    # For testing or standalone usage
    class Document:
        """Placeholder Document class if the actual model is not available."""
        def __init__(self, id: str, content: str, metadata: Dict[str, Any] = None):
            self.id = id
            self.content = content
            self.metadata = metadata or {}

logger = logging.getLogger(__name__)

class DocumentCache:
    """
    A specialized cache for Document objects with LRU eviction policy.
    
    This cache is designed to store Document objects efficiently and provide
    fast access while managing memory usage through size limits and LRU eviction.
    """
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize the document cache.
        
        Args:
            max_size: Maximum number of documents to store in the cache.
                     Defaults to 1000.
        """
        self._max_size = max_size
        self._cache: OrderedDict[str, Document] = OrderedDict()
        self._lock = RLock()  # Reentrant lock for thread safety
        self._metadata_index: Dict[str, Dict[str, Set[str]]] = {}
        logger.info(f"Initialized DocumentCache with max_size={max_size}")
    
    def add(self, document_id: str, document: Document) -> None:
        """
        Add a document to the cache.
        
        Args:
            document_id: Unique identifier for the document.
            document: The Document object to cache.
        """
        with self._lock:
            # If cache is full, remove the least recently used item
            if len(self._cache) >= self._max_size and document_id not in self._cache:
                self._evict_lru()
            
            # Add or update the document in the cache
            self._cache[document_id] = document
            # Move to end to mark as most recently used
            if document_id in self._cache:
                self._cache.move_to_end(document_id)
            
            # Index metadata for faster querying
            self._index_metadata(document_id, document)
            
            logger.debug(f"Added document {document_id} to cache")
    
    def get(self, document_id: str) -> Optional[Document]:
        """
        Retrieve a document from the cache.
        
        Args:
            document_id: The ID of the document to retrieve.
            
        Returns:
            The Document object if found, None otherwise.
        """
        with self._lock:
            if document_id in self._cache:
                # Move to end to mark as most recently used
                self._cache.move_to_end(document_id)
                logger.debug(f"Cache hit for document {document_id}")
                return self._cache[document_id]
            
            logger.debug(f"Cache miss for document {document_id}")
            return None
    
    def remove(self, document_id: str) -> bool:
        """
        Remove a document from the cache.
        
        Args:
            document_id: The ID of the document to remove.
            
        Returns:
            True if the document was removed, False if it wasn't in the cache.
        """
        with self._lock:
            if document_id in self._cache:
                document = self._cache[document_id]
                # Remove from metadata index
                self._remove_from_index(document_id, document)
                # Remove from cache
                del self._cache[document_id]
                logger.debug(f"Removed document {document_id} from cache")
                return True
            
            return False
    
    def contains(self, document_id: str) -> bool:
        """
        Check if a document is in the cache.
        
        Args:
            document_id: The ID of the document to check.
            
        Returns:
            True if the document is in the cache, False otherwise.
        """
        with self._lock:
            return document_id in self._cache
    
    def clear(self) -> None:
        """Clear all documents from the cache."""
        with self._lock:
            self._cache.clear()
            self._metadata_index.clear()
            logger.info("Document cache cleared")
    
    def get_size(self) -> int:
        """
        Get the current number of documents in the cache.
        
        Returns:
            The number of documents in the cache.
        """
        with self._lock:
            return len(self._cache)
    
    def get_max_size(self) -> int:
        """
        Get the maximum cache size.
        
        Returns:
            The maximum number of documents the cache can hold.
        """
        return self._max_size
    
    def set_max_size(self, max_size: int) -> None:
        """
        Set the maximum cache size and evict items if necessary.
        
        Args:
            max_size: The new maximum cache size.
        """
        with self._lock:
            self._max_size = max_size
            # Evict items if the cache is now too large
            while len(self._cache) > self._max_size:
                self._evict_lru()
            logger.info(f"Set max cache size to {max_size}")
    
    def find_by_metadata(self, key: str, value: Any) -> List[Document]:
        """
        Find documents by metadata key-value pair.
        
        Args:
            key: Metadata key to search for.
            value: Value to match.
            
        Returns:
            List of Document objects matching the criteria.
        """
        with self._lock:
            results = []
            # Convert value to string for consistent lookup
            str_value = str(value)
            
            # Check if we have an index for this key
            if key in self._metadata_index and str_value in self._metadata_index[key]:
                # Get document IDs from the index
                doc_ids = self._metadata_index[key][str_value]
                # Retrieve documents
                for doc_id in doc_ids:
                    if doc_id in self._cache:
                        results.append(self._cache[doc_id])
            
            return results
    
    def _evict_lru(self) -> None:
        """Remove the least recently used document from the cache."""
        if self._cache:
            # Get the first item (least recently used)
            doc_id, document = next(iter(self._cache.items()))
            # Remove from metadata index
            self._remove_from_index(doc_id, document)
            # Remove from cache
            self._cache.popitem(last=False)
            logger.debug(f"Evicted document {doc_id} from cache (LRU)")
    
    def _index_metadata(self, doc_id: str, document: Document) -> None:
        """
        Index document metadata for faster querying.
        
        Args:
            doc_id: Document ID.
            document: Document object to index.
        """
        if not hasattr(document, 'metadata') or not document.metadata:
            return
        
        # Index each metadata field
        for key, value in document.metadata.items():
            # Convert value to string for consistent storage
            str_value = str(value)
            
            # Initialize key in index if not exists
            if key not in self._metadata_index:
                self._metadata_index[key] = {}
            
            # Initialize value in key if not exists
            if str_value not in self._metadata_index[key]:
                self._metadata_index[key][str_value] = set()
            
            # Add document ID to the index
            self._metadata_index[key][str_value].add(doc_id)
    
    def _remove_from_index(self, doc_id: str, document: Document) -> None:
        """
        Remove document from metadata index.
        
        Args:
            doc_id: Document ID.
            document: Document object to remove from index.
        """
        if not hasattr(document, 'metadata') or not document.metadata:
            return
        
        # Remove from each metadata field index
        for key, value in document.metadata.items():
            str_value = str(value)
            
            if key in self._metadata_index and str_value in self._metadata_index[key]:
                self._metadata_index[key][str_value].discard(doc_id)
                
                # Clean up empty sets
                if not self._metadata_index[key][str_value]:
                    del self._metadata_index[key][str_value]
                
                # Clean up empty keys
                if not self._metadata_index[key]:
                    del self._metadata_index[key] 