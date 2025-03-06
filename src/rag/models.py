"""
Enhanced Unified Data Model
--------------------------------

Core data model system designed for RAG pipeline operations, providing comprehensive
data structures for document processing, embedding management, and result tracking.

Key Features:
- Multi-modal content support (text, audio, image, PDF, video)
- Comprehensive document lifecycle tracking
- Flexible metadata management
- Embedding vector representations
- Chunk-level processing and relationships
- Search and generation result structures
- Processing metrics and event logging

Technical Details:
- Built on Pydantic for robust validation
- UUID-based unique identifiers
- Timestamp tracking for all operations
- Modular enum-based stage management
- Flexible metadata schemas
- Forward reference handling

Dependencies:
- pydantic>=2.5.0
- numpy>=1.24.0
- python-dateutil>=2.8.2

Example Usage:
    # Create a new document
    doc = Document(
        content="Sample text",
        modality=ContentModality.TEXT,
        metadata={"language": "en", "encoding": "utf-8"}
    )

    # Add processing event
    doc.add_processing_event(ProcessingEvent(
        stage=ProcessingStage.EXTRACTED,
        processor="TextExtractor"
    ))

    # Create and link chunks
    chunk = Chunk(
        document_id=doc.id,
        text="Sample chunk",
        start_pos=0,
        end_pos=12
    )

Performance Considerations:
- Optimized for frequent updates
- Efficient metadata validation
- Minimal memory footprint
- Fast serialization/deserialization

Author: Keith Satuku
Version: 1.0.0
Created: 2025
License: MIT
"""

from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from enum import Enum, auto
from pydantic import BaseModel, Field, field_validator, validator
import numpy as np
from uuid import uuid4
from dataclasses import dataclass, field
import hashlib

class ProcessingStage(str, Enum):
    """Pipeline processing stages."""
    EXTRACTED = "extracted"
    CHUNKED = "chunked"
    ANALYZED = "analyzed"
    EDUCATIONAL_PROCESSED = "educational_processed"
    FEEDBACK_PROCESSED = "feedback_processed"

class ContentModality(Enum):
    """Content modality types."""
    TEXT = 'text'
    PDF = 'pdf'
    AUDIO = 'audio'
    IMAGE = 'image'
    DOCX = 'docx'
    EXCEL = 'excel'
    CSV = 'csv'
    UNKNOWN = 'unknown'
    
    @property
    def required_metadata(self) -> List[str]:
        """Required metadata fields for each modality."""
        return {
            "text": ["encoding", "language"],
            "audio": ["duration", "sample_rate", "channels"],
            "image": ["width", "height", "format"],
            "pdf": ["pages", "title"],
            "video": ["duration", "fps", "resolution"]
        }[self.value]

class ProcessingMetrics(BaseModel):
    """Metrics collected during document processing."""
    processing_time: float
    token_count: Optional[int] = None
    chunk_count: Optional[int] = None
    embedding_dimensions: Optional[int] = None
    confidence_score: Optional[float] = None
    error_rate: Optional[float] = None

class ProcessingEvent(BaseModel):
    """Detailed processing event information."""
    timestamp: datetime = Field(default_factory=datetime.now)
    stage: ProcessingStage
    processor: str
    metrics: Optional[ProcessingMetrics] = None
    config_snapshot: Dict[str, Any] = Field(default_factory=dict)
    status: str = "success"
    error: Optional[str] = None

class Chunk(BaseModel):
    """Enhanced chunk representation with position and relations."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str
    text: str
    start_pos: int
    end_pos: int
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embeddings: Optional[List[float]] = None
    relationships: Dict[str, List[str]] = Field(default_factory=dict)
    confidence_score: Optional[float] = None
    
    def update_embedding(self, embedding: List[float]):
        """Update chunk embedding with new vector."""
        self.embeddings = embedding
        self.metadata["last_embedded"] = datetime.now().isoformat()

class EmbeddingVector(BaseModel):
    """Vector embedding with metadata."""
    vector: List[float]
    model: str
    dimension: int
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class Document(BaseModel):
    """Document representation with metadata."""
    content: Any  # Can be str, bytes, or other content types
    source: str = "unknown"  # Default to "unknown" if not provided
    modality: Optional[ContentModality] = ContentModality.UNKNOWN
    content_type: Optional[str] = None
    doc_info: Dict[str, Any] = Field(default_factory=dict)
    id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=datetime.now)
    processing_events: List[ProcessingEvent] = Field(default_factory=list)
    chunks: List[Chunk] = Field(default_factory=list)
    processing_time: Optional[float] = None
    
    class Config:
        # Allow arbitrary types for content field
        arbitrary_types_allowed = True
        
    @validator('content')
    def validate_content(cls, v):
        """Validate that content is not None and has appropriate type."""
        if v is None:
            raise ValueError("Content cannot be None")
        # Allow various content types (str, bytes, dict, etc.)
        return v
        
    @validator('source')
    def validate_source(cls, v):
        """Ensure source is a string."""
        if v is None:
            return "unknown"
        return str(v)
    
    def add_processing_event(self, event: ProcessingEvent):
        """Add a processing event to the document history."""
        self.processing_events.append(event)
    
    def add_chunk(self, chunk: Chunk):
        """Add a chunk to the document."""
        self.chunks.append(chunk)
    
    def has_multiple_modalities(self) -> bool:
        """Check if document contains multiple content modalities."""
        return len(set(chunk.metadata.get('modality') for chunk in self.chunks)) > 1
    
    @property
    def has_diagrams(self) -> bool:
        """Check if document contains diagrams."""
        return any(chunk.metadata.get('is_diagram', False) for chunk in self.chunks)
    
    @property
    def processed_modalities(self) -> List[str]:
        """Get list of processed content modalities."""
        return list(set(event.processor for event in self.processing_events))

    @property
    def content_hash(self) -> str:
        """
        Compute a hash of the document content.
        
        Returns:
            A string hash representation of the document content
        """
        # Convert content to bytes for hashing
        content_bytes = self.encode()
        
        # Compute hash
        return hashlib.sha256(content_bytes).hexdigest()

    def encode(self) -> bytes:
        """Convert content to bytes."""
        if isinstance(self.content, bytes):
            return self.content
        elif isinstance(self.content, str):
            return self.content.encode('utf-8')
        elif isinstance(self.content, Document):
            return str(self.content.content).encode('utf-8')
        else:
            return str(self.content).encode('utf-8')

class SearchResult(BaseModel):
    """Search result with relevance information."""
    chunk: Chunk
    similarity_score: float
    ranking_position: int
    metadata: Dict[str, Any] = Field(default_factory=dict)

class GenerationResult(BaseModel):
    """Generation result with source tracking."""
    text: str
    chunks_used: List[Chunk]
    confidence_score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now) 