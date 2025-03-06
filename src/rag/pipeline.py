"""
Enhanced RAG Pipeline
--------------------------------

Comprehensive pipeline implementation for processing documents through multiple
stages with robust error handling and metric tracking.

Key Features:
- Modular processing stages
- Multi-modal document support
- Comprehensive error handling
- Metric collection and tracking
- Caching integration
- Asynchronous processing
- Flexible configuration

Technical Details:
- Async/await pattern implementation
- Stage-based processing architecture
- Integrated caching system
- Vector storage capabilities
- Comprehensive logging
- Metric collection at each stage

Dependencies:
- asyncio>=3.4.3
- logging>=2.0.0
- pydantic>=2.5.0
- numpy>=1.24.0

Example Usage:
    # Initialize pipeline
    pipeline = Pipeline(config={})
    
    # Process document
    document = await pipeline.process_document(
        source="path/to/doc",
        modality=ContentModality.TEXT
    )
    
    # Generate response
    result = await pipeline.generate_response(
        query="Sample query"
    )

Performance Considerations:
- Asynchronous processing for better throughput
- Efficient caching mechanisms
- Optimized error handling
- Configurable processing stages

Author: Keith Satuku
Version: 1.0.0
Created: 2025
License: MIT
"""

import logging
import time
import asyncio
from typing import List, Dict, Any, Optional, Union, Set
from pathlib import Path
from datetime import datetime
import yaml
import torch
import ollama
from uuid import uuid4
import sys
import os
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from .models import (
    Document,
    Chunk,
    ProcessingStage,
    ContentModality,
    ProcessingEvent,
    ProcessingMetrics,
    SearchResult,
    GenerationResult
)
from .prompt_engineering import PromptGenerator
from ..cache.advanced_cache import MultiModalCache, CacheConfig
from ..cache.vector_cache import VectorCache
from ..utils.metrics import MetricsCollector
from ..chunking.utils.text import clean_text, split_into_sentences
from ..chunking.utils.validation import validate_chunk, is_complete_sentence
from ..config.config_manager import ConfigManager
from ..config.embedding_config import EMBEDDING_CONFIG
from ..config.domain_config import get_domain_config
from ..document_processing.processors.diagram_analyzer import DiagramAnalyzer
from ..standards.education_standards_manager import StandardsManager
from ..feedback.feedback_processor import FeedbackProcessor
from ..document_processing.processors.math_processor import MathProcessor
from ..nlp.cross_modal_processor import CrossModalProcessor
from ..database.models import (
    DocumentDB,
    Chunk as DBChunk,
    QdrantVectorStore,
    Base,
    DatabaseConnection
)
from ..llm.model_manager import LLMManager, ModelError, ModelErrorType
from ..document_processing.extractors.base import BaseExtractor
from ..document_processing.extractors.audio import AudioExtractor
from ..document_processing.extractors.text import TextExtractor
from ..document_processing.extractors.image import ImageExtractor
from ..document_processing.extractors.pdf import PDFExtractor
from ..document_processing.extractors.docx import DocxExtractor
from ..document_processing.extractors.spreadsheet import ExcelExtractor, CSVExtractor
from ..utils.async_utils import execute_function, ensure_async

logger = logging.getLogger(__name__)

class PipelineStage:
    """Base class for pipeline stages."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics = MetricsCollector()
    
    async def process(self, document: Document) -> Document:
        """Process document and update its state."""
        raise NotImplementedError
    
    def _record_metrics(self, document: Document, metrics: Union[ProcessingMetrics, MetricsCollector]):
        """Record processing metrics."""
        # Convert MetricsCollector to ProcessingMetrics if needed
        if isinstance(metrics, MetricsCollector):
            # Get the average processing time
            processing_time = metrics.get_average_time() if hasattr(metrics, 'get_average_time') else 0.0
            
            # Create ProcessingMetrics from MetricsCollector
            metrics_obj = ProcessingMetrics(
                processing_time=processing_time,
                token_count=metrics.get_counter('tokens') if hasattr(metrics, 'get_counter') else None,
                chunk_count=metrics.get_counter('chunks') if hasattr(metrics, 'get_counter') else None
            )
        else:
            metrics_obj = metrics
            
        event = ProcessingEvent(
            stage=self.stage,
            processor=self.__class__.__name__,
            metrics=metrics_obj,
            config_snapshot=self.config
        )
        document.add_processing_event(event)

class ExtractionStage(PipelineStage):
    """Document extraction stage."""
    
    stage = ProcessingStage.EXTRACTED
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize extraction stage."""
        super().__init__(config)
        
        # Get extractor-specific configs from root level
        pdf_config = config.get('pdf', {})
        audio_config = config.get('audio', {})
        text_config = config.get('text', {})
        image_config = config.get('image', {})
        docx_config = config.get('docx', {})
        excel_config = config.get('excel', {})
        csv_config = config.get('csv', {})
        
        # Configure acceleration settings from root level
        acceleration_config = config.get('acceleration', {
            'device': 'mps',
            'num_threads': 8
        })
        
        # Update PDF config with acceleration settings if not already present
        if 'acceleration' not in pdf_config:
            pdf_config['acceleration'] = acceleration_config
            
        # Ensure secondary API keys are properly configured for fallback
        model_config = config.get('model', {})
        
        # Configure picture annotation with secondary API key if available
        picture_config = pdf_config.get('picture_annotation', {})
        if 'second_api_key' not in picture_config and 'second_api_key' in model_config:
            picture_config['second_api_key'] = model_config['second_api_key']
            pdf_config['picture_annotation'] = picture_config
            
        # Configure layout recognition with secondary API key if available
        layout_config = pdf_config.get('layout', {})
        if 'second_api_key' not in layout_config and 'second_api_key' in model_config:
            layout_config['second_api_key'] = model_config['second_api_key']
            pdf_config['layout'] = layout_config
            
        # Initialize extractors with their configs
        self.extractors = {
            ContentModality.PDF: PDFExtractor(config=pdf_config),
            ContentModality.AUDIO: AudioExtractor(config=audio_config),
            ContentModality.TEXT: TextExtractor(config=text_config),
            ContentModality.IMAGE: ImageExtractor(config=image_config),
            ContentModality.DOCX: DocxExtractor(config=docx_config),
            ContentModality.EXCEL: ExcelExtractor(config=excel_config),
            ContentModality.CSV: CSVExtractor(config=csv_config)
        }
    
    def _detect_modality(self, document: Document) -> ContentModality:
        """Detect document modality from content type or file extension."""
        # First check content type if available
        if document.content_type:
            content_type = document.content_type.lower()
            content_type_mapping = {
                'pdf': ContentModality.PDF,
                'image': ContentModality.IMAGE,
                'audio': ContentModality.AUDIO,
                'text': ContentModality.TEXT,
                'docx': ContentModality.DOCX,
                'excel': ContentModality.EXCEL,
                'csv': ContentModality.CSV
            }
            
            for key, modality in content_type_mapping.items():
                if key in content_type:
                    return modality
        
        # Fallback to file extension detection
        if isinstance(document.source, str):
            ext = document.source.lower().split('.')[-1]
            extension_mapping = {
                'pdf': ContentModality.PDF,
                'jpg': ContentModality.IMAGE,
                'jpeg': ContentModality.IMAGE,
                'png': ContentModality.IMAGE,
                'gif': ContentModality.IMAGE,
                'bmp': ContentModality.IMAGE,
                'mp3': ContentModality.AUDIO,
                'wav': ContentModality.AUDIO,
                'ogg': ContentModality.AUDIO,
                'txt': ContentModality.TEXT,
                'md': ContentModality.TEXT,
                'rst': ContentModality.TEXT,
                'docx': ContentModality.DOCX,
                'doc': ContentModality.DOCX,
                'xlsx': ContentModality.EXCEL,
                'xls': ContentModality.EXCEL,
                'csv': ContentModality.CSV
            }
            
            if ext in extension_mapping:
                return extension_mapping[ext]
        
        raise ValueError(f"Could not detect modality for document: {document.source}")
    
    async def process(self, document: Document) -> Document:
        """Process document through appropriate extractor."""
        try:
            # Detect modality if not set
            if not document.modality:
                modality = self._detect_modality(document)
                document.doc_info['modality'] = modality.value
            else:
                modality = document.modality
            
            # Get appropriate extractor
            extractor = self.extractors.get(modality)
            if not extractor:
                raise ValueError(f"No extractor found for modality: {modality}")
            
            # Process with metrics collection
            with self.metrics.measure_time() as timer:
                processed_document = await extractor.extract(document)
                
                # Record metrics
                self._record_metrics(processed_document, ProcessingMetrics(
                    processing_time=timer.elapsed,
                    token_count=len(str(processed_document.content)) if processed_document.content else 0
                ))
                
                return processed_document
                
        except Exception as e:
            logger.error(f"Extraction stage failed: {str(e)}")
            raise

class ChunkingStage(PipelineStage):
    """Document chunking stage."""
    
    stage = ProcessingStage.CHUNKED
    
    async def process(self, document: Document) -> Document:
        with self.metrics.measure_time() as timer:
            # Clean text
            cleaned_text = clean_text(document.content)
            
            # Split into sentences
            sentences = split_into_sentences(cleaned_text)
            
            # Create chunks
            chunks = []
            current_chunk = []
            current_pos = 0
            
            for sentence in sentences:
                current_chunk.append(sentence)
                chunk_text = " ".join(current_chunk)
                
                # Create a temporary chunk to validate
                # Using a simple object with a text attribute for validation
                class TempChunk:
                    def __init__(self, text):
                        self.text = text
                
                temp_chunk = TempChunk(chunk_text)
                
                if validate_chunk(temp_chunk):
                    # Calculate positions
                    start_pos = current_pos
                    end_pos = start_pos + len(chunk_text)
                    
                    # Create a proper chunk with all required fields
                    chunks.append({
                        "text": chunk_text,
                        "start_pos": start_pos,
                        "end_pos": end_pos
                    })
                    
                    # Update position for next chunk
                    current_pos = end_pos + 1  # +1 for the space between chunks
                    current_chunk = []
            
            # Create proper Chunk objects with all required fields
            document.chunks = [
                Chunk(
                    text=chunk["text"],
                    document_id=document.id,
                    start_pos=chunk["start_pos"],
                    end_pos=chunk["end_pos"]
                ) for chunk in chunks
            ]
            
        self._record_metrics(document, ProcessingMetrics(
            processing_time=timer.elapsed,
            chunk_count=len(chunks)
        ))
        return document

class DiagramAnalysisStage(PipelineStage):
    """Diagram analysis stage."""
    
    stage = ProcessingStage.ANALYZED
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Import here to avoid circular imports
        try:
            from ..document_processing.processors.diagram_analyzer import DiagramAnalyzer
            from ..utils.async_utils import ensure_async
            
            self.analyzer = DiagramAnalyzer(config)
            
            # Find the analysis method and ensure it's async-compatible
            if hasattr(self.analyzer, 'process_diagram'):
                self.analysis_method = ensure_async(self.analyzer.process_diagram)
                self.method_name = 'process_diagram'
            elif hasattr(self.analyzer, 'analyze'):
                self.analysis_method = ensure_async(self.analyzer.analyze)
                self.method_name = 'analyze'
            else:
                self.analysis_method = None
                self.method_name = None
                
            self.logger = logging.getLogger(__name__)
            self.logger.debug(f"DiagramAnalyzer initialized with method: {self.method_name}")
        except ImportError as e:
            self.logger.warning(f"DiagramAnalyzer import failed: {str(e)}")
            self.analyzer = None
            self.analysis_method = None
            self.method_name = None
    
    async def process(self, document: Document) -> Document:
        """Process document to analyze diagrams."""
        try:
            if not hasattr(document, 'has_diagrams') or not document.has_diagrams or not self.analyzer:
                return document
                
            self.logger.debug(f"Starting diagram analysis for document: {document.id if hasattr(document, 'id') else 'unknown'}")
            
            with self.metrics.measure_time() as timer:
                # Get the diagrams from the document
                diagrams = getattr(document, 'diagrams', [])
                if not diagrams:
                    self.logger.debug("No diagrams found in document")
                    return document
                    
                self.logger.debug(f"Processing {len(diagrams)} diagrams")
                
                # Process each diagram
                analyses = []
                for i, diagram in enumerate(diagrams):
                    self.logger.debug(f"Processing diagram {i+1}/{len(diagrams)}")
                    try:
                        if self.analysis_method:
                            analysis = await self.analysis_method(diagram)
                            analyses.append(analysis)
                        else:
                            self.logger.warning("No suitable analysis method found on DiagramAnalyzer")
                    except Exception as e:
                        self.logger.error(f"Error analyzing diagram {i+1}: {str(e)}", exc_info=True)
                        # Continue with other diagrams instead of stopping
                
                # Store results
                document.diagram_analysis = analyses
                self.logger.debug(f"Completed diagram analysis with {len(analyses)} results")
            
            return document
        except Exception as e:
            self.logger.error(f"Error in diagram analysis stage: {str(e)}", exc_info=True)
            # Don't fail the pipeline for diagram analysis errors
            return document

class EducationalProcessingStage(PipelineStage):
    """Educational content processing stage."""
    
    stage = ProcessingStage.EDUCATIONAL_PROCESSED
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.math_processor = MathProcessor(config.get('math', {}))
        self.standards_manager = StandardsManager(config.get('standards', {}))
        self.cross_modal_processor = CrossModalProcessor(config.get('cross_modal', {}))
        self.logger = logging.getLogger(__name__)
    
    async def process(self, document: Document) -> Document:
        """Process document with educational enhancements."""
        with self.metrics.measure_time() as timer:
            try:
                # Process mathematical content if present
                if self.math_processor.has_math_content(document.content):
                    # Use execute_function instead of directly awaiting
                    result = await execute_function(self.math_processor.process, document.content)
                    document.content = result
                
                # Map educational standards
                standards_mapping = await self.standards_manager.map_content(document.content)
                document.doc_info['standards'] = standards_mapping
                
                # Cross-modal educational processing
                if document.has_multiple_modalities():
                    await self.cross_modal_processor.process(document)
                
                # Create ProcessingMetrics manually with the required processing_time
                metrics = ProcessingMetrics(
                    processing_time=timer.elapsed,
                    token_count=len(str(document.content)) if document.content else 0
                )
                
                # Record metrics using our custom metrics object
                self._record_metrics(document, metrics)
                
                return document
            except Exception as e:
                self.logger.error(f"Error in {self.__class__.__name__}: {str(e)}", exc_info=True)
                document.add_processing_event(ProcessingEvent(
                    stage=self.stage,
                    processor=self.__class__.__name__,
                    status="error",
                    error=str(e)
                ))
                raise

class FeedbackProcessingStage(PipelineStage):
    """Educational feedback processing stage."""
    
    stage = ProcessingStage.FEEDBACK_PROCESSED
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.feedback_processor = FeedbackProcessor(config)
    
    async def process(self, document: Document) -> Document:
        """Process document with feedback analysis."""
        with self.metrics.measure_time() as timer:
            # Only process if document has educational content
            if 'standards' in document.doc_info:
                feedback = await self.feedback_processor.process(
                    document.content,
                    document.doc_info.get('standards', {})
                )
                
                # Add feedback to document
                document.doc_info['feedback'] = feedback
            
            # Record metrics
            self._record_metrics(document, ProcessingMetrics(
                processing_time=timer.elapsed,
                token_count=len(str(document.content)) if document.content else 0
            ))
            
            return document

class Pipeline:
    """RAG pipeline implementation."""
    
    def __init__(self, config: Union[str, Dict[str, Any], Path]):
        """Initialize pipeline with configuration.
        
        Args:
            config: Either a path to config file or config dict
        """
        try:
            # Handle config input
            if isinstance(config, (str, Path)):
                with open(config, 'r') as f:
                    self.config = yaml.safe_load(f)
            elif isinstance(config, dict):
                self.config = config
            else:
                raise ValueError(f"Invalid config type: {type(config)}")
            
            # Initialize logger
            self.logger = logging.getLogger(__name__)
            
            # Add request batching configuration
            self.request_config = {
                'batch_size': config.get('batch_size', 5),
                'buffer_time': config.get('buffer_time', 5),
                'max_retries': config.get('max_retries', 3)
            }
            
            # Initialize request tracking
            self.last_request_time = 0
            self.current_batch = []
            
            # Initialize components
            self._init_stages()
            self._init_cache()
            self._init_vector_store()
            self._init_other_components()
            
        except Exception as e:
            logger.error(f"Failed to initialize Pipeline: {str(e)}")
            logger.error(f"Config that failed: {self.config}")
            raise

    def _init_stages(self):
        """Initialize pipeline stages."""
        self.stages = {
            ProcessingStage.EXTRACTED: ExtractionStage(self.config),  # Pass the full config
            ProcessingStage.CHUNKED: ChunkingStage(self.config.get('chunking', {})),
            ProcessingStage.ANALYZED: DiagramAnalysisStage(self.config.get('diagram', {})),
            ProcessingStage.EDUCATIONAL_PROCESSED: EducationalProcessingStage(self.config.get('educational', {})),
            ProcessingStage.FEEDBACK_PROCESSED: FeedbackProcessingStage(self.config.get('feedback', {}))
        }

    def _init_cache(self):
        """Initialize caching system."""
        try:
            cache_config = self.config.get('cache', {})
            if cache_config.get('enabled', False):
                self.cache = MultiModalCache(config=cache_config)
                self.logger.info("Cache initialized successfully")
            else:
                self.cache = None
                self.logger.info("Cache disabled in configuration")
        except Exception as e:
            self.logger.error(f"Cache initialization failed: {str(e)}")
            self.cache = None

    def _init_vector_store(self):
        """Initialize vector store."""
        try:
            # Get vector store config from dict
            vector_store_config = self.config.get('vector_store', {})
            
            # Initialize QdrantVectorStore with config
            self.vector_store = QdrantVectorStore(
                collection_name=vector_store_config.get('collection_name', 'default'),
                dimension=vector_store_config.get('dimension', 1536),
                similarity_threshold=vector_store_config.get('similarity_threshold', 0.8),
                host=vector_store_config.get('host', 'localhost'),
                port=vector_store_config.get('port', 6333)
            )
            
            self.logger.info(f"Vector store initialized with collection: {vector_store_config.get('collection_name')}")
            
        except Exception as e:
            self.logger.error(f"Vector store initialization failed: {str(e)}")
            raise

    def _init_other_components(self):
        """Initialize other pipeline components."""
        try:
            # Initialize prompt generator
            prompt_config = self.config.get('components', {}).get('prompts', {})
            self.prompt_generator = PromptGenerator(prompt_config)
            
            # Initialize database connection
            db_config = self.config.get('database', {})
            self.db_connection = DatabaseConnection(
                connection_string=db_config.get('url', 'sqlite:///rag.db')
            )
            self.db_connection.connect()
            
            # Initialize Ollama client
            model_config = self.config.get('model', {})
            self.llm = ollama.Client(
                host=model_config.get('model_endpoint', 'http://localhost:11434')
            )
            
            # Test model availability
            try:
                model_name = model_config.get('model_name', 'deepseek-r1:32b')  # Default to qwen2.5 if not specified
                self.llm.chat(
                    model=model_name,
                    messages=[{'role': 'system', 'content': 'Test connection'}]
                )
                self.logger.info(f"Successfully connected to Ollama model: {model_name}")
            except Exception as e:
                raise ModelError(
                    message=f"Failed to connect to Ollama model: {str(e)}",
                    error_type=ModelErrorType.INITIALIZATION_ERROR,
                    provider="ollama"
                )
                
        except Exception as e:
            raise ModelError(
                message=f"Component initialization failed: {str(e)}",
                error_type=ModelErrorType.INITIALIZATION_ERROR,
                provider="pipeline"
            )

    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'db_connection'):
            self.db_connection.close()
        
    def _create_document(self, source: Union[str, Path, bytes], modality: ContentModality, options: Optional[Dict[str, Any]] = None) -> Document:
        """Create a document object from source.
        
        Args:
            source: Source content (file path, raw bytes, etc.)
            modality: Content modality
            options: Additional options
            
        Returns:
            Document object
        """
        try:
            # Handle file path
            if isinstance(source, (str, Path)) and Path(source).exists():
                try:
                    with open(source, 'rb') as f:
                        content = f.read()
                    source_path = str(source)
                    self.logger.debug(f"Successfully read file: {source}")
                except Exception as e:
                    self.logger.error(f"Failed to read file {source}: {str(e)}")
                    raise
            # Handle raw content
            else:
                content = source
                source_path = options.get('file_path', 'unknown') if options else 'unknown'
            
            # Create document with required fields
            document = Document(
                content=content,  # This must be the actual content (bytes or string)
                source=str(source_path),  # Must be a string
                modality=modality,
                doc_info={'options': options or {}}
            )
            
            return document
            
        except Exception as e:
            self.logger.error(f"Document creation failed: {str(e)}")
            raise

    async def process_document(
        self,
        source: Union[str, bytes, Document],
        modality: Optional[ContentModality] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Document:
        """
        Process a document through the pipeline.
        
        Args:
            source: Path to document, bytes, or Document object
            modality: Content modality hint (auto-detected if None)
            options: Processing options
            
        Returns:
            Processed Document object
        """
        options = options or {}
        start_time = time.time()
        
        try:
            # Convert source to Document if needed
            if not isinstance(source, Document):
                self.logger.debug(f"Creating document from source: {type(source)}")
                document = Document(
                    content=source,
                    modality=modality,
                    metadata=options.get('metadata', {})
                )
            else:
                document = source
                
            self.logger.info(f"Processing document: {document.id if hasattr(document, 'id') else 'unknown'}")
            self.logger.debug(f"Source type: {type(source)}")
            self.logger.debug(f"Detected modality: {document.modality}")
            
            # Check cache first if enabled
            if self.cache and self.config.get('cache', {}).get('enabled', False):
                try:
                    cache_key = f"doc:{document.content_hash}"
                    cached_doc = self.cache.get(cache_key, 'document')
                    if cached_doc:
                        self.logger.info(f"Retrieved document from cache: {document.id}")
                        return cached_doc
                except Exception as e:
                    # Log the error but continue without using cache
                    self.logger.warning(f"Error accessing document cache: {str(e)}")
            
            # Process through each stage
            for stage_name, stage in self.stages.items():
                try:
                    self.logger.debug(f"Running stage: {stage_name}")
                    # Fix: Call stage.process directly with proper async handling
                    if asyncio.iscoroutinefunction(stage.process):
                        document = await stage.process(document)
                    else:
                        # Run sync function in thread pool
                        with ThreadPoolExecutor() as executor:
                            document = await asyncio.get_event_loop().run_in_executor(
                                executor, stage.process, document
                            )
                    self.logger.debug(f"Completed stage: {stage_name}")
                except Exception as e:
                    self.logger.error(f"Error in stage {stage_name}: {str(e)}", exc_info=True)
                    # Don't stop the pipeline for non-critical stages
                    if stage_name in [ProcessingStage.ANALYZED, ProcessingStage.EDUCATIONAL_PROCESSED, 
                                     ProcessingStage.FEEDBACK_PROCESSED]:
                        self.logger.warning(f"Continuing pipeline despite error in {stage_name}")
                    else:
                        raise Exception(f"Document processing failed: {str(e)}")
            
            # Cache processed document
            if self.cache and self.config.get('cache', {}).get('enabled', False):
                try:
                    cache_key = f"doc:{document.content_hash}"
                    await execute_function(self.cache.set, cache_key, document, 'document')
                    self.logger.debug(f"Document {document.id} cached successfully")
                except Exception as e:
                    # Log the error but continue without caching
                    self.logger.warning(f"Error caching document: {str(e)}")
                    # If it's a Redis connection error, disable cache for future operations
                    if "Connection refused" in str(e):
                        self.logger.warning("Redis connection failed. Caching will be disabled.")
                        if hasattr(self, 'config') and isinstance(self.config, dict):
                            if 'cache' in self.config:
                                self.config['cache']['enabled'] = False
                
            # Record total processing time
            document.processing_time = time.time() - start_time
            
            return document
                
        except Exception as e:
            self.logger.error(f"Document processing failed: {str(e)}", exc_info=True)
            raise Exception(f"Document processing failed: {str(e)}")

    async def generate_response(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> GenerationResult:
        """Generate a response using RAG with rate limiting."""
        try:
            # Apply rate limiting
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.request_config['buffer_time']:
                await asyncio.sleep(self.request_config['buffer_time'] - time_since_last)
            
            model_config = self.config.get('model', {})
            model_name = model_config.get('model_name', 'deepseek-r1:32b')
            
            # Make the API request with retries
            max_retries = self.request_config['max_retries']
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    response = await self.llm.chat(
                        model=model_name,
                        messages=[
                            {
                                'role': 'system',
                                'content': self.prompt_generator.generate_system_prompt(context)
                            },
                            {
                                'role': 'user',
                                'content': query
                            }
                        ]
                    )
                    
                    self.last_request_time = time.time()
                    break
                    
                except Exception as e:
                    retry_count += 1
                    if "429" in str(e) or "rate limit" in str(e).lower():
                        if retry_count == max_retries:
                            raise
                        await asyncio.sleep(self.request_config['buffer_time'] * (2 ** retry_count))
                        continue
                    raise
            
            return GenerationResult(
                text=response.message.content,
                chunks_used=[],
                confidence_score=0.9,
                metadata={
                    "query": query,
                    "model": model_name,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            self.logger.error(f"Response generation failed: {str(e)}")
            raise
    
    async def _cache_document(self, document: Document):
        """Cache processed document."""
        if self.cache and self.config.get('cache', {}).get('enabled', False):
            try:
                await self.cache.set(
                    f"doc:{document.id}", 
                    document,
                    'document'
                )
                self.logger.debug(f"Document {document.id} cached successfully")
            except Exception as e:
                # Log the error but continue without caching
                self.logger.warning(f"Error caching document: {str(e)}")
                # If it's a Redis connection error, disable cache for future operations
                if "Connection refused" in str(e):
                    self.logger.warning("Redis connection failed. Caching will be disabled.")
                    if hasattr(self, 'config') and isinstance(self.config, dict):
                        if 'cache' in self.config:
                            self.config['cache']['enabled'] = False
                
    async def _retrieve_chunks(
        self,
        query: str,
        max_chunks: int = 5,
        similarity_threshold: float = 0.7,
        retrieval_strategy: str = "hybrid"
    ) -> List[SearchResult]:
        """
        Retrieve relevant chunks using multiple retrieval strategies.
        
        Args:
            query: Search query
            max_chunks: Maximum number of chunks to return
            similarity_threshold: Minimum similarity score threshold
            retrieval_strategy: Strategy to use ('semantic', 'keyword', 'hybrid')
            
        Returns:
            List of SearchResult objects
        """
        try:
            # Check cache first
            cache_key = f"search:{hash(query)}"
            cached_results = self.cache.get(cache_key, 'search') if self.cache else None
            if cached_results:
                return cached_results

            results = []
            
            if retrieval_strategy in ['semantic', 'hybrid']:
                # Get query embedding
                query_embedding = await self._get_embedding(query)
                
                # Semantic search using vector store
                semantic_results = await self.vector_store.search(
                    query_embedding,
                    k=max_chunks,
                    threshold=similarity_threshold
                )
                results.extend(semantic_results)

            if retrieval_strategy in ['keyword', 'hybrid']:
                # Keyword-based search
                keyword_results = await self._keyword_search(
                    query,
                    max_chunks=max_chunks
                )
                results.extend(keyword_results)

            # Deduplicate and rank results
            final_results = self._rank_and_deduplicate_results(
                results,
                max_chunks=max_chunks,
                strategy=retrieval_strategy
            )

            # Cache results
            await self.cache.set(cache_key, final_results, modality="search_results")
            
            return final_results

        except Exception as e:
            logger.error(f"Chunk retrieval failed: {str(e)}", exc_info=True)
            raise

    async def _keyword_search(
        self,
        query: str,
        max_chunks: int = 5
    ) -> List[SearchResult]:
        """Perform keyword-based search."""
        try:
            # Preprocess query
            processed_query = self._preprocess_query(query)
            
            # Get all chunks from cache
            chunks = await self.cache.get_all_chunks()
            
            # Calculate BM25 scores
            scores = []
            for chunk in chunks:
                score = self._calculate_bm25_score(processed_query, chunk.text)
                scores.append((chunk, score))
            
            # Sort by score and return top results
            scores.sort(key=lambda x: x[1], reverse=True)
            return [
                SearchResult(
                    chunk=chunk,
                    score=score,
                    strategy="keyword"
                ) for chunk, score in scores[:max_chunks]
            ]
            
        except Exception as e:
            logger.error(f"Keyword search failed: {str(e)}", exc_info=True)
            return []

    def _rank_and_deduplicate_results(
        self,
        results: List[SearchResult],
        max_chunks: int = 5,
        strategy: str = "hybrid"
    ) -> List[SearchResult]:
        """Rank and deduplicate search results."""
        try:
            # Remove duplicates based on chunk ID
            seen_chunks = set()
            unique_results = []
            
            for result in results:
                if result.chunk.id not in seen_chunks:
                    seen_chunks.add(result.chunk.id)
                    unique_results.append(result)

            # Adjust scores based on strategy
            if strategy == "hybrid":
                for result in unique_results:
                    if result.strategy == "semantic":
                        result.score *= 0.7  # Weight for semantic search
                    else:
                        result.score *= 0.3  # Weight for keyword search

            # Sort by score and return top results
            unique_results.sort(key=lambda x: x.score, reverse=True)
            return unique_results[:max_chunks]
            
        except Exception as e:
            logger.error(f"Result ranking failed: {str(e)}", exc_info=True)
            return results[:max_chunks]

    async def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using configured model."""
        try:
            # Check embedding cache
            cache_key = f"emb:{hash(text)}"
            cached_embedding = await self.vector_store.get_embedding(text)
            if cached_embedding is not None:
                return cached_embedding

            # Generate embedding using configured model
            embedding = await self.embedding_model.embed_text(text)
            
            # Cache embedding
            await self.vector_store.set_embedding(text, embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}", exc_info=True)
            raise

    def _preprocess_query(self, query: str) -> str:
        """
        Preprocess search query with tokenization, stopword removal, and normalization.
        
        Args:
            query: Raw query string
            
        Returns:
            Preprocessed query string
        """
        try:
            # Import NLTK components
            from nltk.tokenize import word_tokenize
            from nltk.corpus import stopwords
            from nltk.stem import WordNetLemmatizer
            
            # Download required NLTK data (if not already downloaded)
            import nltk
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            
            # Convert to lowercase
            query = query.lower()
            
            # Tokenize
            tokens = word_tokenize(query)
            
            # Remove stopwords
            stop_words = set(stopwords.words('english'))
            tokens = [token for token in tokens if token not in stop_words]
            
            # Lemmatize
            lemmatizer = WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(token) for token in tokens]
            
            # Remove special characters and numbers
            tokens = [token for token in tokens if token.isalnum()]
            
            return ' '.join(tokens)
            
        except Exception as e:
            self.logger.error(f"Query preprocessing failed: {str(e)}")
            return query.lower()  # Fallback to basic preprocessing

    def _calculate_bm25_score(self, query: str, text: str) -> float:
        """
        Calculate BM25 similarity score between query and text.
        
        Args:
            query: Preprocessed query string
            text: Document text to compare against
            
        Returns:
            BM25 similarity score
        """
        try:
            from rank_bm25 import BM25Okapi
            from nltk.tokenize import word_tokenize
            
            # Tokenize query and text
            query_tokens = word_tokenize(query)
            text_tokens = word_tokenize(text)
            
            # Create BM25 object with single document
            bm25 = BM25Okapi([text_tokens])
            
            # Calculate score
            score = bm25.get_scores(query_tokens)[0]
            
            return float(score)
            
        except Exception as e:
            self.logger.error(f"BM25 scoring failed: {str(e)}")
            return 0.0

    def _get_standards_alignment(
        self,
        text: str,
        grade_level: Optional[str]
    ) -> Dict[str, Any]:
        """Get educational standards alignment for generated response."""
        try:
            standards_stage = self.stages[ProcessingStage.EDUCATIONAL_PROCESSED]
            return standards_stage.standards_manager.get_alignment(
                text, grade_level
            )
        except Exception as e:
            logger.warning(f"Standards alignment failed: {str(e)}")
            return {}

    async def process_educational_session(
        self,
        student_id: str,
        content: str,
        session_id: Optional[str] = None
    ):
        """Process educational content with both vector and relational storage."""
        try:
            # Store vectors in Qdrant
            vector = await self.generate_embedding(content)
            qdrant_id = await self.vector_store.add_vector(
                vector=vector,
                metadata={"student_id": student_id}
            )
            
            # Store educational metadata in PostgreSQL
            with self.db_connection.get_cursor() as cursor:
                cursor.execute("""
                    INSERT INTO content_metadata 
                    (id, qdrant_id, content_type, educational_metadata)
                    VALUES (%s, %s, %s, %s)
                """, (
                    uuid4(),
                    qdrant_id,
                    "educational_content",
                    {
                        "student_id": student_id,
                        "session_id": session_id,
                        "timestamp": datetime.now().isoformat()
                    }
                ))
                
            return qdrant_id
            
        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")
            raise 

    def _detect_modality(self, source: Union[str, Path]) -> ContentModality:
        """Detect content modality from file extension."""
        if isinstance(source, str):
            source = Path(source)
            
        ext = source.suffix.lower()
        
        if ext in ['.pdf']:
            return ContentModality.PDF
        elif ext in ['.txt', '.md', '.rst']:
            return ContentModality.TEXT
        elif ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
            return ContentModality.IMAGE
        elif ext in ['.mp4', '.avi', '.mov', '.wmv']:
            return ContentModality.VIDEO
        elif ext in ['.mp3', '.wav', '.ogg', '.flac']:
            return ContentModality.AUDIO
        elif ext in ['.doc', '.docx']:
            return ContentModality.DOCX
        elif ext in ['.ppt', '.pptx']:
            return ContentModality.PPTX
        elif ext in ['.xls', '.xlsx']:
            return ContentModality.XLSX
        elif ext in ['.html', '.htm']:
            return ContentModality.HTML
        else:
            return ContentModality.UNKNOWN 