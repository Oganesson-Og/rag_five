#!/usr/bin/env python
"""
Test script to verify the pipeline is working correctly.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

async def main():
    """Run a test of the pipeline."""
    try:
        from src.rag.models import Document, ContentModality
        from src.rag.pipeline import Pipeline
        
        # Create a simple configuration
        config = {
            'cache': {'enabled': False},
            'extraction': {'enabled': True},
            'chunking': {'enabled': True},
            'diagram': {'enabled': False},  # Disable diagram analysis to simplify testing
            'educational': {'enabled': False},  # Disable educational processing to simplify testing
            'feedback': {'enabled': False}  # Disable feedback processing to simplify testing
        }
        
        # Initialize the pipeline
        print("Initializing pipeline...")
        pipeline = Pipeline(config)
        
        # Create a simple text document
        text_content = """
        This is a simple test document for the RAG pipeline.
        It contains some basic text content that can be processed.
        The pipeline should be able to extract and chunk this content.
        """
        
        print("Processing document...")
        # Process the document
        document = await pipeline.process_document(
            source=text_content,
            modality=ContentModality.TEXT,
            options={"debug": True}
        )
        
        print(f"\nDocument processed successfully: {document.id}")
        print(f"Modality: {document.modality}")
        if hasattr(document, 'processing_time'):
            print(f"Processing time: {document.processing_time:.2f} seconds")
        
        # Print processing events
        if hasattr(document, 'processing_events'):
            print("\nProcessing Events:")
            for event in document.processing_events:
                print(f"  - {event.stage}: {event.processor}")
                if event.metrics and hasattr(event.metrics, 'processing_time'):
                    print(f"    Time: {event.metrics.processing_time:.2f}s")
        
        # Print chunks
        if hasattr(document, 'chunks') and document.chunks:
            print(f"\nChunks: {len(document.chunks)}")
            for i, chunk in enumerate(document.chunks[:3]):  # Show first 3 chunks
                print(f"  - Chunk {i+1}: {chunk.text[:50]}...")
        
        return document
            
    except Exception as e:
        logging.error(f"Error in test script: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    asyncio.run(main()) 