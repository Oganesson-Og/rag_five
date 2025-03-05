import asyncio
import logging
import sys
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.append(str(Path(__file__).parent))

from src.rag.pipeline import Pipeline
from src.rag.models import Document, ContentModality

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

async def test_pipeline():
    """Test the pipeline with a simple document."""
    try:
        # Create a simple configuration
        config = {
            "extraction": {
                "pdf": {},
                "text": {},
                "audio": {},
                "image": {},
                "docx": {},
                "excel": {},
                "csv": {}
            },
            "chunking": {
                "chunk_size": 1000,
                "chunk_overlap": 200
            },
            "diagram": {},
            "educational": {},
            "feedback": {},
            "cache": {
                "enabled": False
            },
            "vector_store": {
                "enabled": False
            }
        }
        
        # Initialize the pipeline
        pipeline = Pipeline(config)
        
        # Create a simple text document
        document = Document(
            source="test_document",
            content="This is a test document to verify that the pipeline is working correctly.",
            modality=ContentModality.TEXT
        )
        
        # Process the document
        processed_document = await pipeline.process_document(document)
        
        # Print the processed document
        logger.info(f"Processed document: {processed_document}")
        logger.info(f"Document content: {processed_document.content}")
        logger.info(f"Document info: {processed_document.doc_info}")
        
        return processed_document
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    asyncio.run(test_pipeline()) 