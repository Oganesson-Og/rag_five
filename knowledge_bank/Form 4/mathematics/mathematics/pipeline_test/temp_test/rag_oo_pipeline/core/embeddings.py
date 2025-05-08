"""
Manages the loading and provision of text embedding models.

This module defines the `EmbeddingModelFactory` which is responsible for
instantiating the correct embedding model (e.g., from HuggingFace or Ollama)
based on the settings in `config.py`. It uses a singleton pattern to ensure
that the potentially large embedding model is loaded only once during the
application's lifecycle, improving performance and reducing memory usage.
"""
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings
from rich.console import Console

# Assuming config.py is in the parent directory (rag_oo_pipeline)
# and this script is in rag_oo_pipeline/core/
# Adjust import path if structure is different or use absolute imports if project is installed
from ..config import USE_OLLAMA_EMBEDDINGS, OLLAMA_EMBEDDING_MODEL, EMBEDDING_MODEL

console = Console()

class EmbeddingModelFactory:
    """Factory class to create and provide embedding model instances."""

    _embedding_model_instance = None

    @staticmethod
    def get_embedding_model():
        """Get the embedding model instance (singleton pattern)."""
        if EmbeddingModelFactory._embedding_model_instance is None:
            if USE_OLLAMA_EMBEDDINGS:
                console.print(f"[dim]Initializing Ollama embeddings: {OLLAMA_EMBEDDING_MODEL}...[/dim]")
                EmbeddingModelFactory._embedding_model_instance = OllamaEmbeddings(
                    model=OLLAMA_EMBEDDING_MODEL
                )
                console.print(f"[dim]Ollama embeddings ({OLLAMA_EMBEDDING_MODEL}) initialized.[/dim]")
            else:
                console.print(f"[dim]Initializing HuggingFace embeddings: {EMBEDDING_MODEL}...[/dim]")
                EmbeddingModelFactory._embedding_model_instance = HuggingFaceEmbeddings(
                    model_name=EMBEDDING_MODEL,
                    show_progress=False # Usually set to False for cleaner logs in applications
                )
                console.print(f"[dim]HuggingFace embeddings ({EMBEDDING_MODEL}) initialized.[/dim]")
        return EmbeddingModelFactory._embedding_model_instance

if __name__ == '__main__':
    # Example usage:
    print("Attempting to load embedding model...")
    try:
        model = EmbeddingModelFactory.get_embedding_model()
        print(f"Successfully loaded embedding model: {type(model)}")
        
        # Test embedding (optional)
        # query_vector = model.embed_query("This is a test query.")
        # print(f"Test query embedded into vector of dimension: {len(query_vector)}")
        
        # Try getting it again to test singleton behavior
        model2 = EmbeddingModelFactory.get_embedding_model()
        print(f"Is it the same model instance? {model is model2}")

    except Exception as e:
        print(f"Error during embedding model loading test: {e}") 