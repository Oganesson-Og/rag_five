"""
Centralized configuration settings for the RAG OO Pipeline application.

This module consolidates all configuration variables, such as model names,
paths to data stores (Qdrant, JSON knowledge base), collection names,
embedding settings, and other application parameters. Centralizing configuration
makes it easier to manage and modify the application's behavior without
searching through multiple files.
"""
import os

# === Constants from setup_rag_combined.py (or their equivalents) ===

# Qdrant Configuration
QDRANT_PATH = os.getenv("QDRANT_PATH", "./qdrant_db_combined") # Default if not set by env
# Example: COLLECTIONS = {"syllabus": "math_syllabus_prod", "content": "math_content_prod"}
COLLECTIONS = {
    "syllabus": "math_syllabus", 
    "content": "math_content_combined"
}
EMBEDDING_DIMENSION = 768 # Example, adjust based on actual model used for setup

# Embedding Model Configuration
# Set to True to use Ollama for embeddings, False for HuggingFace sentence-transformers
USE_OLLAMA_EMBEDDINGS = True 
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text" # Used if USE_OLLAMA_EMBEDDINGS is True
EMBEDDING_MODEL = "all-MiniLM-L6-v2" # Used if USE_OLLAMA_EMBEDDINGS is False (HuggingFace model)

# === Constants originally in temp_file.py ===

# Available Form Levels for ZIMSEC Math
FORM_LEVELS = ["Form 1", "Form 2", "Form 3", "Form 4"]

# LLM Configuration
# Recommended: Start with a smaller model for faster iteration during development
# Example smaller models: "phi3:mini", "gemma:2b", "mistral:7b"
# Example larger model (potentially slower): "qwen3:32b", "llama3:8b"
LLM_MODEL = "phi4:latest" # Current model from user's temp_file.py

# Path to the local knowledge bank JSON file
# Assuming it's in the same directory as the original temp_file.py or the new main.py
# For a cleaner structure, this might be a relative path from the project root
# or an absolute path managed by environment variables.
KNOWLEDGE_BANK_PATH = "./math_knowledge_bank.json" 

# === Other Application Settings ===

# Default number of documents to retrieve
DEFAULT_SYLLABUS_K = 3
DEFAULT_CONTENT_K = 7

# Truncation limits for LLM prompts (character count, very rough estimate)
MAX_PROMPT_CHARS = 15000

# Ensure QDRANT_PATH directory exists (can be handled by QdrantService on init too)
# os.makedirs(QDRANT_PATH, exist_ok=True) # Moved to be handled by QdrantService initialization


if __name__ == '__main__':
    # Example of how to access and print configs
    print(f"Qdrant Path: {QDRANT_PATH}")
    print(f"Collections: {COLLECTIONS}")
    print(f"Using Ollama Embeddings: {USE_OLLAMA_EMBEDDINGS}")
    print(f"Ollama Embedding Model: {OLLAMA_EMBEDDING_MODEL}")
    print(f"HuggingFace Embedding Model: {EMBEDDING_MODEL}")
    print(f"LLM Model: {LLM_MODEL}")
    print(f"Knowledge Bank Path: {KNOWLEDGE_BANK_PATH}")
    print(f"Default Syllabus K: {DEFAULT_SYLLABUS_K}") 