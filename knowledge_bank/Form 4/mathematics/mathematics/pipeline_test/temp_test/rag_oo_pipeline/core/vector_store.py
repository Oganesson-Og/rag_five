"""
RAG Pipeline - Qdrant Vector Store Service
---------------------------------------------

This module defines the `QdrantService` class, which manages interactions
with the Qdrant vector database.

Key Features:
- Initializes and manages a singleton Qdrant client instance.
- Provides a `search` method to query Qdrant collections (syllabus, content).
- Handles query embedding using an `EmbeddingModelFactory` instance.
- Supports metadata filtering (form, topic, subtopic) during search.
- Converts Qdrant search results into Langchain `Document` objects.
- Ensures Qdrant path and collections are correctly configured via `config.py`.

Technical Details:
- Uses `qdrant_client` for communication with the Qdrant database.
- Implements a static `_get_qdrant_client` method for singleton client access.
- The `search` method constructs Qdrant filters and performs similarity search.
- Handles potential errors during Qdrant client initialization and search operations.

Dependencies:
- os
- sys
- typing (List, Optional)
- langchain.docstore.document (Document)
- qdrant_client (QdrantClient, Filter, FieldCondition, MatchValue)
- rich.console (Console)
- ..config (dynamic_rag_config)
- .embeddings (EmbeddingModelFactory)

Author: Keith Satuku
Version: 1.0.0
Created: 2025
License: MIT
"""
import os
import sys
from typing import List, Optional

from langchain.docstore.document import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, PointStruct # SearchRequest might not be needed if not used
from rich.console import Console

from .. import config as dynamic_rag_config
from .embeddings import EmbeddingModelFactory # To get the embedding model

console = Console()

class QdrantService:
    """Manages interactions with the Qdrant vector store."""

    _client_instance = None

    def __init__(self, embedding_model_instance=None):
        """
        Initializes the QdrantService.
        Args:
            embedding_model_instance: An initialized embedding model. 
                                      If None, it will be fetched from EmbeddingModelFactory.
        """
        self.client = self._get_qdrant_client()
        if embedding_model_instance:
            self.embedding_model = embedding_model_instance
        else:
            self.embedding_model = EmbeddingModelFactory.get_embedding_model()
        
        if not self.embedding_model:
            console.print("[bold red]Error: Embedding model could not be initialized for QdrantService.[/bold red]")
            sys.exit(1)

    @staticmethod
    def _get_qdrant_client() -> QdrantClient:
        """Initializes and returns the Qdrant client (singleton for the service)."""
        if QdrantService._client_instance is None:
            try:
                # Explicitly use the QDRANT_PATH from the config module that might have been patched
                current_qdrant_path = dynamic_rag_config.QDRANT_PATH

                os.makedirs(current_qdrant_path, exist_ok=True)
                console.print(f"[dim]Initializing Qdrant client at path: {current_qdrant_path}...[/dim]")
                client = QdrantClient(path=current_qdrant_path)
                # Verify collections exist (optional, good for early failure detection)
                for collection_name in dynamic_rag_config.COLLECTIONS.values(): # Also use dynamic_rag_config here
                    try:
                        client.get_collection(collection_name=collection_name)
                    except Exception as e:
                        console.print(f"[bold yellow]Warning: Collection '{collection_name}' not found or Qdrant issue: {e}. Ensure setup script was run.[/bold yellow]")
                        # Depending on strictness, you might sys.exit(1) here
                QdrantService._client_instance = client
                console.print(f"[dim]Qdrant client initialized. Connected to: {current_qdrant_path}[/dim]")
            except Exception as e:
                console.print(f"[bold red]Fatal Error connecting to Qdrant at '{current_qdrant_path}': {e}[/bold red]")
                console.print(f"[bold yellow]Ensure Qdrant database exists (run setup script) and path is correct.[/bold yellow]")
                sys.exit(1)
        return QdrantService._client_instance

    def search(
        self,
        query: str,
        collection_name: str,
        form: Optional[str] = None,
        k: int = 5,
        filter_topic: Optional[str] = None,
        filter_subtopic: Optional[str] = None
    ) -> List[Document]:
        """Search Qdrant directly, with optional metadata filtering."""
        # Ensure collection_name is valid by checking against dynamically loaded COLLECTIONS
        if collection_name not in dynamic_rag_config.COLLECTIONS.values():
            console.print(f"[bold red]Error: Collection '{collection_name}' is not defined in configuration.[/bold red]")
            return []
        
        try:
            query_vector = self.embedding_model.embed_query(query)

            qdrant_filters_list = []
            if form:
                qdrant_filters_list.append(FieldCondition(key="form", match=MatchValue(value=form)))
            if filter_topic:
                qdrant_filters_list.append(FieldCondition(key="topic", match=MatchValue(value=filter_topic)))
            if filter_subtopic:
                qdrant_filters_list.append(FieldCondition(key="subtopic", match=MatchValue(value=filter_subtopic)))
            
            active_filter = Filter(must=qdrant_filters_list) if qdrant_filters_list else None

            search_params = {
                "collection_name": collection_name,
                "query_vector": query_vector,
                "query_filter": active_filter,
                "limit": k * 2, # Retrieve more for deduplication
                "with_payload": True,
                "with_vectors": False
            }
            
            # console.print(f"[dim]Qdrant searching: coll='{collection_name}', k={k}, form='{form}', topic='{filter_topic}', subtopic='{filter_subtopic}'[/dim]")
            search_result_points = self.client.search(**search_params)

            documents = []
            seen_ids = set()
            for hit in search_result_points:
                if hit.id in seen_ids:
                    continue
                seen_ids.add(hit.id)
                
                metadata = hit.payload if hit.payload is not None else {}
                doc = Document(
                    page_content=metadata.get("page_content", "Error: Page content not found in Qdrant payload"),
                    metadata=metadata
                )
                documents.append(doc)
                if len(documents) >= k:
                    break
            
            return documents
        except Exception as e:
            console.print(f"[bold red]Error during Qdrant search in '{collection_name}': {e}[/bold red]")
            # console.print_exception(show_locals=True) # For more detailed debugging
            return []

if __name__ == '__main__':
    # Example Usage (requires config.py and embeddings.py to be importable)
    print("Testing QdrantService...")
    
    # Get embedding model instance first
    # In a real app, this might be passed from an orchestrator
    try:
        emb_model = EmbeddingModelFactory.get_embedding_model()
        qdrant_service = QdrantService(embedding_model_instance=emb_model)

        # Example search (assuming 'math_syllabus' collection exists and has data)
        # You would need to have run your setup script to populate Qdrant first.
        test_query_syllabus = "Trigonometry ratios"
        print(f"\nSearching syllabus for: '{test_query_syllabus}'")
        syllabus_docs = qdrant_service.search(
            query=test_query_syllabus, 
            collection_name=dynamic_rag_config.COLLECTIONS["syllabus"],
            form="Form 3", # Example filter
            k=2
        )
        if syllabus_docs:
            print(f"Found {len(syllabus_docs)} syllabus documents:")
            for i, doc in enumerate(syllabus_docs):
                print(f"  Doc {i+1}: Form='{doc.metadata.get('form')}', Topic='{doc.metadata.get('topic')}', Subtopic='{doc.metadata.get('subtopic')}'")
                print(f"    Snippet: {doc.page_content[:100]}...")
        else:
            print("No syllabus documents found for the test query, or collection is empty/misconfigured.")

        # Test singleton behavior for Qdrant client
        qdrant_service2 = QdrantService(embedding_model_instance=emb_model)
        print(f"\nIs Qdrant client instance the same? {qdrant_service.client is qdrant_service2.client}")

    except Exception as e:
        print(f"Error in QdrantService test: {e}")
        print("Ensure Qdrant is running, collections exist, and embedding models can load.") 