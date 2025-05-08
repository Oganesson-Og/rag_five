"""
Manages interactions with the Qdrant vector database.

This module defines the `QdrantService` class, which encapsulates all operations
related to the Qdrant vector store. Responsibilities include:
- Initializing the Qdrant client (connecting to the local database defined in `config.py`).
- Providing a `search` method to query the vector store collections (syllabus, content)
  using a text query and optional metadata filters (form, topic, subtopic).
- Handling the embedding of the query text using the provided embedding model.
- Processing search results and converting them into Langchain `Document` objects.
It relies on an `EmbeddingModelFactory` instance to perform query embeddings.
"""
import os
import sys
from typing import List, Optional

from langchain.docstore.document import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, PointStruct # SearchRequest might not be needed if not used
from rich.console import Console

from ..config import QDRANT_PATH, COLLECTIONS
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
                os.makedirs(QDRANT_PATH, exist_ok=True)
                console.print(f"[dim]Initializing Qdrant client at path: {QDRANT_PATH}...[/dim]")
                client = QdrantClient(path=QDRANT_PATH)
                # Verify collections exist (optional, good for early failure detection)
                for collection_name in COLLECTIONS.values():
                    try:
                        client.get_collection(collection_name=collection_name)
                    except Exception as e:
                        console.print(f"[bold yellow]Warning: Collection '{collection_name}' not found or Qdrant issue: {e}. Ensure setup script was run.[/bold yellow]")
                        # Depending on strictness, you might sys.exit(1) here
                QdrantService._client_instance = client
                console.print(f"[dim]Qdrant client initialized. Connected to: {QDRANT_PATH}[/dim]")
            except Exception as e:
                console.print(f"[bold red]Fatal Error connecting to Qdrant at '{QDRANT_PATH}': {e}[/bold red]")
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
        if collection_name not in COLLECTIONS.values():
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
            collection_name=COLLECTIONS["syllabus"],
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