"""
RAG Pipeline - Syllabus Processor
------------------------------------

This module defines the `SyllabusProcessor` class, responsible for classifying
user questions against the mathematics syllabus using vector search.

Key Features:
- Uses an injected `QdrantService` instance to query the syllabus vector collection.
- The `classify_question` method takes a user question and optional form level filter.
- Retrieves relevant syllabus entries (topic, subtopic, objectives) from Qdrant.
- Returns a ranked list of Langchain `Document` objects representing matched syllabus sections.
- Aims to provide an initial, curriculum-aligned context for the user's query.

Technical Details:
- Relies on the `QdrantService` for actual vector search and embedding.
- Syllabus data is assumed to be pre-loaded into a Qdrant collection (e.g., "math_syllabus").
- Configuration for collection names and default retrieval K value is sourced from `config.py`.

Dependencies:
- typing (List, Optional)
- langchain.docstore.document (Document)
- rich.console (Console)
- ..core.vector_store (QdrantService)
- ..config (COLLECTIONS, DEFAULT_SYLLABUS_K)

Author: Keith Satuku
Version: 1.0.0
Created: 2025
License: MIT
"""
from typing import List, Optional
from langchain.docstore.document import Document
from rich.console import Console

from ..core.vector_store import QdrantService # QdrantService for searching
from ..config import COLLECTIONS, DEFAULT_SYLLABUS_K

console = Console()

class SyllabusProcessor:
    """Processes questions against the syllabus using vector search."""

    def __init__(self, qdrant_service: QdrantService):
        """
        Initializes the SyllabusProcessor.
        Args:
            qdrant_service: An instance of QdrantService.
        """
        self.qdrant_service = qdrant_service

    def classify_question(self, question: str, user_form: Optional[str] = None, k: int = DEFAULT_SYLLABUS_K) -> List[Document]:
        """
        Efficiently classifies a question into syllabus topics/subtopics
        within the specified user_form using vector search.
        Adds 'retrieval_score' to metadata of returned documents.
        Args:
            question: The user's math question.
            user_form: The form level to filter syllabus entries by. If None, no filter is applied.
            k: The number of top matching syllabus sections to return.

        Returns:
            A list of Document objects representing the top matching syllabus sections.
            Returns an empty list if no match found.
        """
        console.print(f"[dim]SyllabusProcessor classifying question against syllabus (Form Filter: {user_form or 'None'}).[/dim]")
        
        try:
            search_results = self.qdrant_service.search(
                query=question,
                collection_name=COLLECTIONS["syllabus"],
                form=user_form, # Pass the user_form as a filter
                k=k
                # No topic/subtopic filters here
            )

            # The QdrantService.search already returns List[Document]. 
            # We need to ensure it includes scores if Qdrant client returns them. 
            # The current search_qdrant in temp_file.py adds score to metadata payload.
            # Assuming QdrantService.search does something similar or we adapt.
            
            # Let's refine to add retrieval_score if not already present from a raw search_result_point
            # This part depends on how QdrantService.search is structured regarding scores.
            # For now, assuming it's handled or we might need to adjust QdrantService.search or here.
            # The current QdrantService.search method in vector_store.py doesn't explicitly add the score to metadata 
            # like the original temp_file.py's classify_syllabus_section did. 
            # This might need adjustment in QdrantService or here. For simplicity, let's assume scores are in metadata for now.

            processed_documents = []
            for doc in search_results:
                if doc.metadata is None:
                    doc.metadata = {}
                # If QdrantService populated score directly, this isn't needed.
                # If QdrantService returns raw qdrant_client PointStructs, we'd access hit.score.
                # Given QdrantService.search maps to Document, we assume metadata might have it.
                # This is a placeholder if score handling needs refinement in QdrantService itself.
                if 'score' not in doc.metadata and hasattr(doc, 'score'): # Placeholder if score comes as an attribute
                    doc.metadata['retrieval_score'] = doc.score
                elif 'retrieval_score' not in doc.metadata and 'score' in doc.metadata:
                     doc.metadata['retrieval_score'] = doc.metadata['score']
                
                processed_documents.append(doc)

            # console.print(f"[dim]SyllabusProcessor found {len(processed_documents)} sections.[/dim]")
            return processed_documents
        except Exception as e:
            console.print(f"[bold red]Error during syllabus classification in SyllabusProcessor: {e}[/bold red]")
            return []

if __name__ == '__main__':
    # Example Usage (requires other core components and config)
    # This is a simplified test and assumes Qdrant is populated.
    from ..core.embeddings import EmbeddingModelFactory
    
    print("Testing SyllabusProcessor...")
    try:
        emb_model = EmbeddingModelFactory.get_embedding_model()
        q_service = QdrantService(embedding_model_instance=emb_model)
        syllabus_proc = SyllabusProcessor(qdrant_service=q_service)

        test_question = "What are vectors in Form 3?"
        print(f"Classifying question: '{test_question}'")
        classified_docs = syllabus_proc.classify_question(test_question, k=2)

        if classified_docs:
            print(f"Found {len(classified_docs)} matching syllabus documents:")
            for i, doc in enumerate(classified_docs):
                score = doc.metadata.get('retrieval_score', doc.metadata.get('score', 'N/A'))
                print(f"  Doc {i+1}: Form='{doc.metadata.get('form')}', Topic='{doc.metadata.get('topic')}', Score={score}")
                print(f"    Snippet: {doc.page_content[:100]}...")
        else:
            print("No syllabus documents found for classification.")
            
    except Exception as e:
        print(f"Error in SyllabusProcessor test: {e}")
        print("Ensure Qdrant is running, collections exist, and models can load.") 