"""
RAG Pipeline - Orchestrator
------------------------------

This module defines the `RAGOrchestrator` class, which serves as the central
controller for the RAG (Retrieval Augmented Generation) pipeline.

Key Features:
- Initializes and manages instances of core services (LLM, Qdrant, KnowledgeBase).
- Initializes and manages pipeline components (SyllabusProcessor, QueryAnalyzer, AnswerGenerator).
- Orchestrates the query processing flow: syllabus classification, query analysis,
  content retrieval (direct JSON match then Qdrant search), and answer generation.
- Handles display of intermediate and final results using `display_utils`.
- Logs timing for different phases of the pipeline for performance monitoring.

Technical Details:
- The `process_query` method is the main entry point for handling a user's question.
- It dynamically loads the LLM for each query to manage resources (though this can be adapted).
- Content retrieval prioritizes direct matches from the JSON knowledge base before
  falling back to semantic search in the Qdrant vector store.
- It integrates various components to form a cohesive RAG workflow.

Dependencies:
- time, typing
- langchain.docstore.document
- rich.console, rich.panel
- ..config (FORM_LEVELS, DEFAULT_SYLLABUS_K, etc.)
- ..core (EmbeddingModelFactory, LanguageModelService, QdrantService, etc.)
- .syllabus_processor (SyllabusProcessor)
- ..ui.display_utils

Author: Keith Satuku
Version: 1.0.0
Created: 2025
License: MIT
"""
import time
from typing import Optional, Tuple, List
from langchain.docstore.document import Document
from rich.console import Console
from rich.panel import Panel

# Configuration
from ..config import (
    FORM_LEVELS, DEFAULT_SYLLABUS_K, DEFAULT_CONTENT_K, LLM_MODEL
)

# Core Services
from ..core.embeddings import EmbeddingModelFactory
from ..core.llm import LanguageModelService, QueryAnalyzer
from ..core.vector_store import QdrantService
from ..core.knowledge_base import KnowledgeBaseRetriever
from ..core.answer_generation import AnswerGenerator

# Pipeline Components
from .syllabus_processor import SyllabusProcessor

# UI Utilities
from ..ui.display_utils import (
    display_documents_summary_rich,
    display_full_document_rich,
    display_generated_answer_rich
)

console = Console()

class RAGOrchestrator:
    """Manages the RAG pipeline execution and component interactions."""

    def __init__(self):
        console.print("[bold blue]Initializing RAG Orchestrator and its components...[/bold blue]")
        # Initialize services (models are loaded lazily/singleton by their services)
        self.embedding_model = EmbeddingModelFactory.get_embedding_model()
        self.llm_service = LanguageModelService() # Service to get LLM instance
        # self.llm_instance = self.llm_service.get_llm() # Load LLM once if orchestrator lives long
        # Decided to load LLM per process_query call for now to match temp_file.py's timer placement

        self.qdrant_service = QdrantService(embedding_model_instance=self.embedding_model)
        self.kb_retriever = KnowledgeBaseRetriever() # Loads JSON data on init
        
        self.syllabus_processor = SyllabusProcessor(qdrant_service=self.qdrant_service)
        # self.query_analyzer = QueryAnalyzer(llm_instance=self.llm_instance) # LLM instance is per-query
        # self.answer_generator = AnswerGenerator(llm_instance=self.llm_instance) # LLM instance is per-query
        console.print("[bold blue]RAG Orchestrator initialized.[/bold blue]")

    def process_query(
        self, 
        question: str, 
        user_form: str, 
        show_sources: bool = False, 
        show_full_documents: bool = False
    ):
        """Processes a single query through the RAG pipeline."""
        total_query_start_time = time.time()

        if user_form not in FORM_LEVELS:
            console.print(f"[bold red]Invalid form level: {user_form}. Available: {', '.join(FORM_LEVELS)}[/bold red]")
            return

        console.print(Panel(f"[bold green]Question:[/bold green] {question}\n[bold blue]Form Level (User Default):[/bold blue] {user_form}", title="New RAG Query Processing", border_style="blue"))

        # Load LLM for this query processing session
        llm_load_start_time = time.time()
        llm_instance = self.llm_service.get_llm() # Get (or load if first time) LLM
        llm_load_duration = time.time() - llm_load_start_time
        console.print(f"[dim]LLM ({LLM_MODEL}) loading/retrieval took {llm_load_duration:.2f} seconds.[/dim]")

        if not llm_instance:
            console.print("[bold red]Failed to load LLM. Aborting query processing.[/bold red]")
            return
        
        # Instantiate components that depend on the per-query LLM instance
        query_analyzer = QueryAnalyzer(llm_instance=llm_instance)
        answer_generator = AnswerGenerator(llm_instance=llm_instance)

        # --- Phase 1: Syllabus Classification ---
        console.print("\n[Phase 1: Classifying question against syllabus...]")
        syll_class_start_time = time.time()
        classified_syllabus_docs = self.syllabus_processor.classify_question(
            question=question, 
            user_form=user_form, # Filter by user's specified form
            k=DEFAULT_SYLLABUS_K
        )
        syll_class_duration = time.time() - syll_class_start_time
        console.print(f"[dim]Syllabus classification took {syll_class_duration:.2f} seconds.[/dim]")

        classified_form_actual = user_form # Default - Start assuming the user's form is the one we want
        classified_topic_actual = ""
        classified_subtopic_actual = ""
        if classified_syllabus_docs:
            top_syll_meta = classified_syllabus_docs[0].metadata
            classified_form_actual = top_syll_meta.get("form", user_form)
            classified_topic_actual = top_syll_meta.get("topic", "")
            classified_subtopic_actual = top_syll_meta.get("subtopic", "")
            score = top_syll_meta.get('retrieval_score', top_syll_meta.get('score', 'N/A'))
            console.print(f"[bold]Syllabus Classification (Top Match):[/bold] Form='{classified_form_actual}', Topic='{classified_topic_actual}', Subtopic='{classified_subtopic_actual}', Score={score}")
        else:
            console.print("[yellow]No direct syllabus classification found. Will rely more on LLM analysis for topic/form guidance.[/yellow]")

        # --- Phase 2: LLM Query Analysis ---
        console.print("\n[Phase 2: Analyzing question with LLM for rephrasing and intent...]")
        llm_analysis_start_time = time.time()
        llm_topic, llm_subtopic, llm_form, rephrased_query, request_type, is_specific_problem = \
            query_analyzer.analyze_query(question, user_form) # Pass user_form as default to LLM
        llm_analysis_duration = time.time() - llm_analysis_start_time
        console.print(f"[dim]LLM analysis (rephrasing/intent) took {llm_analysis_duration:.2f} seconds.[/dim]")
        console.print(f"[bold]LLM Analysis Results:[/bold] Topic='{llm_topic}', Subtopic='{llm_subtopic}', Form='{llm_form}', Type='{request_type}', Rephrased='{rephrased_query}'")

        # Determine final form for content retrieval
        final_form_for_content = classified_form_actual if classified_syllabus_docs and classified_form_actual else llm_form
        console.print(f"[dim]Using form '{final_form_for_content}' for content retrieval.[/dim]")

        # --- Phase 3: Content Retrieval ---
        console.print(f"\n[Phase 3: Retrieving content documents...]")
        content_retrieval_start_time = time.time()
        content_docs: List[Document] = []

        # Try direct JSON lookup first
        if classified_form_actual and classified_topic_actual and classified_subtopic_actual:
            # console.print(f"[dim]Attempting direct JSON lookup with: F='{classified_form_actual}', T='{classified_topic_actual}', ST='{classified_subtopic_actual}'[/dim]")
            json_lookup_start_time = time.time()
            content_docs = self.kb_retriever.retrieve_direct_match(
                classified_form_actual, classified_topic_actual, classified_subtopic_actual
            )
            json_lookup_duration = time.time() - json_lookup_start_time
            if content_docs:
                console.print(f"[green]Direct JSON lookup successful. Found {len(content_docs)} doc(s). Took {json_lookup_duration:.2f}s.[/green]")
            else:
                console.print(f"[yellow]Direct JSON lookup found no match. Took {json_lookup_duration:.2f}s. Proceeding to Qdrant.[/yellow]")
        
        # Fallback to Qdrant if no direct JSON match or if classification was incomplete
        if not content_docs:
            qdrant_content_search_start_time = time.time()
            content_search_query = rephrased_query
            if classified_topic_actual: # Enhance Qdrant query with classified topic/subtopic if available
                content_search_query = f"Topic: {classified_topic_actual}"
                if classified_subtopic_actual: content_search_query += f" Subtopic: {classified_subtopic_actual}"
                content_search_query += f" Details: {rephrased_query}"
            # console.print(f"[dim]Qdrant content search query: '{content_search_query}'[/dim]")
            
            content_docs = self.qdrant_service.search(
                query=content_search_query,
                collection_name=COLLECTIONS["content"],
                form=final_form_for_content,
                k=DEFAULT_CONTENT_K,
                filter_topic=classified_topic_actual if classified_topic_actual else None,
                filter_subtopic=classified_subtopic_actual if classified_subtopic_actual else None
            )
            qdrant_content_search_duration = time.time() - qdrant_content_search_start_time
            console.print(f"[dim]Qdrant content search took {qdrant_content_search_duration:.2f} seconds. Found {len(content_docs)} docs.[/dim]")

        total_content_retrieval_duration = time.time() - content_retrieval_start_time
        console.print(f"[dim]Total content retrieval phase took {total_content_retrieval_duration:.2f} seconds.[/dim]")

        # Display retrieved documents
        if show_sources or show_full_documents:
            console.print("\n[bold underline]Retrieved Syllabus Documents[/]")
            if classified_syllabus_docs:
                if show_full_documents:
                    for i, doc in enumerate(classified_syllabus_docs): display_full_document_rich(doc, "Syllabus", i)
                else:
                    display_documents_summary_rich(classified_syllabus_docs, "Syllabus Sources (Classification)")
            else: console.print("(No syllabus documents from classification)")

            console.print("\n[bold underline]Retrieved Content Documents[/]")
            if content_docs:
                if show_full_documents:
                    for i, doc in enumerate(content_docs): display_full_document_rich(doc, "Content", i)
                else:
                    display_documents_summary_rich(content_docs, "Content Sources")
            else: console.print("(No content documents found)")
        console.print("\n")

        # --- Phase 4: Generate Answer ---
        console.print("\n[Phase 4: Generating answer with LLM...]")
        answer_gen_start_time = time.time()
        if not classified_syllabus_docs and not content_docs:
            console.print("[yellow]Warning: No relevant documents found from syllabus or content. Answer quality may be limited.[/yellow]")
        
        final_answer = answer_generator.generate_llm_answer(
            query=rephrased_query, # Use rephrased query for answer generation
            syllabus_docs=classified_syllabus_docs,
            content_docs=content_docs,
            request_type=request_type,
            is_specific_problem=is_specific_problem
        )
        answer_gen_duration = time.time() - answer_gen_start_time
        console.print(f"[dim]Answer generation took {answer_gen_duration:.2f} seconds.[/dim]")

        display_generated_answer_rich(final_answer, request_type)
        
        total_query_duration = time.time() - total_query_start_time
        console.print(f"[bold magenta]Total query processing time (Orchestrator): {total_query_duration:.2f} seconds.[/bold magenta]")


if __name__ == '__main__':
    # Example Usage for RAGOrchestrator
    # This setup requires all core components and config to be in place.
    # Ensure KNOWLEDGE_BANK_PATH and QDRANT_PATH are correctly set in config.py
    # and that the Qdrant database is populated and Ollama is running.
    
    console.print("[bold]Starting RAG Orchestrator Test...[/bold]")
    orchestrator = RAGOrchestrator()
    
    test_q = "If min=12 and max=98, what is the range?"
    test_f = "Form 4"
    console.print(f"\n--- Processing Test Query: '{test_q}' for {test_f} ---")
    orchestrator.process_query(question=test_q, user_form=test_f, show_sources=True, show_full_documents=False)

    # test_q2 = "Explain trigonometry for Form 3"
    # test_f2 = "Form 3"
    # console.print(f"\n--- Processing Test Query: '{test_q2}' for {test_f2} ---")
    # orchestrator.process_query(question=test_q2, user_form=test_f2, show_sources=True, show_full_documents=True)
    
    # test_q3 = "Give me practice questions on algebra for Form 1"
    # test_f3 = "Form 1"
    # console.print(f"\n--- Processing Test Query: '{test_q3}' for {test_f3} ---")
    # orchestrator.process_query(question=test_q3, user_form=test_f3, show_sources=True, show_full_documents=False)
    
    console.print("[bold]RAG Orchestrator Test Finished.[/bold]") 