"""
RAG Pipeline - Main CLI
--------------------------

This module serves as the main command-line interface (CLI) entry point for the
RAG Pipeline application. It provides commands to interact with the RAG
pipeline for querying, classification, and interactive sessions.

Key Features:
- CLI commands: `query`, `classify`, `interactive`.
- Initializes and uses `RAGOrchestrator` for core logic.
- Parses user inputs (question, form level, display options).
- Integrates with Typer for CLI argument parsing.
- Uses Rich for console output.

Technical Details:
- Uses Typer for CLI command definitions.
- Leverages `RAGOrchestrator` from `pipeline.orchestrator`.
- Utilizes `display_utils` for Rich console output formatting.
- Form level validation against `config.FORM_LEVELS`.

Dependencies:
- typer
- rich
- sys
- rag_oo_pipeline.pipeline.orchestrator (RAGOrchestrator)
- rag_oo_pipeline.config (FORM_LEVELS)
- rag_oo_pipeline.ui.display_utils (display_full_document_rich)

To run this application, execute it as a module from the parent directory:
`python -m rag_oo_pipeline.main <command> [OPTIONS]`
Example: `python -m rag_oo_pipeline.main interactive`

Author: Keith Satuku
Version: 1.0.0
Created: 2025
License: MIT
"""
import typer
from rich.console import Console
from rich.panel import Panel
import sys # For sys.exit

# Assuming the rag_oo_pipeline package is in python path or installed
from rag_oo_pipeline.pipeline.orchestrator import RAGOrchestrator
from rag_oo_pipeline.config import FORM_LEVELS # For form validation
from rag_oo_pipeline.ui.display_utils import display_full_document_rich # For classify command display

app = typer.Typer()
console = Console()

# Initialize the orchestrator once when the app starts
# This allows components like embedding models, KB retriever, etc., to be loaded once.
# LLM instance itself is loaded per-query by the orchestrator for now.
rag_orchestrator = RAGOrchestrator()

@app.command()
def query(
    question: str = typer.Option(..., "--question", "-q", help="Your math question for full RAG processing"),
    form: str = typer.Option("Form 4", "--form", "-f", help=f"Your form level ({', '.join(FORM_LEVELS)})"),
    show_sources: bool = typer.Option(False, "--show-sources", "-s", help="Show summary table of retrieved sources"),
    show_full_documents: bool = typer.Option(False, "--show-full-documents", "-d", help="Show the full content of retrieved documents"),
):
    """Answers a math question using the full RAG pipeline via the RAGOrchestrator."""
    if form not in FORM_LEVELS:
        console.print(f"[bold red]Invalid form level: {form}. Available: {', '.join(FORM_LEVELS)}[/bold red]")
        raise typer.Exit(code=1)
    
    rag_orchestrator.process_query(
        question=question,
        user_form=form,
        show_sources=show_sources,
        show_full_documents=show_full_documents
    )

@app.command()
def classify(
    question: str = typer.Option(..., "--question", "-q", help="The math question to classify using the syllabus processor"),
    show_details: bool = typer.Option(False, "--show-details", "-d", help="Show full details of the matched syllabus section")
):
    """Efficiently identifies the syllabus Topic/Subtopic for a given question."""
    console.print(Panel(f"[bold cyan]Classifying Question (OO Version):[/bold cyan] {question}", title="Syllabus Classification", border_style="cyan"))

    # Directly use the syllabus_processor from the orchestrator
    classified_docs = rag_orchestrator.syllabus_processor.classify_question(question, k=1) # k=1 for top match

    if classified_docs:
        top_match_doc = classified_docs[0]
        metadata = top_match_doc.metadata if top_match_doc.metadata else {}
        score = metadata.get('retrieval_score', metadata.get('score', 'N/A'))

        console.print("\n[bold green underline]Classification Result (Top Match):[/bold green underline]")
        table = typer.Table(show_header=False, box=None, padding=(0, 1)) # Using typer.Table might not exist, should be rich.Table
        from rich.table import Table as RichTable # Correcting import for Table
        rich_table_instance = RichTable(show_header=False, box=None, padding=(0,1))
        rich_table_instance.add_column("Field", style="bold blue")
        rich_table_instance.add_column("Value")
        rich_table_instance.add_row("Topic", metadata.get('topic', "[Not Detected]"))
        rich_table_instance.add_row("Subtopic", metadata.get('subtopic', "[Not Detected]"))
        rich_table_instance.add_row("Form", metadata.get('form', "[Not Detected]"))
        rich_table_instance.add_row("Similarity Score", f"{score:.4f}" if isinstance(score, float) else str(score))
        console.print(rich_table_instance)

        if show_details:
            console.print("\n[bold magenta]--- Matched Syllabus Section Details ---[/bold magenta]")
            display_full_document_rich(top_match_doc, "Syllabus Classification Details", 0)
    else:
        console.print("[bold red]Could not classify the question based on syllabus entries.[/bold red]")

@app.command()
def interactive():
    """Starts an interactive session using the RAGOrchestrator."""
    console.print(Panel("[bold green]Math RAG OO Interactive Session[/bold green]\nType 'exit' or 'quit' to end.", border_style="blue"))

    user_form = ""
    while user_form not in FORM_LEVELS:
        user_form = typer.prompt(f"Enter your form level ({'/'.join(FORM_LEVELS)})", default="Form 4")
        if user_form not in FORM_LEVELS:
            console.print(f"[bold red]Invalid form level.[/bold red]")

    show_sources_pref = typer.confirm("Show summary of retrieved sources each time?", default=False)
    show_full_docs_pref = typer.confirm("Show full content of retrieved documents each time?", default=False)

    console.print(f"\n[bold green]Settings:[/bold green] Form Level='{user_form}', Show Sources='{show_sources_pref}', Show Full Docs='{show_full_docs_pref}'")
    console.print("[bold]Enter your math question or type 'exit'/'quit'.[/bold]")

    while True:
        question_input = typer.prompt("\nYour Question", default="")
        if not question_input.strip(): # Handle empty input
            continue
        if question_input.lower() in ["exit", "quit"]:
            console.print("[bold blue]Exiting interactive session.[/bold blue]")
            break
        
        rag_orchestrator.process_query(
            question=question_input,
            user_form=user_form,
            show_sources=show_sources_pref,
            show_full_documents=show_full_docs_pref
        )

if __name__ == "__main__":
    # This allows running the Typer app directly
    # Ensure that KNOWLEDGE_BANK_PATH and QDRANT_PATH in config.py are accessible from here.
    # Typically, you'd run this from the directory containing the rag_oo_pipeline package.
    # e.g., if your project root has rag_oo_pipeline/ and temp_file.py,
    # you might run `python -m rag_oo_pipeline.main interactive` or `python rag_oo_pipeline/main.py interactive`
    # depending on your PYTHONPATH setup.
    
    # For direct execution like `python rag_oo_pipeline/main.py` to work with relative imports like `from ..config`,
    # you often need to ensure the parent directory of `rag_oo_pipeline` is in sys.path or run as a module.
    # One common way if running `python main.py` from within `rag_oo_pipeline` directory itself:
    # import os, sys
    # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    # This line above can be tricky. Simpler is to run as module from parent dir: python -m rag_oo_pipeline.main
    app() 