"""
Utility functions for formatting and displaying pipeline output using the Rich library.

This module provides helper functions to present information to the user in a 
readable and visually appealing format in the terminal. Functions include:
- `display_documents_summary_rich`: Shows a summary table of retrieved documents.
- `display_full_document_rich`: Displays the detailed content of a single document
  within a panel, adapting the format based on the document source (direct JSON or Qdrant).
- `display_generated_answer_rich`: Formats and displays the final LLM-generated answer
  using Rich Markdown and Panel components.
These functions help separate the presentation logic from the core pipeline processing.
"""
from typing import List
from langchain.docstore.document import Document
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown # Ensure Markdown is imported

console = Console()

def display_documents_summary_rich(docs: List[Document], title: str):
    """Display retrieved documents summary in a Rich table."""
    if not docs:
        console.print(f"\n[dim](No relevant {title.lower()} found for summary)[/dim]")
        return

    table = Table(title=title, box=None, show_header=True, header_style="bold magenta")
    table.add_column("ID", style="dim", width=12, justify="right")
    table.add_column("Source", style="blue", width=18)
    table.add_column("Form", style="cyan", width=8)
    table.add_column("Topic", style="green", max_width=25, overflow="ellipsis")
    table.add_column("Subtopic", style="yellow", max_width=25, overflow="ellipsis")
    table.add_column("Type/Title", style="magenta", max_width=30, overflow="ellipsis")
    table.add_column("Snippet", style="default", max_width=40, overflow="ellipsis")

    for doc in docs:
        metadata = doc.metadata if doc.metadata else {}
        doc_id_display = str(metadata.get('_id', metadata.get('id', metadata.get("json_item_index", "N/A"))))
        source_display = metadata.get("source", "qdrant_vector_search")
        
        type_or_title = metadata.get("original_title", metadata.get("type", metadata.get("document_type", "")))
        content_preview = doc.page_content.replace('\n', ' ').strip()

        table.add_row(
            doc_id_display[:12],
            source_display,
            metadata.get("form", ""),
            metadata.get("topic", ""),
            metadata.get("subtopic", ""),
            str(type_or_title)[:30],
            content_preview
        )
    console.print(table)

def display_full_document_rich(doc: Document, title_prefix: str, doc_index: int):
    """Display the full content of a document using Rich Panel, adapted for various sources."""
    metadata = doc.metadata if doc.metadata else {}

    panel_title_parts = [title_prefix, f"#{doc_index + 1}"]
    doc_id = metadata.get('_id', metadata.get('id', metadata.get("json_item_index", None)))
    if doc_id is not None:
        panel_title_parts.append(f"(ID: {str(doc_id)[:12]})")
    
    source = metadata.get("source", "Unknown")
    panel_title_parts.append(f"Source: {source}")
    
    panel_title = " ".join(panel_title_parts)
    
    # Prepare content for display
    display_items = []
    if source == "direct_json_match":
        # For direct JSON matches, the page_content is already structured.
        # We can use Markdown to render it nicely if it contains Markdown elements.
        display_items.append(f"[bold cyan]Form:[/] {metadata.get('form', 'N/A')}")
        display_items.append(f"[bold green]Topic:[/] {metadata.get('topic', 'N/A')}")
        display_items.append(f"[bold yellow]Subtopic:[/] {metadata.get('subtopic', 'N/A')}")
        display_items.append(f"[bold magenta]Original Title:[/] {metadata.get('original_title', 'N/A')}")
        display_items.append("\n--- Full Content from JSON Match ---")
        display_items.append(doc.page_content) # This content has headers like "--- NOTES SECTIONS ---"
    else: # For Qdrant results or other general documents
        display_items.append(f"[bold cyan]Form:[/] {metadata.get('form', 'N/A')}")
        display_items.append(f"[bold green]Subject:[/] {metadata.get('subject', 'Mathematics')}")
        display_items.append(f"[bold green]Topic:[/] {metadata.get('topic', 'N/A')}")
        display_items.append(f"[bold yellow]Subtopic:[/] {metadata.get('subtopic', 'N/A')}")
        doc_type_qdrant = metadata.get('type', metadata.get('document_type', 'N/A'))
        display_items.append(f"[bold magenta]Type (from Qdrant):[/] {doc_type_qdrant}")
        if 'notes_title' in metadata and metadata['notes_title']:
            display_items.append(f"[bold blue]Section Topic (from Qdrant):[/] {metadata.get('notes_title')}")
        display_items.append("\n--- Retrieved Chunk Content (page_content) ---")
        display_items.append(doc.page_content)

    # Determine border style based on source or type
    border_color = "blue"
    if source == "direct_json_match":
        border_color = "green"
    elif metadata.get('document_type') == 'syllabus' or title_prefix.lower() == "syllabus":
        border_color = "yellow"
        
    console.print(Panel(
        "\n".join(display_items),
        title=panel_title,
        border_style=border_color,
        width=100, # Adjust as needed
        expand=False
    ))

def display_generated_answer_rich(answer_text: str, request_type: str):
    """Displays the final generated answer using Rich Panel and Markdown."""
    console.print(f"\n[bold green underline]Final Answer ({request_type}):[/bold green underline]")
    console.print(Panel(Markdown(answer_text), title="Generated Answer", border_style="green", expand=False))

if __name__ == '__main__':
    # Example usage for display_utils
    console.print("[bold]Testing Rich Display Utilities...[/bold]")

    # Mock documents
    mock_doc_syllabus = Document(
        page_content="This is syllabus content for Topic X, Subtopic Y.", 
        metadata={'form': 'Form 1', 'topic': 'Topic X', 'subtopic': 'Subtopic Y', 'document_type': 'syllabus', 'id': 'syllabus001'}
    )
    mock_doc_qdrant_content = Document(
        page_content="This is a qdrant content chunk about specific details of Y.", 
        metadata={'form': 'Form 1', 'topic': 'Topic X', 'subtopic': 'Subtopic Y', 'type': 'notes', 'id': 'qdrant002'}
    )
    mock_doc_direct_json = Document(
        page_content='''Title: Form 1: Topic X - Subtopic Y
--- NOTES SECTIONS ---
Note 1 content.
--- WORKED EXAMPLES ---
Example 1 content.
--- PRACTICE QUESTIONS ---
Question 1 content.''', 
        metadata={'form': 'Form 1', 'topic': 'Topic X', 'subtopic': 'Subtopic Y', 'source': 'direct_json_match', 'original_title': 'Form 1: Topic X - Subtopic Y', 'json_item_index': 0}
    )

    # Test summary display
    display_documents_summary_rich([mock_doc_syllabus, mock_doc_qdrant_content, mock_doc_direct_json], "Test Summary")

    # Test full document display
    console.print("\n[bold]--- Full Syllabus Document ---[/bold]")
    display_full_document_rich(mock_doc_syllabus, "Syllabus", 0)
    console.print("\n[bold]--- Full Qdrant Content Document ---[/bold]")
    display_full_document_rich(mock_doc_qdrant_content, "Content", 0)
    console.print("\n[bold]--- Full Direct JSON Matched Document ---[/bold]")
    display_full_document_rich(mock_doc_direct_json, "Content", 0)

    # Test answer display
    mock_answer = "To solve this, first do X, then Y. The answer is Z. \n* Step 1\n* Step 2"
    display_generated_answer_rich(mock_answer, "SOLVE") 