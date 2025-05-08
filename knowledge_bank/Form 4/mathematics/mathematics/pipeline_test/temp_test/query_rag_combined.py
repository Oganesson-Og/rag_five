#!/usr/bin/env python3
import json
import os
import sys
from typing import List, Dict, Any, Optional, Tuple

import typer
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.syntax import Syntax
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

# Import constants from the COMBINED setup script
from setup_rag_combined import COLLECTIONS, EMBEDDING_MODEL, USE_OLLAMA_EMBEDDINGS, OLLAMA_EMBEDDING_MODEL, QDRANT_PATH, EMBEDDING_DIMENSION

console = Console()
app = typer.Typer()

# Available form levels
FORM_LEVELS = ["Form 1", "Form 2", "Form 3", "Form 4"]

# LLM settings
LLM_MODEL = "qwen3:32b"  # Changed from phi4 to deepseek-r1:32b

def get_embedding_model():
    """Get the embedding model"""
    if USE_OLLAMA_EMBEDDINGS:
        return OllamaEmbeddings(
            model=OLLAMA_EMBEDDING_MODEL
        )
    else:
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            show_progress=False
        )

def get_llm_model():
    """Get the LLM model"""
    try:
        return OllamaLLM(model=LLM_MODEL)
    except Exception as e:
        console.print(f"[bold red]Error loading LLM model: {e}[/bold red]")
        console.print("[bold yellow]Please make sure the Ollama service is running and the model is available.[/bold yellow]")
        sys.exit(1)

def get_qdrant_client():
    """Get the Qdrant client (using the combined path)"""
    try:
        # Ensure directory exists (uses QDRANT_PATH from setup_rag_combined)
        os.makedirs(QDRANT_PATH, exist_ok=True)
        return QdrantClient(path=QDRANT_PATH)
    except Exception as e:
        console.print(f"[bold red]Error connecting to Qdrant: {e}[/bold red]")
        sys.exit(1)

def search_qdrant(query: str, collection_name: str, form: str = None, k: int = 5) -> List[Document]:
    """Search Qdrant directly using the client for the combined setup"""
    embeddings = get_embedding_model()
    client = get_qdrant_client()

    try:
        query_vector = embeddings.embed_query(query)

        search_params = {
            "collection_name": collection_name,
            "query_vector": query_vector,
            "limit": k * 2,  # Increase limit for potential post-filtering
            "with_payload": True
        }

        search_result = client.search(**search_params)

        # Convert results to Document objects and apply manual filtering
        documents = []
        seen_ids = set() # Use Qdrant point ID for deduplication

        for result in search_result:
            metadata = result.payload
            doc_id = result.id # Use the actual point ID from Qdrant

            # Deduplicate based on Qdrant ID
            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)

            # Check form compatibility if specified
            if form and metadata.get("form", "") != form:
                continue

            # Construct Document object, ensuring page_content comes from payload
            doc = Document(
                page_content=metadata.get("page_content", "Error: Page content not found in payload"), # Get it from payload
                metadata=metadata
            )
            documents.append(doc)

            if len(documents) >= k:
                break

        return documents
    except Exception as e:
        console.print(f"[bold red]Error searching Qdrant in {collection_name}: {e}[/bold red]")
        console.print(f"[bold yellow]Error details: {str(e)}[/bold yellow]")
        return []

def extract_topic_form_from_query(query: str, user_form: str) -> tuple:
    """Extract topic, form information, and query intent from the question (Revised JSON parsing)"""
    llm = get_llm_model()
    
    prompt_template = """
    You are a helpful assistant that extracts information from math questions.
    
    USER QUERY: {query}
    
    Based on the query, determine:
    1. The most likely math topic and subtopic from the curriculum
    2. The form level (use default if not mentioned)
    3. The type of request the user is making
    
    Default form level: {user_form}
    
    REQUEST TYPES:
    - EXPLAIN: User wants to understand a mathematical concept
    - SOLVE: User wants help solving a specific problem
    - PRACTICE: User wants sample questions or practice problems
    
    Respond in the following JSON format only:
    {{
        "topic": "The main topic (e.g., REAL NUMBERS, ALGEBRA, GEOMETRY)",
        "subtopic": "The subtopic if mentioned (e.g., Number Concepts, Equations, etc.)",
        "form": "The form level (e.g., Form 1, Form 2, etc.)",
        "query_rephrased": "A clearer version of the query focused on the math problem",
        "request_type": "EXPLAIN or SOLVE or PRACTICE",
        "is_specific_problem": true/false
    }}
    
    IMPORTANT: Your output MUST be strictly valid JSON, nothing else. No explanations or text outside the JSON.
    """
    
    try:
        response = llm.invoke(prompt_template.format(query=query, user_form=user_form))
        response_text = response.strip()
        
        # Improved JSON extraction: find first '{' and last '}'
        start_index = response_text.find('{')
        end_index = response_text.rfind('}')
        
        if start_index != -1 and end_index != -1 and end_index > start_index:
            json_str = response_text[start_index : end_index + 1]
        else:
            # Fallback: assume the whole response might be the JSON if no braces found
            # This is less reliable
            json_str = response_text 
            # Basic check if it looks like JSON content might be missing braces
            if not (json_str.startswith('{') and json_str.endswith('}')):
                 # Log a warning if we have to guess braces might be missing
                 # console.print("[yellow]Warning: LLM response for topic extraction might be malformed JSON.[/yellow]")
                 pass # Avoid adding braces automatically unless necessary

        result = json.loads(json_str)
        return (
            result.get("topic", ""),
            result.get("subtopic", ""),
            result.get("form", user_form),
            result.get("query_rephrased", query),
            result.get("request_type", "SOLVE"),
            result.get("is_specific_problem", False)
        )
    except Exception as e:
        console.print(f"[bold yellow]Error parsing LLM response for topic extraction: {e}[/bold yellow]")
        raw_response = response if 'response' in locals() else 'No response generated'
        console.print(f"[bold yellow]Raw response:\n---\n{raw_response}\n---[/bold yellow]")
        console.print("[bold yellow]Using default values for topic extraction[/bold yellow]")
        return ("", "", user_form, query, "SOLVE", False)

def retrieve_from_syllabus(topic: str, subtopic: str, form: str, k: int = 3) -> List[Document]:
    """Retrieve relevant syllabus documents"""
    # Build a search query that emphasizes form and topic/subtopic
    search_query = f"Form: {form} Topic: {topic} Subtopic: {subtopic}"
    
    return search_qdrant(
        query=search_query,
        collection_name=COLLECTIONS["syllabus"],
        form=form,
        k=k
    )

def retrieve_from_content(query: str, form: str, k: int = 10) -> List[Document]:
    """Retrieve relevant content documents (notes, examples, etc.)"""
    try:
        return search_qdrant(
            query=query,
            collection_name=COLLECTIONS["content"],
            form=form,
            k=k
        )
    except Exception as e:
        console.print(f"[bold red]Error retrieving content documents: {e}[/bold red]")
        return []

def display_full_document(doc: Document, title: str, doc_index: int):
    """Display the full content of a document, adapted for combined setup"""
    # console.print(f"\n[bold]{title} #{doc_index + 1}[/bold]") # Title is now part of the panel
    metadata = doc.metadata
    
    # Basic info
    header_section = [
        f"[bold cyan]Form:[/bold cyan] {metadata.get('form', '')}",
        f"[bold green]Subject:[/bold green] {metadata.get('subject', 'Mathematics')}",
        f"[bold green]Topic:[/bold green] {metadata.get('topic', '')}",
        f"[bold yellow]Subtopic:[/bold yellow] {metadata.get('subtopic', '')}",
        f"[bold magenta]Type:[/bold magenta] {metadata.get('type', metadata.get('document_type', ''))}"
    ]
    # Add the Notes Title if present (from combined setup)
    if 'notes_title' in metadata and metadata['notes_title']:
        header_section.append(f"[bold blue]Section Topic:[/bold blue] {metadata.get('notes_title')}")
    
    # Display the actual retrieved chunk content
    chunk_content_section = [
        f"\n[bold]--- Retrieved Chunk Content ---[/bold]",
        doc.page_content # Display the actual page_content from the Document object
    ]

    # Detailed metadata breakdown (optional addition to chunk content)
    metadata_breakdown_section = []
    doc_type = metadata.get('type', metadata.get('document_type', ''))
    metadata_breakdown_section.append(f"\n[bold]--- Metadata Details ---[/bold]")

    if doc_type == 'syllabus':
        if 'objectives' in metadata: metadata_breakdown_section.append(f"Objectives: {metadata.get('objectives', '')}")
        if 'content' in metadata: metadata_breakdown_section.append(f"Content Summary: {metadata.get('content', '')}")
        if 'suggested_activities_notes' in metadata: metadata_breakdown_section.append(f"Suggested Activities: {metadata.get('suggested_activities_notes', '')}")
        if 'suggested_resources' in metadata: metadata_breakdown_section.append(f"Suggested Resources: {metadata.get('suggested_resources', '')}")
    elif doc_type == 'notes':
        if 'heading' in metadata: metadata_breakdown_section.append(f"Heading: {metadata.get('heading', '')}")
        # Note: The main content is already shown in the 'Retrieved Chunk Content' section
        if metadata.get('has_image'): metadata_breakdown_section.append(f"Image: {metadata.get('image_path', '')} ({metadata.get('image_description', '')})")
    elif doc_type == 'worked_example':
        if 'problem' in metadata: metadata_breakdown_section.append(f"Problem: {metadata.get('problem', '')}")
        if 'steps' in metadata and metadata['steps']: metadata_breakdown_section.append(f"Steps: {metadata.get('steps', [])}")
        if 'answer' in metadata: metadata_breakdown_section.append(f"Answer: {metadata.get('answer', '')}")
    elif doc_type == 'question':
        if 'question_text' in metadata: metadata_breakdown_section.append(f"Question: {metadata.get('question_text', '')}")
        if 'steps' in metadata and metadata['steps']: metadata_breakdown_section.append(f"Steps: {metadata.get('steps', [])}")
        if 'answer' in metadata: metadata_breakdown_section.append(f"Answer: {metadata.get('answer', '')}")
        if metadata.get('has_image'): metadata_breakdown_section.append(f"Image: {metadata.get('image_path', '')} ({metadata.get('image_description', '')})")
    else:
        # Fallback for unknown types
        metadata_breakdown_section.append(f"Raw Content Field: {metadata.get('content', 'N/A')}")
        
    # Combine display parts
    display_content = "\n".join(header_section) 
    display_content += "\n".join(chunk_content_section) 
    display_content += "\n".join(metadata_breakdown_section)
    
    doc_display_id = metadata.get('_id', doc.page_content.split(': ')[-1] if ": " in doc.page_content else "Unknown ID") 

    console.print(Panel(
        display_content,
        title=f"{title} #{doc_index + 1} (ID: {doc_display_id})", 
        border_style="blue" if doc_type != 'syllabus' else "green",
        width=100,
        expand=False
    ))

def display_full_documents(syllabus_docs: List[Document], content_docs: List[Document]):
    """Display the full content of all retrieved documents"""
    console.print("\n[bold]===== SYLLABUS DOCUMENTS =====")
    if not syllabus_docs:
        console.print("(No relevant syllabus documents found)")
    else:
        for i, doc in enumerate(syllabus_docs):
            display_full_document(doc, "Syllabus", i)

    console.print("\n[bold]===== CONTENT DOCUMENTS =====")
    if not content_docs:
        console.print("(No relevant content documents found)")
    else:
        for i, doc in enumerate(content_docs):
            display_full_document(doc, "Content", i)
    
    console.print("\n")

def generate_answer(query: str, syllabus_docs: List[Document], content_docs: List[Document], request_type: str = "SOLVE", is_specific_problem: bool = False) -> str:
    """Generate an answer using the retrieved documents (adapted for combined context)"""
    llm = get_llm_model()
    
    # Extract relevant contexts using the actual page_content
    syllabus_contexts = []
    for i, doc in enumerate(syllabus_docs):
        metadata = doc.metadata
        # Create context string using page_content and key metadata
        context_header = f"SYLLABUS CONTEXT {i+1} (Form: {metadata.get('form', 'N/A')}, Topic: {metadata.get('topic', 'N/A')}, Subtopic: {metadata.get('subtopic', 'N/A')})"
        syllabus_text = f"{context_header}\n---\n{doc.page_content}\n---" 
        syllabus_contexts.append(syllabus_text.strip())
    
    content_contexts = []
    for i, doc in enumerate(content_docs):
        metadata = doc.metadata
        doc_type = metadata.get('type', 'Generic Content')
        # Create context string using page_content and key metadata
        context_header = f"DOCUMENT CONTEXT {i+1} (Type: {doc_type}, Form: {metadata.get('form', 'N/A')}, Topic: {metadata.get('topic', 'N/A')}, Subtopic: {metadata.get('subtopic', 'N/A')})"
        if 'notes_title' in metadata and metadata['notes_title']:
             context_header += f" (Section Topic: {metadata.get('notes_title')})"
        content_text = f"{context_header}\n---\n{doc.page_content}\n---"
        content_contexts.append(content_text.strip())

    # Select appropriate prompt template (prompts expect context)
    if request_type == "EXPLAIN":
        prompt_template = """
        You are a knowledgeable math tutor helping a student understand a mathematical concept.
        
        QUESTION: {query}
        
        AVAILABLE CONTEXT (Syllabus and Documents):
        {syllabus_contexts}
        
        {content_contexts}
        
        Based *only* on the AVAILABLE CONTEXT provided, provide a comprehensive explanation that includes:
        1. A clear introduction to the mathematical concept based on the context.
        2. The fundamental principles and rules mentioned in the context.
        3. Step-by-step explanations or examples IF found in the context.
        4. Important formulas mentioned and their meanings.
        
        Format your answer clearly using Markdown.
        If the context does not contain enough information, state that clearly. Do not invent information.
        Avoid mentioning "context", "syllabus" or "documents" in the final explanation.
        """
    elif request_type == "SOLVE":
        prompt_template = """
        You are a patient math tutor helping a student solve a specific problem, using only the provided context.
        
        PROBLEM: {query}
        
        AVAILABLE CONTEXT (Syllabus and Documents):
        {syllabus_contexts}
        
        {content_contexts}
        
        Based *only* on the AVAILABLE CONTEXT provided, provide a detailed solution that includes:
        1. Analysis of what the problem is asking.
        2. Identification of relevant mathematical concepts IF mentioned in the context.
        3. Step-by-step solution using methods or examples found in the context. Reference the context if helpful.
        4. Final answer.
        
        Format your solution clearly using Markdown.
        If the context lacks the necessary information or methods, clearly state that and explain what's missing. Do not invent information.
        Avoid mentioning "context", "syllabus" or "documents" in the final solution.
        """
    else:  # PRACTICE
        prompt_template = """
        You are a math teacher providing practice materials based only on the provided context.
        
        TOPIC REQUEST: {query}
        
        AVAILABLE CONTEXT (Syllabus and Documents):
        {syllabus_contexts}
        
        {content_contexts}
        
        Based *only* on the AVAILABLE CONTEXT provided, provide practice materials. This might include:
        1. Extracting existing questions or examples (type 'question' or 'worked_example') from the context.
        2. Summarizing key concepts from the context relevant to practice.
        
        Format your response clearly using Markdown.
        If the context contains practice questions, present them clearly, including answers/steps if available.
        If the context does not contain suitable practice questions, state that clearly. Do not invent questions.
        Avoid mentioning "context", "syllabus" or "documents" directly.
        """

    try:
        # Combine contexts cleanly
        full_context_syllabus = "\n\n".join(syllabus_contexts) if syllabus_contexts else "No relevant syllabus context found."
        full_context_content = "\n\n".join(content_contexts) if content_contexts else "No relevant document context found."
        
        combined_context = f"{full_context_syllabus}\n\n{full_context_content}"

        formatted_prompt = prompt_template.format(
            query=query,
            # Pass combined context to the placeholders used in the templates
            syllabus_contexts=full_context_syllabus, # Keep for template structure if needed 
            content_contexts=full_context_content   # Keep for template structure if needed
            # Alternatively, if templates are simplified to use one {{context}} variable:
            # context=combined_context 
        )
        
        response = llm.invoke(formatted_prompt)
        return response
    except Exception as e:
        console.print(f"[bold red]Error generating answer: {e}[/bold red]")
        return "I'm sorry, but I wasn't able to generate an answer due to a technical issue. Please try again with a different question."


def display_documents(docs: List[Document], title: str):
    """Display retrieved documents summary in a table"""
    if not docs:
        console.print(f"\n[yellow]No relevant {title.lower()} found.[/yellow]")
        return
        
    table = Table(title=title)
    table.add_column("ID", style="dim", width=10)  # Make ID column smaller
    table.add_column("Form", style="cyan", width=8)
    table.add_column("Topic", style="green", max_width=25)
    table.add_column("Subtopic", style="yellow", max_width=25)
    table.add_column("Type", style="magenta", width=15)
    table.add_column("Content", style="blue", max_width=40)  # Changed from Notes Title to Content
    
    for doc in docs:
        metadata = doc.metadata
        doc_display_id = metadata.get('_id', "Unknown")
        
        # Get a condensed version of content for display
        content_preview = doc.page_content.split('\n')[0][:40] + "..." if doc.page_content else "N/A"
        
        table.add_row(
            str(doc_display_id),
            metadata.get("form", ""),
            metadata.get("topic", ""),
            metadata.get("subtopic", ""),
            metadata.get("type", metadata.get("document_type", "")),
            content_preview  # Show content preview instead of notes_title
        )
    
    console.print(table)
@app.command()
def query(
    question: str = typer.Option(..., "--question", "-q", help="Your math question"),
    form: str = typer.Option("Form 4", "--form", "-f", help="Your form level (Form 1-4)"),
    show_sources: bool = typer.Option(False, "--show-sources", "-s", help="Show the retrieved sources"),
    show_full_documents: bool = typer.Option(False, "--show-full-documents", "-d", help="Show the full content of retrieved documents"),
):
    """Query the Math RAG pipeline with your question"""
    if form not in FORM_LEVELS:
        console.print(f"[bold red]Invalid form level: {form}[/bold red]")
        console.print(f"[bold yellow]Available form levels: {', '.join(FORM_LEVELS)}[/bold yellow]")
        sys.exit(1)
    
    console.print(Panel(f"[bold green]Question:[/bold green] {question}\n[bold blue]Form Level:[/bold blue] {form}"))
    
    # Extract topic and form information from the query
    console.print("[bold]Analyzing your question...[/bold]")
    topic, subtopic, extracted_form, rephrased_query, request_type, is_specific_problem = extract_topic_form_from_query(question, form)
    
    if topic:
        console.print(f"[bold]Detected Topic:[/bold] {topic}")
    if subtopic:
        console.print(f"[bold]Detected Subtopic:[/bold] {subtopic}")
    if extracted_form != form:
        console.print(f"[bold]Using Form Level:[/bold] {extracted_form}")
    console.print(f"[bold]Request Type:[/bold] {request_type}")
    
    # Retrieve relevant documents
    console.print("[bold]Retrieving relevant syllabus information...[/bold]")
    syllabus_docs = retrieve_from_syllabus(topic, subtopic, extracted_form)
    
    console.print("[bold]Retrieving relevant content...[/bold]")
    content_docs = retrieve_from_content(rephrased_query, extracted_form)
    
    # Show sources if requested
    if show_sources:
        display_documents(syllabus_docs, "Syllabus Sources")
        display_documents(content_docs, "Content Sources")
    
    # Show full document content if requested
    if show_full_documents:
        display_full_documents(syllabus_docs, content_docs)
    
    # Generate answer
    console.print("[bold]Generating answer...[/bold]")
    answer = generate_answer(rephrased_query, syllabus_docs, content_docs, request_type, is_specific_problem)
    
    # Display answer
    console.print(Panel(Markdown(answer), title=f"Answer ({request_type})", border_style="green"))

@app.command()
def interactive():
    """Start an interactive session with the Math RAG pipeline"""
    console.print(Panel("[bold green]Math RAG Interactive Session[/bold green]\nType 'exit' or 'quit' to end the session."))
    
    # Ask for form level once
    form = ""
    while form not in FORM_LEVELS:
        form = typer.prompt("Enter your form level (Form 1-4)", default="Form 4")
        if form not in FORM_LEVELS:
            console.print(f"[bold red]Invalid form level: {form}[/bold red]")
            console.print(f"[bold yellow]Available form levels: {', '.join(FORM_LEVELS)}[/bold yellow]")
    
    # Ask about display preferences
    show_sources = typer.confirm("Would you like to see a summary of retrieved sources?", default=False)
    show_full_documents = typer.confirm("Would you like to see the full content of retrieved documents?", default=False)
    
    console.print(f"[bold green]Using form level:[/bold green] {form}")
    console.print("[bold]You can now ask math questions. Type 'exit' or 'quit' to end.[/bold]")
    
    while True:
        question = typer.prompt("\nYour math question")
        
        if question.lower() in ["exit", "quit"]:
            console.print("[bold green]Goodbye![/bold green]")
            break
        
        # Extract topic and form information from the query
        console.print("[bold]Analyzing your question...[/bold]")
        topic, subtopic, extracted_form, rephrased_query, request_type, is_specific_problem = extract_topic_form_from_query(question, form)
        
        if topic:
            console.print(f"[bold]Detected Topic:[/bold] {topic}")
        if subtopic:
            console.print(f"[bold]Detected Subtopic:[/bold] {subtopic}")
        if extracted_form != form:
            console.print(f"[bold]Using Form Level:[/bold] {extracted_form}")
        console.print(f"[bold]Request Type:[/bold] {request_type}")
        
        # Retrieve relevant documents
        console.print("[bold]Retrieving relevant syllabus information...[/bold]")
        syllabus_docs = retrieve_from_syllabus(topic, subtopic, extracted_form)
        
        console.print("[bold]Retrieving relevant content...[/bold]")
        content_docs = retrieve_from_content(rephrased_query, extracted_form)
        
        # Show sources if requested
        if show_sources:
            display_documents(syllabus_docs, "Syllabus Sources")
            display_documents(content_docs, "Content Sources")
        
        # Show full document content if requested
        if show_full_documents:
            display_full_documents(syllabus_docs, content_docs)
        
        # Generate answer
        console.print("[bold]Generating answer...[/bold]")
        answer = generate_answer(rephrased_query, syllabus_docs, content_docs, request_type, is_specific_problem)
        
        # Display answer
        console.print(Panel(Markdown(answer), title=f"Answer ({request_type})", border_style="green"))

if __name__ == "__main__":
    app() 