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

# Import constants from setup script
from setup_rag import COLLECTIONS, EMBEDDING_MODEL, USE_OLLAMA_EMBEDDINGS, OLLAMA_EMBEDDING_MODEL, QDRANT_PATH, EMBEDDING_DIMENSION

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
    """Get the Qdrant client"""
    try:
        # Ensure directory exists
        os.makedirs(QDRANT_PATH, exist_ok=True)
        return QdrantClient(path=QDRANT_PATH)
    except Exception as e:
        console.print(f"[bold red]Error connecting to Qdrant: {e}[/bold red]")
        sys.exit(1)

def search_qdrant(query: str, collection_name: str, form: str = None, k: int = 5) -> List[Document]:
    """Search Qdrant directly using the client"""
    embeddings = get_embedding_model()
    client = get_qdrant_client()
    
    try:
        # Embed the query
        query_vector = embeddings.embed_query(query)
        
        # Perform a broader search without filters first
        search_params = {
            "collection_name": collection_name,
            "query_vector": query_vector,
            "limit": k * 2,  # Increase limit to account for post-filtering
            "with_payload": True  # Ensure we get the full payload
        }
        
        # Search without filter
        search_result = client.search(**search_params)
        
        # Convert to Document objects and apply manual filtering
        documents = []
        for result in search_result:
            metadata = result.payload
            
            # Skip documents that don't match the form if form is specified
            if form and metadata.get("form", "") != form:
                continue
                
            # Construct the page content including metadata and actual content
            content_parts = []
            
            # Add all basic metadata fields
            content_parts.append(f"Form: {metadata.get('form', '')}")
            content_parts.append("")
            content_parts.append(f"Subject: {metadata.get('subject', 'Mathematics')}")
            content_parts.append("")
            content_parts.append(f"Topic: {metadata.get('topic', '')}")
            content_parts.append("")
            content_parts.append(f"Subtopic: {metadata.get('subtopic', '')}")
            content_parts.append("")
            if "type" in metadata:
                content_parts.append(f"Type: {metadata.get('type', '')}")
            
            # Create Document object with metadata
            doc = Document(
                page_content="\n".join(content_parts),
                metadata=metadata
            )
            documents.append(doc)
            
            # Only return up to k documents
            if len(documents) >= k:
                break
        
        return documents
    except Exception as e:
        console.print(f"[bold red]Error searching Qdrant: {e}[/bold red]")
        console.print(f"[bold yellow]Error details: {str(e)}[/bold yellow]")
        return []

def extract_topic_form_from_query(query: str, user_form: str) -> tuple:
    """Extract topic, form information, and query intent from the question"""
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
        # Use the prompt with LLM directly
        response = llm.invoke(prompt_template.format(query=query, user_form=user_form))
        
        # Try to find JSON in the response
        response_text = response.strip()
        
        # Try to extract JSON using regex if needed
        import re
        json_pattern = r'({.*})'
        match = re.search(json_pattern, response_text, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            json_str = response_text
            
        # Parse the JSON
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
        console.print(f"[bold yellow]Error parsing LLM response: {e}[/bold yellow]")
        console.print(f"[bold yellow]Raw response: {response if 'response' in locals() else 'No response'}[/bold yellow]")
        console.print("[bold yellow]Using default values[/bold yellow]")
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
    """Display the full content of a document"""
    console.print(f"\n[bold]{title} #{doc_index + 1}[/bold]")
    
    # Extract important fields for display
    metadata = doc.metadata
    
    # Build content sections
    header_section = [
        f"[bold cyan]Form:[/bold cyan] {metadata.get('form', '')}",
        f"[bold green]Subject:[/bold green] {metadata.get('subject', 'Mathematics')}",
        f"[bold green]Topic:[/bold green] {metadata.get('topic', '')}",
        f"[bold yellow]Subtopic:[/bold yellow] {metadata.get('subtopic', '')}",
        f"[bold magenta]Type:[/bold magenta] {metadata.get('type', metadata.get('document_type', ''))}"
    ]
    
    # Check for specific content fields
    content_section = []
    
    # Add objectives if present
    if 'objectives' in metadata:
        content_section.append(f"\n[bold yellow]Objectives:[/bold yellow]\n{metadata.get('objectives', '')}")
    
    # Always add content field if present, with special formatting
    if 'content' in metadata:
        content = metadata.get('content', '')
        if content:  # Check if content is not empty
            # Check if content is a dictionary (might contain image data)
            if isinstance(content, dict) and 'description' in content:
                content_section.append(f"\n[bold yellow]Content:[/bold yellow]\n{content['description']}")
                if 'image_path' in content:
                    content_section.append(f"[italic]Image: {content['image_path']}[/italic]")
            else:
                # Use rich formatting for code blocks in content
                formatted_content = str(content)
                content_section.append(f"\n[bold yellow]Content:[/bold yellow]\n{formatted_content}")
    
    # Add suggested activities if present
    if 'suggested_activities_notes' in metadata:
        activities = metadata.get('suggested_activities_notes', '')
        if activities:
            content_section.append(f"\n[bold yellow]Suggested Activities:[/bold yellow]\n{activities}")
    
    # Add suggested resources if present
    if 'suggested_resources' in metadata:
        resources = metadata.get('suggested_resources', '')
        if resources:
            content_section.append(f"\n[bold yellow]Suggested Resources:[/bold yellow]\n{resources}")
    
    # Don't add page_content if it's just a repeat of metadata
    # which we already display in a formatted way
    
    # Combine all sections
    display_content = "\n".join(header_section)
    if content_section:
        display_content += "\n\n" + "\n".join(content_section)
    
    # Display in panel with appropriate width
    console.print(Panel(
        display_content,
        title=f"{title} #{doc_index + 1}",
        border_style="green",
        width=100,
        expand=False
    ))

def display_full_documents(syllabus_docs: List[Document], content_docs: List[Document]):
    """Display the full content of all retrieved documents"""
    console.print("\n[bold]===== SYLLABUS DOCUMENTS =====")
    for i, doc in enumerate(syllabus_docs):
        display_full_document(doc, "Syllabus", i)

    console.print("\n[bold]===== CONTENT DOCUMENTS =====")
    for i, doc in enumerate(content_docs):
        display_full_document(doc, "Content", i)
    
    console.print("\n")

def generate_answer(query: str, syllabus_docs: List[Document], content_docs: List[Document], request_type: str = "SOLVE", is_specific_problem: bool = False) -> str:
    """Generate an answer using the retrieved documents based on request type"""
    llm = get_llm_model()
    
    # Extract relevant contexts from syllabus and content
    syllabus_contexts = []
    for i, doc in enumerate(syllabus_docs):
        syllabus_text = f"SYLLABUS {i+1}:\n"
        syllabus_text += f"Form: {doc.metadata.get('form', '')}\n"
        syllabus_text += f"Topic: {doc.metadata.get('topic', '')}\n"
        syllabus_text += f"Subtopic: {doc.metadata.get('subtopic', '')}\n"
        
        if 'objectives' in doc.metadata:
            syllabus_text += f"Objectives: {doc.metadata.get('objectives', '')}\n"
            
        if 'content' in doc.metadata:
            content = doc.metadata.get('content', '')
            if isinstance(content, dict) and 'description' in content:
                syllabus_text += f"Content: {content['description']}\n"
            else:
                syllabus_text += f"Content: {content}\n"
                
        syllabus_contexts.append(syllabus_text)
    
    content_contexts = []
    for i, doc in enumerate(content_docs):
        content_text = f"CONTENT {i+1}:\n"
        content_text += f"Form: {doc.metadata.get('form', '')}\n"
        content_text += f"Topic: {doc.metadata.get('topic', '')}\n"
        content_text += f"Subtopic: {doc.metadata.get('subtopic', '')}\n"
        content_text += f"Type: {doc.metadata.get('type', '')}\n"
        
        if 'content' in doc.metadata:
            content = doc.metadata.get('content', '')
            if isinstance(content, dict) and 'description' in content:
                content_text += f"Content: {content['description']}\n"
                if 'image_path' in content:
                    content_text += f"[This content includes an image at: {content['image_path']}]\n"
            else:
                content_text += f"Content: {content}\n"
                
        content_contexts.append(content_text)

    # Select appropriate prompt template based on request type
    if request_type == "EXPLAIN":
        prompt_template = """
        You are a knowledgeable math tutor helping a student understand a mathematical concept.
        
        QUESTION: {query}
        
        SYLLABUS INFORMATION:
        {syllabus_contexts}
        
        RELEVANT CONTENT:
        {content_contexts}
        
        Provide a comprehensive explanation that includes:
        1. A clear introduction to the mathematical concept
        2. The fundamental principles and rules
        3. Step-by-step explanations with simple examples
        4. Important formulas and their meanings
        5. Real-world applications
        6. Common misconceptions and how to avoid them
        
        Format your answer with clear sections and bullet points for key concepts.
        Speak naturally and avoid mentioning "syllabus" or "curriculum".
        """
    elif request_type == "SOLVE":
        prompt_template = """
        You are a patient math tutor helping a student solve a specific problem.
        
        PROBLEM: {query}
        
        SYLLABUS INFORMATION:
        {syllabus_contexts}
        
        RELEVANT CONTENT:
        {content_contexts}
        
        Provide a detailed solution that includes:
        1. Analysis of what the problem is asking
        2. Identification of the relevant mathematical concepts needed
        3. Step-by-step solution with clear explanations for each step
        4. Final answer with appropriate units or format
        5. Verification of the answer
        6. Similar problems the student might encounter
        
        Format your solution clearly with numbered steps.
        Explain your thinking process but keep the focus on solving the specific problem.
        Speak naturally and avoid mentioning "syllabus" or "curriculum".
        """
    else:  # PRACTICE
        prompt_template = """
        You are a math teacher providing practice materials for a student.
        
        TOPIC REQUEST: {query}
        
        SYLLABUS INFORMATION:
        {syllabus_contexts}
        
        RELEVANT CONTENT:
        {content_contexts}
        
        Provide a set of practice materials that includes:
        1. A mix of easy, medium, and challenging questions
        2. Step-by-step solutions for each question
        3. Tips for approaching each type of problem
        4. Common pitfalls to avoid
        
        Format your response with clear sections for questions and solutions.
        Include 3-4 practice questions of varying difficulty.
        Speak naturally and avoid mentioning "syllabus" or "curriculum".
        """

    try:
        formatted_prompt = prompt_template.format(
            query=query,
            syllabus_contexts="\n\n".join(syllabus_contexts),
            content_contexts="\n\n".join(content_contexts)
        )
        response = llm.invoke(formatted_prompt)
        return response
    except Exception as e:
        console.print(f"[bold red]Error generating answer: {e}[/bold red]")
        return "I'm sorry, but I wasn't able to generate an answer due to a technical issue. Please try again with a different question."

def display_documents(docs: List[Document], title: str):
    """Display retrieved documents in a readable format"""
    table = Table(title=title)
    table.add_column("Form", style="cyan")
    table.add_column("Topic", style="green")
    table.add_column("Subtopic", style="yellow")
    table.add_column("Type", style="magenta")
    
    for doc in docs:
        table.add_row(
            doc.metadata.get("form", ""),
            doc.metadata.get("topic", ""),
            doc.metadata.get("subtopic", ""),
            doc.metadata.get("type", doc.metadata.get("document_type", ""))
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