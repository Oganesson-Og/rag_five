#!/usr/bin/env python3
import json
import os
from typing import List, Dict, Any, Optional

import numpy as np
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
# Use the updated import for OllamaEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain.docstore.document import Document
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

# Constants
COLLECTIONS = {
    "syllabus": "math_syllabus",
    "content": "math_content"
}

# Local models setup
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Using a sentence-transformer model instead
# Alternative: Use Ollama for embeddings
USE_OLLAMA_EMBEDDINGS = True  # Set to False to use sentence-transformers instead
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_DIMENSION = 768  # nomic-embed-text dimension is 768, not 384

# Qdrant settings
QDRANT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qdrant_db")

class MathDocument(BaseModel):
    """Schema for math documents"""
    form: str
    subject: str
    topic: str
    subtopic: str
    type: Optional[str] = Field(default=None)  # Only for content, not in syllabus
    content: Any  # Can be string or dictionary with image data

def get_embeddings():
    """Get the appropriate embedding model"""
    if USE_OLLAMA_EMBEDDINGS:
        console.print(f"[bold]Using Ollama embeddings: {OLLAMA_EMBEDDING_MODEL}[/bold]")
        # OllamaEmbeddings doesn't support show_progress
        return OllamaEmbeddings(
            model=OLLAMA_EMBEDDING_MODEL
        )
    else:
        console.print(f"[bold]Using HuggingFace embeddings: {EMBEDDING_MODEL}[/bold]")
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            show_progress=False
        )

def load_json_data(file_path: str) -> List[Dict[str, Any]]:
    """Load data from a JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        console.print(f"[bold red]Error loading {file_path}: {e}[/bold red]")
        return []

def prepare_syllabus_documents(data: List[Dict[str, Any]]) -> List[Document]:
    """Convert syllabus JSON data to LangChain documents"""
    documents = []
    
    for item in data:
        # Extract metadata with all fields included
        metadata = {
            "form": item.get("form", ""),
            "subject": item.get("subject", ""),
            "topic": item.get("topic", ""),
            "subtopic": item.get("subtopic", ""),
            "document_type": "syllabus",
            "objectives": item.get("objectives", ""),
            "content": item.get("content", ""),
            "suggested_activities_notes": item.get("suggested_activities_notes", ""),
            "suggested_resources": item.get("suggested_resources", "")
        }
        
        # Combine content fields for better retrieval
        content_parts = []
        content_parts.append(f"Form: {item.get('form', '')}")
        content_parts.append(f"Subject: {item.get('subject', '')}")
        content_parts.append(f"Topic: {item.get('topic', '')}")
        content_parts.append(f"Subtopic: {item.get('subtopic', '')}")
        
        if "objectives" in item:
            content_parts.append(f"Objectives: {item.get('objectives', '')}")
        if "content" in item:
            content_parts.append(f"Content: {item.get('content', '')}")
        if "suggested_activities_notes" in item:
            content_parts.append(f"Suggested Activities: {item.get('suggested_activities_notes', '')}")
        
        # Create document
        doc = Document(
            page_content="\n\n".join(content_parts),
            metadata=metadata
        )
        documents.append(doc)
    
    return documents

def prepare_content_documents(data: List[Dict[str, Any]]) -> List[Document]:
    """Convert content JSON data to LangChain documents"""
    documents = []
    
    for item in data:
        # Extract metadata with content included
        metadata = {
            "form": item.get("form", ""),
            "subject": item.get("subject", ""),
            "topic": item.get("topic", ""),
            "subtopic": item.get("subtopic", ""),
            "type": item.get("type", ""),
            "document_type": "content",
            "content": item.get("content", "")  # Store the actual content in metadata
        }
        
        # Combine content fields for better retrieval
        content_parts = []
        content_parts.append(f"Form: {item.get('form', '')}")
        content_parts.append(f"Subject: {item.get('subject', '')}")
        content_parts.append(f"Topic: {item.get('topic', '')}")
        content_parts.append(f"Subtopic: {item.get('subtopic', '')}")
        content_parts.append(f"Type: {item.get('type', '')}")
        
        # Handle content field which could be string or dict for images
        content = item.get("content", "")
        if isinstance(content, dict) and "description" in content:
            content_parts.append(f"Content: {content['description']}")
            metadata["has_image"] = True
            metadata["image_path"] = content.get("image_path", "")
        else:
            content_parts.append(f"Content: {content}")
        
        # Create document
        doc = Document(
            page_content="\n\n".join(content_parts),
            metadata=metadata
        )
        documents.append(doc)
    
    return documents

def setup_qdrant_collections():
    """Set up Qdrant collections"""
    # Ensure the directory exists
    os.makedirs(QDRANT_PATH, exist_ok=True)
    
    # Use local file system for Qdrant to avoid URL encoding issues
    client = QdrantClient(path=QDRANT_PATH)
    
    # Create collection for syllabus
    try:
        client.get_collection(COLLECTIONS["syllabus"])
        console.print(f"[bold yellow]Collection {COLLECTIONS['syllabus']} already exists[/bold yellow]")
    except Exception as e:
        console.print(f"[bold]Creating new collection: {COLLECTIONS['syllabus']}[/bold]")
        client.create_collection(
            collection_name=COLLECTIONS["syllabus"],
            vectors_config=rest.VectorParams(
                size=EMBEDDING_DIMENSION,  # Size for embedding model
                distance=rest.Distance.COSINE
            )
        )
        console.print(f"[bold green]Created collection {COLLECTIONS['syllabus']}[/bold green]")
    
    # Create collection for content (books, notes, examples)
    try:
        client.get_collection(COLLECTIONS["content"])
        console.print(f"[bold yellow]Collection {COLLECTIONS['content']} already exists[/bold yellow]")
    except Exception as e:
        console.print(f"[bold]Creating new collection: {COLLECTIONS['content']}[/bold]")
        client.create_collection(
            collection_name=COLLECTIONS["content"],
            vectors_config=rest.VectorParams(
                size=EMBEDDING_DIMENSION,  # Size for embedding model
                distance=rest.Distance.COSINE
            )
        )
        console.print(f"[bold green]Created collection {COLLECTIONS['content']}[/bold green]")
    
    return client

def embed_documents_in_batches(documents, embeddings, collection_name, qdrant_client, batch_size=100):
    """Embed documents in smaller batches to avoid memory issues"""
    total_docs = len(documents)
    console.print(f"[bold]Embedding {total_docs} documents in batches of {batch_size}...[/bold]")
    
    for i in range(0, total_docs, batch_size):
        end_idx = min(i + batch_size, total_docs)
        batch = documents[i:end_idx]
        console.print(f"[bold]Processing batch {i//batch_size + 1}/{(total_docs-1)//batch_size + 1} ({i}-{end_idx})[/bold]")
        
        # Get text from documents
        texts = [doc.page_content for doc in batch]
        metadatas = [doc.metadata for doc in batch]
        
        # Get embeddings for this batch
        try:
            embedded = embeddings.embed_documents(texts)
            
            # Create points for Qdrant
            points = []
            for j, (vec, metadata) in enumerate(zip(embedded, metadatas)):
                point_id = i + j  # Unique ID across all batches
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=vec,
                        payload=metadata
                    )
                )
            
            # Add points to Qdrant
            qdrant_client.upsert(
                collection_name=collection_name,
                points=points
            )
            
            console.print(f"[green]Successfully embedded batch {i//batch_size + 1}[/green]")
        except Exception as e:
            console.print(f"[bold red]Error embedding batch {i//batch_size + 1}: {str(e)}[/bold red]")
            raise e

def main():
    console.print("[bold green]Setting up Math RAG Pipeline[/bold green]")
    
    # Setup Qdrant collections
    console.print("[bold]Setting up vector stores...[/bold]")
    qdrant_client = setup_qdrant_collections()
    
    # Initialize embedding model
    console.print("[bold]Loading embedding model...[/bold]")
    embeddings = get_embeddings()
    
    # Load syllabus data
    console.print("[bold]Loading syllabus data...[/bold]")
    syllabus_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mathematics_syllabus_chunked.json")
    syllabus_data = load_json_data(syllabus_path)
    syllabus_docs = prepare_syllabus_documents(syllabus_data)
    console.print(f"[bold]Loaded {len(syllabus_docs)} syllabus documents[/bold]")
    
    # Load content data (combine both files)
    console.print("[bold]Loading content data...[/bold]")
    examples_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "worked_examples_chunked.json")
    notes_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "first_maths_notes_chunked_cleaned.json")
    
    examples_data = load_json_data(examples_path)
    notes_data = load_json_data(notes_path)
    
    all_content_data = examples_data + notes_data
    content_docs = prepare_content_documents(all_content_data)
    console.print(f"[bold]Loaded {len(content_docs)} content documents[/bold]")
    
    # Create Qdrant vector stores
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        # Add syllabus documents to vector store
        task = progress.add_task("[green]Embedding syllabus documents...", total=1)
        try:
            # Instead of using from_documents, we'll use our custom batching function
            embed_documents_in_batches(
                documents=syllabus_docs,
                embeddings=embeddings,
                collection_name=COLLECTIONS["syllabus"],
                qdrant_client=qdrant_client,
                batch_size=20  # Use a small batch size for safety
            )
            progress.update(task, completed=1)
        except Exception as e:
            console.print(f"[bold red]Error embedding syllabus documents: {e}[/bold red]")
            return
        
        # Add content documents to vector store
        task = progress.add_task("[green]Embedding content documents...", total=1)
        try:
            # Instead of using from_documents, we'll use our custom batching function
            embed_documents_in_batches(
                documents=content_docs,
                embeddings=embeddings,
                collection_name=COLLECTIONS["content"],
                qdrant_client=qdrant_client,
                batch_size=20  # Use a small batch size for safety
            )
            progress.update(task, completed=1)
        except Exception as e:
            console.print(f"[bold red]Error embedding content documents: {e}[/bold red]")
            return
    
    console.print("[bold green]Math RAG Pipeline setup complete![/bold green]")
    console.print("[bold]You can now use query_rag.py to interact with the system.[/bold]")

if __name__ == "__main__":
    main() 