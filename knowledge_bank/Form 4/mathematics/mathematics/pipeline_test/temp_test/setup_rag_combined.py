#!/usr/bin/env python3
import json
import os
import uuid
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
    "content": "math_content_combined" # Changed collection name for clarity
}

# Local models setup
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Using a sentence-transformer model instead
# Alternative: Use Ollama for embeddings
USE_OLLAMA_EMBEDDINGS = True  # Set to False to use sentence-transformers instead
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_DIMENSION = 768  # nomic-embed-text dimension is 768, not 384

# Qdrant settings
QDRANT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qdrant_db_combined") # Changed DB path

class MathDocument(BaseModel):
    """Schema for math documents"""
    form: str
    subject: str
    topic: str
    subtopic: str
    type: Optional[str] = Field(default=None)  # e.g., 'notes', 'worked_example', 'question', 'syllabus_objective'
    content: Any  # Can be string or dictionary with image data
    heading: Optional[str] = Field(default=None) # Specific heading for notes/sections
    problem: Optional[str] = Field(default=None) # Specific for worked examples/questions
    steps: Optional[List[str]] = Field(default=None) # Specific for worked examples
    answer: Optional[str] = Field(default=None) # Specific for worked examples/questions
    question_text: Optional[str] = Field(default=None) # Specific for questions
    image_description: Optional[str] = Field(default=None)
    image_path: Optional[str] = Field(default=None)

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
        # Check if data is a list, if not, wrap it in a list
        if isinstance(data, dict):
             # Assuming the structure is { "form": ..., "subject": ..., "topics": [...] }
             # or just a list of topics directly. Need to handle based on actual structure.
             # For temp_file_upload_processed.json, it seems to be a list of topic/subtopic blocks.
             pass # If it's already a list, do nothing. Let's assume it's a list for now.
        elif not isinstance(data, list):
             console.print(f"[bold yellow]Warning: Loaded data from {file_path} is not a list. Attempting to process anyway.[/bold yellow]")
             # Handle cases where it might be a single dictionary object if needed
             # data = [data] # Example: wrap it in a list if it's a single dict representing the whole content

        return data if isinstance(data, list) else [] # Ensure we return a list

    except Exception as e:
        console.print(f"[bold red]Error loading {file_path}: {e}[/bold red]")
        return []

def prepare_syllabus_documents(data: List[Dict[str, Any]]) -> List[Document]:
    """Convert syllabus JSON data to LangChain documents (Unchanged from original)"""
    documents = []
    for item in data:
        metadata = {
            "form": item.get("form", ""),
            "subject": item.get("subject", ""),
            "topic": item.get("topic", ""),
            "subtopic": item.get("subtopic", ""),
            "document_type": "syllabus", # Keep distinguishing syllabus
            "objectives": item.get("objectives", ""),
            "content": item.get("content", ""), # Syllabus content field
            "suggested_activities_notes": item.get("suggested_activities_notes", ""),
            "suggested_resources": item.get("suggested_resources", "")
        }
        content_parts = [
            f"Form: {metadata['form']}",
            f"Subject: {metadata['subject']}",
            f"Topic: {metadata['topic']}",
            f"Subtopic: {metadata['subtopic']}",
            f"Objectives: {metadata['objectives']}",
            f"Content: {metadata['content']}",
            f"Suggested Activities: {metadata['suggested_activities_notes']}"
        ]
        doc = Document(page_content="\n\n".join(filter(None, content_parts)), metadata=metadata)
        documents.append(doc)
    return documents

def prepare_content_documents_combined(data: List[Dict[str, Any]]) -> List[Document]:
    """Convert combined content JSON data to LangChain documents"""
    documents = []
    # The input `data` is a list of dictionaries, each representing a topic/subtopic block.
    for item in data:
        base_metadata = {
            "form": item.get("form", ""),
            "subject": item.get("subject", ""),
            "topic": item.get("topic", ""),
            "subtopic": item.get("subtopic", ""),
            "document_type": "content" # Mark as general content
        }

        # --- Extract Notes Title --- 
        notes_data = item.get("notes", {}) # Expect a dictionary
        notes_title = None
        if isinstance(notes_data, dict):
            notes_title = notes_data.get("title")
        # Add notes_title to base metadata if found
        if notes_title:
            base_metadata["notes_title"] = notes_title
        # --- End Extract Notes Title ---

        # Process Notes Sections
        # notes_data is already fetched above
        if isinstance(notes_data, dict):
            sections = notes_data.get("sections", [])
            if isinstance(sections, list):
                for section in sections: # Iterate through sections
                    metadata = base_metadata.copy() # Inherits notes_title if present
                    metadata["type"] = "notes"
                    metadata["heading"] = section.get("heading", "")
                    note_content = section.get("content", "")
                    metadata["content"] = note_content
                    metadata["image_description"] = section.get("image_description")
                    metadata["image_path"] = section.get("image_path")
                    metadata["has_image"] = bool(metadata["image_path"])

                    content_parts = [
                        f"Form: {metadata['form']}",
                        f"Subject: {metadata['subject']}",
                        f"Topic: {metadata['topic']}",
                        f"Subtopic: {metadata['subtopic']}",
                        f"Type: Notes",
                        # Include overall notes title if available
                        f"Section Topic: {notes_title}" if notes_title else None,
                        f"Heading: {metadata['heading']}",
                        f"Content: {note_content}"
                    ]
                    if metadata["has_image"]:
                         content_parts.append(f"Image Description: {metadata['image_description']}")

                    doc = Document(
                        page_content="\n\n".join(filter(None, content_parts)),
                        metadata=metadata
                    )
                    documents.append(doc)
            else:
                if "sections" in notes_data:
                     console.print(f"[yellow]Warning: 'notes[\"sections\"]' field is not a list in item for topic '{base_metadata['topic']}'[/yellow]")

        elif notes_data is not None:
            console.print(f"[yellow]Warning: 'notes' field is not a dictionary in item for topic '{base_metadata['topic']}'[/yellow]")

        # Process Worked Examples
        worked_examples = item.get("worked_examples", [])
        if isinstance(worked_examples, list):
            for example in worked_examples:
                metadata = base_metadata.copy() # Inherits notes_title if present
                metadata["type"] = "worked_example"
                metadata["problem"] = example.get("problem", "")
                metadata["steps"] = example.get("steps", [])
                metadata["answer"] = example.get("answer", "")

                content_parts = [
                    f"Form: {metadata['form']}",
                    f"Subject: {metadata['subject']}",
                    f"Topic: {metadata['topic']}",
                    f"Subtopic: {metadata['subtopic']}",
                    f"Type: Worked Example",
                    # Include overall notes title if available
                    f"Section Topic: {notes_title}" if notes_title else None,
                    f"Problem: {metadata['problem']}",
                    f"Steps: \n- " + "\n- ".join(metadata["steps"]) if metadata["steps"] else "",
                    f"Answer: {metadata['answer']}"
                ]
                doc = Document(
                    page_content="\n\n".join(filter(None, content_parts)),
                    metadata=metadata
                )
                documents.append(doc)
        else:
            console.print(f"[yellow]Warning: 'worked_examples' field is not a list in item for topic '{base_metadata['topic']}'[/yellow]")

        # Process Questions
        questions = item.get("questions", [])
        if isinstance(questions, list):
             for question in questions:
                metadata = base_metadata.copy() # Inherits notes_title if present
                metadata["type"] = "question"
                metadata["question_text"] = question.get("question_text", "")
                metadata["answer"] = question.get("answer", "")
                metadata["steps"] = question.get("steps", [])
                metadata["image_description"] = question.get("image_description")
                metadata["image_path"] = question.get("image_path")
                metadata["has_image"] = bool(metadata["image_path"])

                content_parts = [
                    f"Form: {metadata['form']}",
                    f"Subject: {metadata['subject']}",
                    f"Topic: {metadata['topic']}",
                    f"Subtopic: {metadata['subtopic']}",
                    f"Type: Question",
                    # Include overall notes title if available
                    f"Section Topic: {notes_title}" if notes_title else None,
                    f"Question: {metadata['question_text']}"
                ]
                if metadata["steps"]:
                    content_parts.append(f"Steps: \n- " + "\n- ".join(metadata["steps"]))
                if metadata["answer"]:
                    content_parts.append(f"Answer: {metadata['answer']}")
                if metadata["has_image"]:
                    content_parts.append(f"Image Description: {metadata['image_description']}")

                doc = Document(
                    page_content="\n\n".join(filter(None, content_parts)),
                    metadata=metadata
                )
                documents.append(doc)
        else:
            console.print(f"[yellow]Warning: 'questions' field is not a list in item for topic '{base_metadata['topic']}'[/yellow]")

    return documents


def setup_qdrant_collections():
    """Set up Qdrant collections"""
    # Ensure the directory exists
    os.makedirs(QDRANT_PATH, exist_ok=True)

    # Use local file system for Qdrant
    client = QdrantClient(path=QDRANT_PATH)

    # Create collection for syllabus (if needed, keep separate)
    try:
        client.get_collection(COLLECTIONS["syllabus"])
        console.print(f"[bold yellow]Collection {COLLECTIONS['syllabus']} already exists[/bold yellow]")
    except Exception as e:
        console.print(f"[bold]Creating new collection: {COLLECTIONS['syllabus']}[/bold]")
        client.create_collection(
            collection_name=COLLECTIONS["syllabus"],
            vectors_config=rest.VectorParams(size=EMBEDDING_DIMENSION, distance=rest.Distance.COSINE)
        )
        console.print(f"[bold green]Created collection {COLLECTIONS['syllabus']}[/bold green]")

    # Create collection for combined content
    try:
        client.get_collection(COLLECTIONS["content"])
        console.print(f"[bold yellow]Collection {COLLECTIONS['content']} already exists[/bold yellow]")
    except Exception as e:
        console.print(f"[bold]Creating new collection: {COLLECTIONS['content']}[/bold]")
        client.create_collection(
            collection_name=COLLECTIONS["content"],
            vectors_config=rest.VectorParams(size=EMBEDDING_DIMENSION, distance=rest.Distance.COSINE)
        )
        console.print(f"[bold green]Created collection {COLLECTIONS['content']}[/bold green]")

    return client

def embed_documents_in_batches(documents, embeddings, collection_name, qdrant_client, batch_size=100):
    """Embed documents in smaller batches to avoid memory issues"""
    total_docs = len(documents)
    if total_docs == 0:
        console.print(f"[yellow]No documents to embed for collection {collection_name}.[/yellow]")
        return
        
    console.print(f"[bold]Embedding {total_docs} documents into '{collection_name}' in batches of {batch_size}...[/bold]")

    for i in range(0, total_docs, batch_size):
        end_idx = min(i + batch_size, total_docs)
        batch = documents[i:end_idx]
        console.print(f"[bold]Processing batch {i//batch_size + 1}/{(total_docs-1)//batch_size + 1} ({i+1}-{end_idx}) for '{collection_name}'[/bold]")

        # Get text from documents
        texts = [doc.page_content for doc in batch]
        metadatas = [doc.metadata for doc in batch]

        # Get embeddings for this batch
        try:
            embedded = embeddings.embed_documents(texts)

            # Create points for Qdrant
            points = []

            for j, (vec, metadata) in enumerate(zip(embedded, metadatas)):
                point_id = str(uuid.uuid4())  # Generate a valid UUID
                
                # Add the UUID to metadata as _id
                point_payload = metadata.copy()
                point_payload["_id"] = point_id  # Store UUID in metadata
                point_payload["page_content"] = texts[j]
                
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=vec,
                        payload=point_payload
                    )
                )
            # Add points to Qdrant
            qdrant_client.upsert(
                collection_name=collection_name,
                points=points,
                wait=True # Ensure operation completes
            )

            console.print(f"[green]Successfully embedded batch {i//batch_size + 1} into '{collection_name}'[/green]")
        except Exception as e:
            console.print(f"[bold red]Error embedding batch {i//batch_size + 1} for '{collection_name}': {str(e)}[/bold red]")
            # Optionally continue to next batch or raise error
            # raise e # Uncomment to stop on error

def main():
    console.print("[bold green]Setting up Math RAG Pipeline (Combined File Version)[/bold green]")

    # Setup Qdrant collections
    console.print("[bold]Setting up vector stores...[/bold]")
    qdrant_client = setup_qdrant_collections()

    # Initialize embedding model
    console.print("[bold]Loading embedding model...[/bold]")
    embeddings = get_embeddings()

    # --- Syllabus Processing (Keep if syllabus is separate) ---
    console.print("[bold]Loading syllabus data...[/bold]")
    syllabus_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mathematics_syllabus_chunked.json")
    syllabus_data = load_json_data(syllabus_path)
    if syllabus_data:
        syllabus_docs = prepare_syllabus_documents(syllabus_data)
        console.print(f"[bold]Loaded {len(syllabus_docs)} syllabus documents[/bold]")
        # Embed syllabus documents
        embed_documents_in_batches(
            documents=syllabus_docs,
            embeddings=embeddings,
            collection_name=COLLECTIONS["syllabus"],
            qdrant_client=qdrant_client,
            batch_size=50 # Smaller batch size might be safer
        )
    else:
        console.print("[yellow]Skipping syllabus processing as data failed to load.[/yellow]")
        syllabus_docs = []


    # --- Combined Content Processing ---
    console.print("[bold]Loading combined content data...[/bold]")
    # Use the new combined file name
    combined_content_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "math_knowledge_bank.json")
    combined_content_data = load_json_data(combined_content_path)

    if combined_content_data:
        content_docs = prepare_content_documents_combined(combined_content_data)
        console.print(f"[bold]Prepared {len(content_docs)} content documents from combined file[/bold]")
        # Embed content documents
        embed_documents_in_batches(
            documents=content_docs,
            embeddings=embeddings,
            collection_name=COLLECTIONS["content"],
            qdrant_client=qdrant_client,
            batch_size=50 # Use a consistent, potentially smaller batch size
        )
    else:
        console.print("[bold red]Failed to load combined content data. Skipping content embedding.[/bold red]")
        content_docs = []


    # Removed the progress bar as batch processing provides progress updates

    console.print("[bold green]Math RAG Pipeline (Combined) setup process finished.[/bold green]")
    # Update the query script name if needed
    console.print("[bold]You may need a corresponding query script adapted for this setup.[/bold]")

if __name__ == "__main__":
    main() 