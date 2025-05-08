#!/usr/bin/env python3
import json
import os
import sys
import time # Added for timing
from typing import List, Dict, Any, Optional, Tuple, Union
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
from qdrant_client.models import Filter, FieldCondition, MatchValue, PointStruct, SearchRequest

# Import constants from the COMBINED setup script
# Ensure setup_rag_combined.py is in the same directory or Python path
try:
    from setup_rag_combined import COLLECTIONS, EMBEDDING_MODEL, USE_OLLAMA_EMBEDDINGS, OLLAMA_EMBEDDING_MODEL, QDRANT_PATH, EMBEDDING_DIMENSION
except ImportError:
    print("Error: Could not import configuration from setup_rag_combined.py.")
    print("Ensure the file exists and is in the correct path.")
    # Define fallbacks or exit
    COLLECTIONS = {"syllabus": "math_syllabus", "content": "math_content_combined"}
    # Add other fallback constants if needed, or exit
    sys.exit(1)


console = Console()
app = typer.Typer()

# Available form levels
FORM_LEVELS = ["Form 1", "Form 2", "Form 3", "Form 4"]

# LLM settings
LLM_MODEL = "phi4:latest" # CHANGED to a smaller model for faster inference

# Path to the knowledge bank JSON file (assuming it's in the same directory)
KNOWLEDGE_BANK_PATH = "./math_knowledge_bank.json"

def get_embedding_model():
    """Get the embedding model"""
    if USE_OLLAMA_EMBEDDINGS:
        console.print(f"[dim]Using Ollama embeddings: {OLLAMA_EMBEDDING_MODEL}[/dim]")
        return OllamaEmbeddings(
            model=OLLAMA_EMBEDDING_MODEL
        )
    else:
        console.print(f"[dim]Using HuggingFace embeddings: {EMBEDDING_MODEL}[/dim]")
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            show_progress=False
        )

def get_llm_model(model_name: str = LLM_MODEL):
    """Loads and returns the LLM model instance."""
    try:
        console.print(f"[dim]Initializing LLM model: {model_name}... This may take a moment.[/dim]")
        llm = OllamaLLM(model=model_name)
        # You could add a small test invocation here if needed to confirm model is working
        # For example: llm.invoke("Hello!") 
        console.print(f"[dim]LLM model {model_name} initialized successfully.[/dim]")
        return llm
    except Exception as e:
        console.print(f"[bold red]Error loading LLM model '{model_name}': {e}[/bold red]")
        console.print("[bold yellow]Please make sure the Ollama service is running and the model ('ollama pull {model_name}') is available.[/bold yellow]")
        sys.exit(1)

def get_qdrant_client():
    """Get the Qdrant client (using the combined path)"""
    try:
        # Ensure directory exists (uses QDRANT_PATH from setup_rag_combined)
        os.makedirs(QDRANT_PATH, exist_ok=True)
        client = QdrantClient(path=QDRANT_PATH)
        # Optional: Verify connection by trying to get collection info
        client.get_collection(collection_name=COLLECTIONS["syllabus"])
        client.get_collection(collection_name=COLLECTIONS["content"])
        console.print(f"[dim]Connected to Qdrant at: {QDRANT_PATH}[/dim]")
        return client
    except Exception as e:
        console.print(f"[bold red]Error connecting to Qdrant at '{QDRANT_PATH}': {e}[/bold red]")
        console.print(f"[bold yellow]Ensure Qdrant database exists (run setup script) and path is correct.[/bold yellow]")
        sys.exit(1)

def _parse_knowledge_bank_title(title: str) -> Optional[Tuple[str, str, str]]:
    """Robustly parses titles like 'Form 4: Statistics - Measures of Central Tendency and Dispersion'"""
    try:
        form_part, rest_of_title = title.split(':', 1)
        form_part = form_part.strip()
        
        topic_subtopic_part = rest_of_title.strip()
        if '-' in topic_subtopic_part:
            topic_part, subtopic_part = topic_subtopic_part.split('-', 1)
            topic_part = topic_part.strip()
            subtopic_part = subtopic_part.strip()
        else:
            # Handles cases where there might not be a subtopic after the topic
            topic_part = topic_subtopic_part.strip()
            subtopic_part = "" # Default to empty if no '-' found
            
        if not form_part or not topic_part: # Basic validation
            # console.print(f"[yellow]Warning: Could not fully parse title: '{title}'. Missing form or topic.[/yellow]")
            return None
            
        return form_part, topic_part, subtopic_part
    except ValueError: # Handles errors from split if format is unexpected
        # console.print(f"[yellow]Warning: Title '{title}' does not match expected format for parsing.[/yellow]")
        return None
    except Exception as e:
        # console.print(f"[yellow]Unexpected error parsing title '{title}': {e}[/yellow]")
        return None

def search_qdrant(
    query: str,
    collection_name: str,
    form: Optional[str] = None,
    k: int = 5,
    filter_topic: Optional[str] = None,
    filter_subtopic: Optional[str] = None
) -> List[Document]:
    """Search Qdrant directly using the client, with optional metadata filtering."""
    embeddings = get_embedding_model()
    client = get_qdrant_client()

    try:
        query_vector = embeddings.embed_query(query)

        # Build Qdrant filter
        qdrant_filters = []
        if form:
            qdrant_filters.append(FieldCondition(key="form", match=MatchValue(value=form)))
        if filter_topic: # Ensure topic is not empty
            qdrant_filters.append(FieldCondition(key="topic", match=MatchValue(value=filter_topic)))
        if filter_subtopic: # Ensure subtopic is not empty
            qdrant_filters.append(FieldCondition(key="subtopic", match=MatchValue(value=filter_subtopic)))
        
        active_filter = None
        if qdrant_filters:
            active_filter = Filter(must=qdrant_filters)
            # console.print(f"[dim]Using Qdrant filter: {active_filter.model_dump_json(indent=2)}[/dim]") # Debugging

        search_params = {
            "collection_name": collection_name,
            "query_vector": query_vector,
            "query_filter": active_filter, # Pass the filter to Qdrant
            "limit": k * 2,  # Retrieve more to allow for deduplication, though filtering might reduce need
            "with_payload": True,
            "with_vectors": False # Usually don't need vectors in result
        }
        
        # console.print(f"[dim]Searching Qdrant with params: coll={collection_name}, k={k}, form='{form}', topic='{filter_topic}', subtopic='{filter_subtopic}'[/dim]")
        search_result = client.search(**search_params)

        # Convert results to Document objects and apply manual filtering
        documents = []
        seen_ids = set() # Use Qdrant point ID for deduplication

        for hit in search_result:
            metadata = hit.payload
            doc_id = hit.id # Use the actual point ID from Qdrant

            # Deduplicate based on Qdrant ID
            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)

            # Form filtering is now handled by Qdrant's query_filter
            # if form and metadata.get("form", "").strip().lower() != form.strip().lower():
            #     # console.print(f"[dim]Skipping doc {doc_id}: form mismatch ('{metadata.get('form')}' != '{form}') [/dim]")
            #     continue

            # Construct Document object, ensuring page_content comes from payload
            doc = Document(
                page_content=metadata.get("page_content", "Error: Page content not found in payload"), # Get it from payload
                metadata=metadata
            )
            documents.append(doc)

            if len(documents) >= k: # Stop once we have k deduplicated, filtered docs
                break
        
        # console.print(f"[dim]Retrieved {len(documents)} documents from Qdrant after filtering/deduplication for query: '{query}'[/dim]")
        return documents
    except Exception as e:
        console.print(f"[bold red]Error searching Qdrant collection '{collection_name}': {e}[/bold red]")
        console.print(f"[bold yellow]Error details: {str(e)}[/bold yellow]")
        return []

# =========================================================================
# == EFFICIENT SYLLABUS CLASSIFICATION FUNCTIONALITY ==
# =========================================================================

def classify_syllabus_section(
    question: str,
    k: int = 1 # Retrieve only the top 1 match by default
) -> List[Document]:
    """
    Efficiently classifies a question into a syllabus topic/subtopic
    using direct vector search on the syllabus collection.

    Args:
        question: The user's math question.
        k: The number of top matching syllabus sections to return.

    Returns:
        A list of Document objects representing the top matching syllabus sections.
        Returns an empty list if no match found. Includes 'retrieval_score' in metadata.
    """
    console.print(f"[cyan]Attempting to classify question against syllabus:[/cyan] '{question}' (k={k})")
    embeddings = get_embedding_model()
    client = get_qdrant_client()
    collection_name = COLLECTIONS["syllabus"] # Target only the syllabus

    try:
        # 1. Embed the question
        query_vector = embeddings.embed_query(question)

        # 2. Search ONLY the syllabus collection
        search_result = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=k,
            with_payload=True, # Need metadata
            with_vectors=False, # Don't need the vector itself
            score_threshold=0.5 # Optional: Add a score threshold if desired
        )

        # 3. Process results
        if not search_result:
            console.print("[yellow]No matching syllabus section found.[/yellow]")
            return [] # Return empty list

        # 4. Convert results to Document objects
        syllabus_documents = []
        for hit in search_result:
            metadata = hit.payload
            # Add score to metadata for potential use later
            metadata['retrieval_score'] = hit.score # Store the score

            # Construct Document object
            doc = Document(
                page_content=metadata.get("page_content", "Error: Page content not found in payload"),
                metadata=metadata
            )
            syllabus_documents.append(doc)
        
        # console.print(f"[dim]Found {len(syllabus_documents)} syllabus sections from classification.[/dim]")
        return syllabus_documents

    except Exception as e:
        console.print(f"[bold red]Error during syllabus classification search: {e}[/bold red]")
        return []

# =========================================================================
# == ORIGINAL QUESTION ANSWERING FUNCTIONALITY ==
# =========================================================================

def extract_topic_form_from_query(query: str, user_form: str, llm: OllamaLLM) -> tuple:
    """
    Uses the provided LLM instance to guess topic, form, and intent from the question.
    """
    # llm = get_llm_model() # LLM is now passed as an argument

    prompt_template = """
    You are a helpful assistant that extracts information from math questions.

    USER QUERY: {query}

    Based ONLY on the USER QUERY, determine:
    1. The most likely SINGLE primary math topic (e.g., Algebra, Geometry, Trigonometry, Financial Mathematics, Graphs, Variation, Statistics, Probability, Transformation, Vectors, Measures and Mensuration). BE SPECIFIC.
    2. The most likely SINGLE primary subtopic IF discernible (e.g., Consumer arithmetic, Functional graphs, Equations, Combined Events). BE SPECIFIC.
    3. The form level mentioned, if any (e.g., Form 1, Form 2, Form 3, Form 4). If not mentioned, use the default.
    4. A slightly rephrased version of the query focusing on the core math task.
    5. The type of request (Choose ONE: EXPLAIN, SOLVE, PRACTICE).
    6. Is the query a specific problem to be solved? (true/false).

    Default form level: {user_form}

    Respond in the following JSON format ONLY. Ensure keys and string values are double-quoted. DO NOT ADD ANY TEXT BEFORE OR AFTER THE JSON OBJECT.
    {{
        "topic": "Primary Topic Name",
        "subtopic": "Primary Subtopic Name or empty string",
        "form": "Form Level",
        "query_rephrased": "Rephrased query",
        "request_type": "EXPLAIN or SOLVE or PRACTICE",
        "is_specific_problem": true or false
    }}
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
            json_str = response_text
            # Avoid automatically adding braces, rely on json.loads failure
            # console.print("[yellow]Warning: LLM response for topic extraction might be malformed JSON (missing braces?).[/yellow]")

        result = json.loads(json_str)
        return (
            result.get("topic", ""),
            result.get("subtopic", ""),
            result.get("form", user_form), # Use default if parsing fails or key missing
            result.get("query_rephrased", query),
            result.get("request_type", "SOLVE"),
            result.get("is_specific_problem", False)
        )
    except json.JSONDecodeError as json_e:
        console.print(f"[bold red]Error parsing JSON from LLM response: {json_e}[/bold red]")
        raw_response = response if 'response' in locals() else 'No response generated'
        console.print(f"[bold yellow]Raw response that failed parsing:\n---\n{raw_response}\n---[/bold yellow]")
        console.print("[bold yellow]Using default values for topic extraction[/bold yellow]")
        return ("", "", user_form, query, "SOLVE", False) # Return defaults on error
    except Exception as e:
        console.print(f"[bold red]Unexpected error during topic extraction: {e}[/bold red]")
        console.print("[bold yellow]Using default values for topic extraction[/bold yellow]")
        return ("", "", user_form, query, "SOLVE", False) # Return defaults on error


def retrieve_from_syllabus(topic: str, subtopic: str, form: str, k: int = 5) -> List[Document]:
    """Retrieve relevant syllabus documents based on *guessed* topic/subtopic."""
    # Build a search query that emphasizes form and topic/subtopic
    # This relies on the LLM's guess, which might not be optimal
    search_query = f"Syllabus for Form: {form} Topic: {topic} Subtopic: {subtopic}"
    if not topic and not subtopic:
        search_query = f"General syllabus for {form}" # Fallback query

    console.print(f"[dim]Retrieving syllabus for query: '{search_query}'[/dim]")
    return search_qdrant(
        query=search_query,
        collection_name=COLLECTIONS["syllabus"],
        form=form,
        k=k
        # No topic/subtopic filters here for syllabus meta-search
    )

def retrieve_from_content(
    query_for_semantic_search: str, 
    form_for_qdrant_fallback: str,
    classified_form: Optional[str] = None,
    classified_topic: Optional[str] = None, 
    classified_subtopic: Optional[str] = None,
    k: int = 7 # Default k for content retrieval
) -> List[Document]:
    """Retrieve content: 1. Direct JSON lookup, 2. Qdrant semantic search as fallback."""
    
    # Strategy 1: Direct JSON Lookup
    if classified_form and classified_topic and classified_subtopic:
        json_lookup_start_time = time.time()
        console.print(f"[cyan]Attempting direct JSON lookup for: Form='{classified_form}', Topic='{classified_topic}', Subtopic='{classified_subtopic}'[/cyan]")
        try:
            with open(KNOWLEDGE_BANK_PATH, 'r', encoding='utf-8') as f:
                knowledge_bank_data = json.load(f)
            
            matched_json_docs = []
            for item_index, item in enumerate(knowledge_bank_data):
                if "notes" in item and isinstance(item["notes"], dict) and "title" in item["notes"]:
                    parsed_title_tuple = _parse_knowledge_bank_title(item["notes"]["title"])
                    if parsed_title_tuple:
                        kb_form, kb_topic, kb_subtopic = parsed_title_tuple
                        # Normalize for robust comparison
                        if (
                            kb_form.strip().lower() == classified_form.strip().lower() and
                            kb_topic.strip().lower() == classified_topic.strip().lower() and
                            kb_subtopic.strip().lower() == classified_subtopic.strip().lower()
                        ):
                            console.print(f"[bold green]Direct JSON match found: '{item['notes']['title']}'[/bold green]")
                            
                            page_content_parts = [f"Title: {item['notes']['title']}\n"]
                            metadata = {
                                "form": kb_form,
                                "topic": kb_topic,
                                "subtopic": kb_subtopic,
                                "source": "direct_json_match",
                                "original_title": item['notes']['title'],
                                "json_item_index": item_index
                            }

                            if "sections" in item["notes"] and isinstance(item["notes"]["sections"], list):
                                page_content_parts.append("\n--- NOTES SECTIONS ---")
                                for sec_idx, section in enumerate(item["notes"]["sections"]):
                                    if isinstance(section, dict):
                                        page_content_parts.append(f"Section {sec_idx+1}: Heading: {section.get('heading', 'N/A')}")
                                        page_content_parts.append(f"Content: {section.get('content', 'N/A')}")
                            
                            if "worked_examples" in item and isinstance(item["worked_examples"], list):
                                page_content_parts.append("\n--- WORKED EXAMPLES ---")
                                for ex_idx, ex in enumerate(item["worked_examples"]):
                                    if isinstance(ex, dict):
                                        page_content_parts.append(f"Example {ex_idx+1}: Problem: {ex.get('problem', 'N/A')}")
                                        # Safely join steps if they are a list, otherwise stringify
                                        steps_content = ex.get('steps', [])
                                        if isinstance(steps_content, list):
                                            steps_str = "\n".join([f"  - {s}" for s in steps_content])
                                        else:
                                            steps_str = str(steps_content)
                                        page_content_parts.append(f"Steps:\n{steps_str}")
                                        page_content_parts.append(f"Answer: {ex.get('answer', 'N/A')}")

                            if "questions" in item and isinstance(item["questions"], list):
                                page_content_parts.append("\n--- PRACTICE QUESTIONS ---")
                                for q_idx, q_item in enumerate(item["questions"]):
                                    if isinstance(q_item, dict):
                                        page_content_parts.append(f"Question {q_idx+1} (Level: {q_item.get('level', 'N/A')}): {q_item.get('question_text', 'N/A')}")
                            
                            doc = Document(page_content="\n".join(page_content_parts), metadata=metadata)
                            matched_json_docs.append(doc)
                            json_lookup_duration = time.time() - json_lookup_start_time
                            console.print(f"[dim]Direct JSON lookup took {json_lookup_duration:.2f} seconds.[/dim]")
                            console.print(f"[dim]Created {len(matched_json_docs)} document(s) from direct JSON match. Returning.[/dim]")
                            return matched_json_docs
            
            if not matched_json_docs: # Checked all items, no exact title match
                console.print(f"[yellow]No exact title match in JSON for '{classified_form} - {classified_topic} - {classified_subtopic}'.[/yellow]")
            json_lookup_duration = time.time() - json_lookup_start_time
            console.print(f"[dim]Direct JSON lookup (no match) took {json_lookup_duration:.2f} seconds.[/dim]")

        except FileNotFoundError:
            json_lookup_duration = time.time() - json_lookup_start_time
            console.print(f"[bold red]Error: Knowledge bank file '{KNOWLEDGE_BANK_PATH}' not found. (Attempted lookup in {json_lookup_duration:.2f}s)[/bold red]")
        except json.JSONDecodeError:
            json_lookup_duration = time.time() - json_lookup_start_time
            console.print(f"[bold red]Error: Could not decode JSON from '{KNOWLEDGE_BANK_PATH}'. Check its format. (Attempted lookup in {json_lookup_duration:.2f}s)[/bold red]")
        except Exception as e:
            json_lookup_duration = time.time() - json_lookup_start_time
            console.print(f"[bold red]Unexpected error during direct JSON lookup: {e} (Attempted lookup in {json_lookup_duration:.2f}s)[/bold red]")
    else:
        if not (classified_form and classified_topic and classified_subtopic):
             console.print("[dim]Skipping direct JSON lookup: One or more of classified_form, classified_topic, or classified_subtopic is missing or empty.[/dim]")
        else:
             console.print("[dim]Skipping direct JSON lookup for other reasons (e.g. classified info not specific enough).[/dim]")

    # Strategy 2: Qdrant Semantic Search (Fallback if direct lookup was skipped or found nothing)
    qdrant_lookup_start_time = time.time()
    console.print(f"[cyan]Falling back to Qdrant semantic search for content.[/cyan]")
    console.print(f"[dim] Qdrant Query: '{query_for_semantic_search}', Form: '{form_for_qdrant_fallback}', Topic Filter: '{classified_topic}', Subtopic Filter: '{classified_subtopic}'[/dim]")
    try:
        qdrant_docs = search_qdrant(
            query=query_for_semantic_search,
            collection_name=COLLECTIONS["content"],
            form=form_for_qdrant_fallback, 
            k=k,
            filter_topic=classified_topic if classified_topic else None, # Pass if available
            filter_subtopic=classified_subtopic if classified_subtopic else None # Pass if available
        )
        qdrant_lookup_duration = time.time() - qdrant_lookup_start_time
        console.print(f"[dim]Qdrant content search took {qdrant_lookup_duration:.2f} seconds.[/dim]")
        console.print(f"[dim]Retrieved {len(qdrant_docs)} documents from Qdrant content search.[/dim]")
        return qdrant_docs
    except Exception as e:
        qdrant_lookup_duration = time.time() - qdrant_lookup_start_time
        console.print(f"[bold red]Error retrieving content documents via Qdrant: {e} (Attempted search in {qdrant_lookup_duration:.2f}s)[/bold red]")
        return []

def display_full_document(doc: Document, title: str, doc_index: int):
    """Display the full content of a document, adapted for combined setup"""
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
        f"\n[bold]--- Retrieved Chunk Content (page_content) ---[/bold]",
        # Use rich.syntax.Syntax for potential code/math formatting if appropriate
        # For now, just display the text
        doc.page_content
    ]

    # Combine display parts
    display_content = "\n".join(header_section)
    display_content += "\n".join(chunk_content_section)

    # Try to get a unique ID for display
    doc_display_id = metadata.get('_id', metadata.get('id', 'N/A')) # Qdrant payload might not have _id if not added during setup

    panel_title = f"{title} #{doc_index + 1}"
    if doc_display_id != 'N/A':
        panel_title += f" (ID: {doc_display_id})"

    console.print(Panel(
        display_content,
        title=panel_title,
        border_style="blue" if metadata.get('document_type') == 'content' else "green", # Different border for syllabus/content
        width=100, # Adjust width as needed
        expand=False
    ))

def display_full_documents(syllabus_docs: List[Document], content_docs: List[Document]):
    """Display the full content of all retrieved documents"""
    console.print("\n[bold underline]Retrieved Syllabus Documents[/]")
    if not syllabus_docs:
        console.print("(No relevant syllabus documents found based on initial analysis)")
    else:
        for i, doc in enumerate(syllabus_docs):
            display_full_document(doc, "Syllabus", i)

    console.print("\n[bold underline]Retrieved Content Documents[/]")
    if not content_docs:
        console.print("(No relevant content documents found)")
    else:
        for i, doc in enumerate(content_docs):
            display_full_document(doc, "Content", i)

    console.print("\n")

def generate_answer(query: str, syllabus_docs: List[Document], content_docs: List[Document], llm: OllamaLLM, request_type: str = "SOLVE", is_specific_problem: bool = False) -> str:
    """Generate an answer using retrieved documents, LLM, and structured content if available."""
    
    # Check if content_docs come from a direct JSON match and have structured parts
    direct_json_content = None
    has_direct_notes = False
    has_direct_examples = False
    has_direct_questions = False
    # Assuming only one doc if it's a direct full JSON match
    if content_docs and content_docs[0].metadata.get("source") == "direct_json_match":
        direct_json_content = content_docs[0] # The whole matched JSON entry as one Document
        # We infer presence of parts from the page_content structure we created
        # This is a simplification; ideally, metadata would explicitly state this.
        if "--- NOTES SECTIONS ---" in direct_json_content.page_content:
            has_direct_notes = True
        if "--- WORKED EXAMPLES ---" in direct_json_content.page_content:
            has_direct_examples = True
        if "--- PRACTICE QUESTIONS ---" in direct_json_content.page_content:
            has_direct_questions = True
        console.print("[dim]Direct JSON content detected with notes: {}, examples: {}, questions: {}.[/dim]".format(has_direct_notes, has_direct_examples, has_direct_questions))

    syllabus_contexts = []
    for i, doc in enumerate(syllabus_docs):
        metadata = doc.metadata
        context_header = f"REFERENCE SYLLABUS {i+1} (Form: {metadata.get('form', 'N/A')}, Topic: {metadata.get('topic', 'N/A')}, Subtopic: {metadata.get('subtopic', 'N/A')})"
        syllabus_text = f"{context_header}\n---\n{doc.page_content}\n---"
        syllabus_contexts.append(syllabus_text.strip())
    full_context_syllabus = "\n\n".join(syllabus_contexts) if syllabus_contexts else "No relevant syllabus context found."

    # For Qdrant results (non-direct JSON match) or if direct match is not comprehensive
    qdrant_content_contexts = []
    if not direct_json_content: # Only build this if not using a direct full JSON doc
        for i, doc in enumerate(content_docs):
            metadata = doc.metadata
            doc_type = metadata.get('type', 'Generic Content') # This 'type' is from Qdrant metadata if available
            context_header = f"REFERENCE DOCUMENT {i+1} (Type: {doc_type}, Form: {metadata.get('form', 'N/A')}, Topic: {metadata.get('topic', 'N/A')}, Subtopic: {metadata.get('subtopic', 'N/A')})"
            if 'notes_title' in metadata and metadata['notes_title']:
                 context_header += f" (Section Topic: {metadata.get('notes_title')})"
            content_text = f"""{context_header}\n---{doc.page_content}\n---"""
            qdrant_content_contexts.append(content_text.strip())
    full_context_qdrant_content = "\n\n".join(qdrant_content_contexts) if qdrant_content_contexts else "No relevant Qdrant document context found."

    # Use direct_json_content if available, otherwise qdrant_content for the main content context
    # The structure of `page_content` for direct_json_content already includes titles, notes, examples, questions.
    main_content_context = direct_json_content.page_content if direct_json_content else full_context_qdrant_content
    if direct_json_content:
        console.print("[dim]Using directly matched JSON content for LLM context.[/dim]")
    else:
        console.print("[dim]Using Qdrant retrieved content for LLM context.[/dim]")

    prompt_template_str = ""

    if request_type == "EXPLAIN":
        prompt_template_str = """
        You are a knowledgeable and patient math tutor explaining a concept to a student.
        Your goal is to help the student understand the topic based on the provided reference material.

        STUDENT'S QUESTION: {query}

        AVAILABLE REFERENCE MATERIAL:
        --- Syllabus Context ---
        {syllabus_contexts}
        --- End Syllabus Context ---

        --- Content Context (Notes, Examples, Definitions from Knowledge Bank) ---
        {main_content_context}
        --- End Content Context ---

        INSTRUCTIONS:
        1. Carefully review the STUDENT'S QUESTION and the AVAILABLE REFERENCE MATERIAL.
        2. If the Content Context is from a "direct_json_match" (it will be clearly structured with "Title:", "--- NOTES SECTIONS ---", etc.), primarily use the "--- NOTES SECTIONS ---" to formulate your explanation. These notes are curated for teaching.
        3. Provide a clear, step-by-step explanation of the concept related to the student's question. Break down complex ideas into smaller, understandable parts.
        4. Use examples from the "--- WORKED EXAMPLES ---" in the Content Context if they are relevant and help clarify the explanation.
        5. Maintain an encouraging and supportive tone.
        6. Structure your answer clearly. Use Markdown for formatting (headings, bold, lists).
        7. If the reference material does not sufficiently cover the question, state that you can only provide information based on the available references and briefly indicate what's missing.
        8. DO NOT invent information or use external knowledge not present in the references.
        9. Avoid directly quoting the reference headers (e.g., "REFERENCE SYLLABUS 1"). Instead, synthesize the information.
        Respond with the explanation.
        """
    elif request_type == "SOLVE":
        prompt_template_str = """
        You are a helpful math tutor guiding a student through solving a problem.
        Your goal is to demonstrate how to solve the problem using methods and information from the provided reference material.

        STUDENT'S PROBLEM: {query}

        AVAILABLE REFERENCE MATERIAL:
        --- Syllabus Context ---
        {syllabus_contexts}
        --- End Syllabus Context ---

        --- Content Context (Notes, Examples, Definitions from Knowledge Bank) ---
        {main_content_context}
        --- End Content Context ---

        INSTRUCTIONS:
        1. Analyze the STUDENT'S PROBLEM and identify the mathematical concepts involved.
        2. If the Content Context is from a "direct_json_match" (structured with "Title:", "--- WORKED EXAMPLES ---", etc.), prioritize using the methods and steps from the "--- WORKED EXAMPLES ---" to solve the student's problem. These examples are curated and show the expected level-appropriate approach.
        3. Provide a detailed, step-by-step solution. Explain each step clearly.
        4. If relevant, refer to formulas or definitions from the "--- NOTES SECTIONS ---" in the Content Context.
        5. If the reference material does not provide a clear method or necessary information to solve the problem, state this, explain what is missing, and do not attempt to solve it using unverified methods.
        6. Structure your solution clearly using Markdown. Show calculations and formulas appropriately.
        7. DO NOT invent information or use external knowledge not present in the references.
        8. Avoid directly quoting the reference headers. Synthesize the information.
        Respond with the solution.
        """
    elif request_type == "PRACTICE":
        if has_direct_questions and direct_json_content:
            prompt_template_str = """
            You are a math tutor providing practice questions.
            The student has requested practice on a topic. You have access to a curated list of questions from the knowledge bank for this specific topic.

            STUDENT'S REQUEST (Topic Indication): {query}

            AVAILABLE PRE-DEFINED PRACTICE QUESTIONS (from Knowledge Bank - Content Context):
            {main_content_context} 
            (Focus on the "--- PRACTICE QUESTIONS ---" section within the above Content Context)
            
            INSTRUCTIONS:
            1. Identify a relevant practice question from the "--- PRACTICE QUESTIONS ---" section of the AVAILABLE PRE-DEFINED PRACTICE QUESTIONS.
            2. Present ONLY ONE of these questions to the student clearly. For example: "Here is a practice question for you: [Question Text]"
            3. After presenting the question, you can add: "Let me know when you have an answer, or if you'd like another practice question from this set!"
            4. DO NOT solve the question for the student unless they attempt it and ask for help with the solution later (that will be a separate request).
            5. DO NOT generate a new question if relevant pre-defined questions are available.
            Respond by presenting one question.
            """
        else:
            prompt_template_str = """
            You are a math teacher generating practice questions for a student.
            The student needs practice on a topic related to their query.
            No pre-defined questions were found for this specific request, so you need to create suitable ones.

            STUDENT'S REQUEST (Topic Indication): {query}

            AVAILABLE REFERENCE MATERIAL (for inspiration and context):
            --- Syllabus Context ---
            {syllabus_contexts}
            --- End Syllabus Context ---

            --- Content Context (Notes, Examples from Knowledge Bank) ---
            {main_content_context}
            --- End Content Context ---

            INSTRUCTIONS:
            1. Based on the STUDENT'S REQUEST and the AVAILABLE REFERENCE MATERIAL (especially any notes or worked examples), generate 1-2 practice questions that are similar in style and difficulty to what is found in the material.
            2. The questions should be solvable using the concepts and methods described in the reference material.
            3. Present the questions clearly using Markdown.
            4. If the material provides answers or steps for similar examples, you can optionally provide answers hidden in a <details> tag or indicate that answers can be provided upon request.
            5. If the material is insufficient to create relevant practice questions, state that you couldn't generate suitable questions based on the provided context.
            Respond with the generated practice question(s).
            """
    else: # Default or unknown request_type
        prompt_template_str = """You are a helpful AI assistant. Please answer the query based on the provided context.
        Query: {query}
        Syllabus: {syllabus_contexts}
        Content: {main_content_context}
        Answer:"""

    try:
        final_prompt = prompt_template_str.format(
            query=query,
            syllabus_contexts=full_context_syllabus,
            main_content_context=main_content_context
        )
        
        # Basic truncation if prompt is excessively long (adjust limit as needed for the model)
        # This is a very rough estimate. Tokenization is more accurate.
        MAX_PROMPT_CHARS = 15000 # Example limit, might need adjustment
        if len(final_prompt) > MAX_PROMPT_CHARS:
            console.print(f"[yellow]Warning: Prompt length ({len(final_prompt)} chars) exceeds {MAX_PROMPT_CHARS} chars. Truncating.[/yellow]")
            final_prompt = final_prompt[:MAX_PROMPT_CHARS] + "... [PROMPT TRUNCATED]"

        response = llm.invoke(final_prompt)
        return response
    except Exception as e:
        console.print(f"[bold red]Error generating answer with LLM: {e}[/bold red]")
        console.print(f"[bold yellow]Failed prompt (first 500 chars):\n{final_prompt[:500]}...[/bold yellow]")
        return "I encountered an error while trying to generate the answer based on the retrieved information. Please check the logs or try again."


def display_documents_summary(docs: List[Document], title: str):
    """Display retrieved documents summary in a table"""
    if not docs:
        console.print(f"\n[dim](No relevant {title.lower()} found)[/dim]")
        return

    table = Table(title=title, box=None, show_header=True, header_style="bold magenta")
    table.add_column("ID", style="dim", width=10, justify="right")
    table.add_column("Form", style="cyan", width=8)
    table.add_column("Topic", style="green", max_width=25)
    table.add_column("Subtopic", style="yellow", max_width=25)
    table.add_column("Type", style="magenta", width=15)
    table.add_column("Content Snippet", style="blue", max_width=40, overflow="ellipsis")

    for doc in docs:
        metadata = doc.metadata
        # Try to get Qdrant ID, fallback to something else if needed
        doc_display_id = str(metadata.get('_id', metadata.get('id', 'N/A')))[:10] # Shorten ID display

        # Get a condensed version of content for display
        content_preview = doc.page_content.replace('\n', ' ').strip()

        table.add_row(
            doc_display_id,
            metadata.get("form", ""),
            metadata.get("topic", ""),
            metadata.get("subtopic", ""),
            metadata.get("type", metadata.get("document_type", "")),
            content_preview
        )

    console.print(table)

# =========================================================================
# == TYPER COMMANDS ==
# =========================================================================

@app.command()
def classify(
    question: str = typer.Option(..., "--question", "-q", help="The math question to classify"),
    show_details: bool = typer.Option(False, "--show-details", "-d", help="Show full details of the matched syllabus section")
):
    """
    Efficiently identifies the syllabus Topic/Subtopic for a given question using RAG vector search.
    """
    console.print(Panel(f"[bold cyan]Classifying Question:[/bold cyan] {question}", title="Syllabus Classification", border_style="cyan"))

    # Use k=1 for the most likely classification by default for this command
    classified_docs = classify_syllabus_section(question, k=1)

    if classified_docs:
        top_match_doc = classified_docs[0] # Get the first document (since k=1 usually)
        metadata = top_match_doc.metadata
        score = metadata.get('retrieval_score', 0.0) # Get score from metadata

        console.print("\n[bold green underline]Classification Result (Top Match):[/bold green underline]")
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Field", style="bold blue")
        table.add_column("Value")
        table.add_row("Topic", metadata.get('topic', "[Not Detected]"))
        table.add_row("Subtopic", metadata.get('subtopic', "[Not Detected]"))
        table.add_row("Form", metadata.get('form', "[Not Detected]"))
        table.add_row("Similarity Score", f"{score:.4f}") # Display score
        console.print(table)

        if show_details:
             console.print("\n[bold magenta]--- Matched Syllabus Section Details ---[/bold magenta]")
             # Display relevant details directly from the document's metadata and page_content
             console.print(f"[bold]Form:[/bold] {metadata.get('form', 'N/A')}")
             console.print(f"[bold]Topic:[/bold] {metadata.get('topic', 'N/A')}")
             console.print(f"[bold]Subtopic:[/bold] {metadata.get('subtopic', 'N/A')}")
             objectives = metadata.get('objectives', '')
             if objectives: # Only print if objectives exist
                console.print(f"\n[bold]Objectives:[/bold]\n{objectives}")
             console.print(f"\n[bold]--- Retrieved Syllabus Chunk Content ---[/bold]\n{top_match_doc.page_content}")
    else:
        console.print("[bold red]Could not classify the question based on syllabus entries.[/bold red]")

@app.command()
def query(
    question: str = typer.Option(..., "--question", "-q", help="Your math question for full RAG processing"),
    form: str = typer.Option("Form 4", "--form", "-f", help="Your form level (Form 1-4)"),
    show_sources: bool = typer.Option(False, "--show-sources", "-s", help="Show summary table of retrieved sources"),
    show_full_documents: bool = typer.Option(False, "--show-full-documents", "-d", help="Show the full content of retrieved documents"),
):
    """
    Answers a math question using the full RAG pipeline: Accurate Syllabus Classification -> LLM Analysis for Intent -> Retrieval -> LLM generation.
    """
    total_query_start_time = time.time()

    if form not in FORM_LEVELS:
        console.print(f"[bold red]Invalid form level: {form}. Available: {', '.join(FORM_LEVELS)}[/bold red]")
        sys.exit(1)

    console.print(Panel(f"[bold green]Question:[/bold green] {question}\n[bold blue]Form Level (User Default):[/bold blue] {form}", title="Full RAG Query", border_style="green"))

    # Load LLM once for the entire query processing
    llm_load_start_time = time.time()
    llm = get_llm_model() # Load LLM once
    llm_load_duration = time.time() - llm_load_start_time
    console.print(f"[dim]LLM model loading took {llm_load_duration:.2f} seconds.[/dim]")

    # Phase 1: Accurate Syllabus Classification (Vector Search Based)
    console.print("\n[Phase 1: Classifying question against syllabus...]")
    syllabus_classification_start_time = time.time()
    classified_syllabus_docs = classify_syllabus_section(question, k=3)
    syllabus_classification_duration = time.time() - syllabus_classification_start_time
    console.print(f"[dim]Syllabus classification took {syllabus_classification_duration:.2f} seconds.[/dim]")

    classified_form = form # Default to user_form from options
    classified_topic = ""
    classified_subtopic = ""
    top_score = 0.0

    if classified_syllabus_docs:
        # Use metadata from the top classified syllabus document
        top_syllabus_doc_metadata = classified_syllabus_docs[0].metadata
        classified_form = top_syllabus_doc_metadata.get("form", form) # Prioritize classified form
        classified_topic = top_syllabus_doc_metadata.get("topic", "")
        classified_subtopic = top_syllabus_doc_metadata.get("subtopic", "")
        top_score = top_syllabus_doc_metadata.get("retrieval_score", 0.0)
        
        console.print(f"[bold]Syllabus Classification (Top Match):[/bold]")
        summary_table = Table(show_header=False, box=None, padding=(0,1))
        summary_table.add_column("Field", style="dim")
        summary_table.add_column("Value")
        summary_table.add_row("Form", classified_form)
        summary_table.add_row("Topic", classified_topic or "[Not Detected]")
        summary_table.add_row("Subtopic", classified_subtopic or "[Not Detected]")
        summary_table.add_row("Score", f"{top_score:.4f}")
        console.print(summary_table)
    else:
        console.print("[yellow]No direct syllabus classification found. Will rely on LLM analysis for topic/form guidance.[/yellow]")

    # Phase 2: LLM Analysis for Query Rephrasing and Intent
    console.print("\n[Phase 2: Analyzing question with LLM for rephrasing and intent...]")
    llm_analysis_start_time = time.time()
    llm_topic, llm_subtopic, llm_form, rephrased_query, request_type, is_specific_problem = \
        extract_topic_form_from_query(question, form, llm)
    llm_analysis_duration = time.time() - llm_analysis_start_time
    console.print(f"[dim]LLM analysis (rephrasing/intent) took {llm_analysis_duration:.2f} seconds.[/dim]")

    console.print(f"[bold]LLM Analysis (for rephrasing/intent):[/bold]")
    analysis_table = Table(show_header=False, box=None, padding=(0,1))
    analysis_table.add_column("Field", style="dim")
    analysis_table.add_column("Value")
    analysis_table.add_row("LLM Detected Topic", llm_topic or "[Not Detected]")
    analysis_table.add_row("LLM Detected Subtopic", llm_subtopic or "[Not Detected]")
    analysis_table.add_row("LLM Suggested Form", llm_form) # Form suggested by LLM (can be compared with classified_form)
    analysis_table.add_row("Request Type", request_type)
    analysis_table.add_row("Rephrased Query for Content", rephrased_query)
    console.print(analysis_table)

    # Determine final form for content retrieval: prioritize syllabus classification's form if available and relevant.
    # Otherwise, use the form suggested by LLM, or fall back to the user's initial form choice.
    final_form_for_content = classified_form if classified_syllabus_docs else llm_form
    console.print(f"[dim]Using form '{final_form_for_content}' for content retrieval.[/dim]")

    # Phase 3: Retrieve Content Documents
    console.print(f"\n[Phase 3: Retrieving content documents...]")
    
    # Construct a more targeted query for content retrieval
    content_search_query = rephrased_query
    if classified_topic: # Add topic/subtopic if available from syllabus classification
        content_search_query = f"Topic: {classified_topic}"
        if classified_subtopic:
            content_search_query += f" Subtopic: {classified_subtopic}"
        content_search_query += f" Details: {rephrased_query}"
        console.print(f"[dim]Refined content search query using syllabus context: '{content_search_query}'[/dim]")

    content_retrieval_start_time = time.time()
    content_docs = retrieve_from_content(
        query_for_semantic_search=content_search_query, 
        form_for_qdrant_fallback=final_form_for_content, 
        classified_form=classified_form if classified_syllabus_docs and classified_form else None,
        classified_topic=classified_topic if classified_syllabus_docs and classified_topic else None,
        classified_subtopic=classified_subtopic if classified_syllabus_docs and classified_subtopic else None,
        k=7
    )
    content_retrieval_duration = time.time() - content_retrieval_start_time
    console.print(f"[dim]Total content retrieval (retrieve_from_content function call) took {content_retrieval_duration:.2f} seconds.[/dim]")

    # Syllabus documents are `classified_syllabus_docs` from Phase 1.

    # Display retrieved documents
    if show_full_documents:
        display_full_documents(classified_syllabus_docs, content_docs)
    else:
        display_documents_summary(classified_syllabus_docs, "Syllabus")
        display_documents_summary(content_docs, "Content")

    # Phase 4: Generate answer using LLM and retrieved context
    console.print("\n[Phase 4: Generating answer with LLM...]")
    answer_generation_start_time = time.time()
    if not classified_syllabus_docs and not content_docs:
        console.print("[yellow]Warning: No relevant documents found from syllabus or content. Answer quality may be limited.[/yellow]")

    answer = generate_answer(rephrased_query, classified_syllabus_docs, content_docs, llm, request_type, is_specific_problem)
    answer_generation_duration = time.time() - answer_generation_start_time
    console.print(f"[dim]Answer generation took {answer_generation_duration:.2f} seconds.[/dim]")

    # Display final answer
    console.print(f"\n[bold green underline]Final Answer ({request_type}):[/bold green underline]")
    console.print(Panel(Markdown(answer), title="Generated Answer", border_style="green", expand=False))

    total_query_duration = time.time() - total_query_start_time
    console.print(f"[bold magenta]Total query processing time: {total_query_duration:.2f} seconds.[/bold magenta]")

@app.command()
def interactive():
    """Start an interactive session using the full RAG query pipeline"""
    console.print(Panel("[bold green]Math RAG Interactive Session (Full Pipeline)[/bold green]\nType 'exit' or 'quit' to end.", border_style="blue"))

    # Ask for form level once
    form = ""
    while form not in FORM_LEVELS:
        form = typer.prompt(f"Enter your form level ({'/'.join(FORM_LEVELS)})", default="Form 4")
        if form not in FORM_LEVELS:
            console.print(f"[bold red]Invalid form level.[/bold red]")

    # Ask about display preferences
    show_sources = typer.confirm("Show summary of retrieved sources each time?", default=False)
    show_full_documents = typer.confirm("Show full content of retrieved documents each time?", default=False)

    console.print(f"\n[bold green]Settings:[/bold green] Form Level='{form}', Show Sources='{show_sources}', Show Full Docs='{show_full_documents}'")
    console.print("[bold]Enter your math question or type 'exit'/'quit'.[/bold]")

    while True:
        question = typer.prompt("\nYour Question")

        if question.lower() in ["exit", "quit"]:
            console.print("[bold blue]Exiting interactive session.[/bold blue]")
            break

        # Use the existing 'query' function's logic directly
        query(
            question=question,
            form=form,
            show_sources=show_sources,
            show_full_documents=show_full_documents
        )


if __name__ == "__main__":
    app()