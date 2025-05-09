# rag_integration.py

import json
from typing import List, Dict, Optional, Any
from langchain.docstore.document import Document # For type hinting
import os

# Attempt to import from the sibling rag_oo_pipeline package using absolute paths
# (assuming 'temp_test' which contains both this system and rag_oo_pipeline is in sys.path)
try:
    # from ..rag_oo_pipeline.core.embeddings import EmbeddingModelFactory # OLD
    from rag_oo_pipeline.core.embeddings import EmbeddingModelFactory # NEW
    # from ..rag_oo_pipeline.core.vector_store import QdrantService # OLD
    from rag_oo_pipeline.core.vector_store import QdrantService # NEW
    # from ..rag_oo_pipeline.core.knowledge_base import KnowledgeBaseRetriever # OLD
    from rag_oo_pipeline.core.knowledge_base import KnowledgeBaseRetriever # NEW
    # from ..rag_oo_pipeline.pipeline.syllabus_processor import SyllabusProcessor # OLD
    from rag_oo_pipeline.pipeline.syllabus_processor import SyllabusProcessor # NEW
    # from ..rag_oo_pipeline.config import ( # OLD
    from rag_oo_pipeline.config import ( # NEW
        KNOWLEDGE_BANK_PATH, 
        # QDRANT_PATH, # Will be used by QdrantService internally from config
        # QDRANT_API_KEY, # Will be used by QdrantService internally from config
        COLLECTIONS, # Import the whole dictionary
        DEFAULT_SYLLABUS_K,
    )
except ImportError as e:
    print(f"Error importing from rag_oo_pipeline: {e}")
    print("Ensure that the rag_oo_pipeline directory is a sibling to zimsec_tutor_agent_system and paths are correct.")
    # Fallback or re-raise, depending on desired strictness
    raise

# --- Initialize RAG Components ---
# These components will be loaded once when this module is imported.

_initialized = False
syllabus_processor_instance: Optional[SyllabusProcessor] = None
kb_retriever_instance: Optional[KnowledgeBaseRetriever] = None

def _initialize_rag_components():
    global _initialized, syllabus_processor_instance, kb_retriever_instance
    if _initialized:
        return

    print("Initializing RAG components for agent system...")
    try:
        # 0. Path adjustments and Config Monkey-Patching
        # from ..rag_oo_pipeline import config as rag_config # OLD
        from rag_oo_pipeline import config as rag_config # NEW

        current_dir = os.path.dirname(os.path.abspath(__file__))
        temp_test_dir = os.path.dirname(current_dir) 
        rag_pipeline_base_dir = os.path.join(temp_test_dir, 'rag_oo_pipeline')

        # Resolve and monkey-patch KNOWLEDGE_BANK_PATH (though constructor takes precedence for KBRetriever)
        # This is more for illustration if other components used it directly from config
        original_kb_path_setting = rag_config.KNOWLEDGE_BANK_PATH
        actual_abs_kb_path = os.path.abspath(os.path.join(rag_pipeline_base_dir, original_kb_path_setting))
        # rag_config.KNOWLEDGE_BANK_PATH = actual_abs_kb_path # We pass it to constructor, so not strictly needed here

        # Resolve and monkey-patch QDRANT_PATH
        original_qdrant_path_setting = rag_config.QDRANT_PATH
        actual_abs_qdrant_path = os.path.abspath(os.path.join(rag_pipeline_base_dir, original_qdrant_path_setting))
        rag_config.QDRANT_PATH = actual_abs_qdrant_path
        
        print(f"[Debug rag_integration] Patched rag_config.QDRANT_PATH to: {rag_config.QDRANT_PATH}")
        print(f"[Debug rag_integration] Using Knowledge Bank Path: {actual_abs_kb_path}")


        # 1. Embedding Model
        # Ensure EmbeddingModelFactory.get_embedding_model() doesn't require complex global setup
        # or pass necessary config if it does.
        embedding_model = EmbeddingModelFactory.get_embedding_model()

        # 2. Qdrant Service
        # QdrantService might need path, api_key, etc. These should come from rag_oo_pipeline.config
        qdrant_service = QdrantService(
            embedding_model_instance=embedding_model,
            # qdrant_path=QDRANT_PATH, # QdrantService constructor takes these directly
            # api_key=QDRANT_API_KEY,  # if applicable
            # host="localhost", # or from config
            # port=6333 # or from config
            # The current RAG Orchestrator init doesn't pass these to QdrantService,
            # implying QdrantService gets them from config.py itself via imports.
        )

        # 3. Syllabus Processor
        syllabus_processor_instance = SyllabusProcessor(qdrant_service=qdrant_service)

        # 4. Knowledge Base Retriever
        # Pass the correctly resolved absolute path directly to the constructor
        kb_retriever_instance = KnowledgeBaseRetriever(knowledge_bank_path=actual_abs_kb_path)
        
        print("RAG components initialized successfully.")
        _initialized = True

        # It's good practice to restore patched values if the config module could be used by other,
        # unrelated parts of the application later in the same session, though for QdrantService,
        # it initializes its client (and thus reads QDRANT_PATH) only once due to the singleton pattern.
        # For KNOWLEDGE_BANK_PATH, we passed it directly to constructor, so patching wasn't vital for KBRetriever.
        # rag_config.QDRANT_PATH = original_qdrant_path_setting
        # rag_config.KNOWLEDGE_BANK_PATH = original_kb_path_setting


    except Exception as e:
        print(f"Failed to initialize RAG components: {e}")
        # Decide on error handling: raise, or leave components as None
        # For now, let them be None and functions will handle it.
        _initialized = False # Ensure it retries if called again, though typically import is once
        # raise # Or re-raise to make failure explicit

_initialize_rag_components() # Initialize on first import


def get_syllabus_alignment_from_rag(query: str, subject_hint: str, form_level: str) -> Dict[str, Any]:
    """
    Uses the SyllabusProcessor from rag_oo_pipeline to classify the query
    and returns a structured dictionary for the CurriculumAlignmentAgent.
    Args:
        query: The user's query string.
        subject_hint: The current subject context (e.g., "Mathematics").
        form_level: The specific form level (e.g., "Form 4") to filter syllabus by.
    """
    if not syllabus_processor_instance:
        return {
            "is_in_syllabus": False,
            "error": "SyllabusProcessor not initialized",
            "notes_for_orchestrator": "RAG system component (SyllabusProcessor) is not available."
        }

    try:
        # Use COLLECTIONS["syllabus"] for the collection name (SyllabusProcessor does this internally)
        print(f"SyllabusProcessor classifying question against syllabus (Form Filter: {form_level}).") # Added print
        classified_docs: List[Document] = syllabus_processor_instance.classify_question(
            question=query, 
            user_form=form_level, # Use the passed form_level here
            k=1 # Get top 1 match
        )

        if not classified_docs:
            return {
                "is_in_syllabus": False,
                "alignment_score": 0.0,
                "notes_for_orchestrator": "Query did not match any syllabus entries."
            }

        top_match_doc = classified_docs[0]
        metadata = top_match_doc.metadata if top_match_doc.metadata else {}
        
        # Extract details from metadata. Keys are based on common RAG patterns.
        # These might need adjustment based on actual keys in your syllabus Qdrant metadata.
        identified_topic = metadata.get("topic", "Unknown Topic")
        identified_subtopic = metadata.get("subtopic", "Unknown Subtopic")
        # identified_form should now reliably come from the matched document due to filtering
        identified_form = metadata.get("form", form_level) # Fallback to input form_level
        score = metadata.get("retrieval_score", metadata.get("score", 0.0)) # 'score' or 'retrieval_score'
        
        # Placeholder for outcomes and terms - these need to be in Qdrant metadata
        # If they are simple strings in metadata:
        # matched_outcomes = [metadata.get("outcome")] if metadata.get("outcome") else []
        # mandatory_terms = metadata.get("mandatory_terms", "").split(",") if metadata.get("mandatory_terms") else []
        # For now, using placeholders as their structure in metadata is unknown:
        matched_outcomes_str = metadata.get("outcomes", "") # e.g., "Obj1, Obj2" or JSON string
        mandatory_terms_str = metadata.get("mandatory_terms", "") # e.g., "term1, term2" or JSON string

        # Attempt to parse if they are comma-separated or list-like strings
        try:
            if isinstance(matched_outcomes_str, list):
                matched_outcomes = matched_outcomes_str
            elif matched_outcomes_str:
                matched_outcomes = [s.strip() for s in matched_outcomes_str.split(',') if s.strip()]
            else:
                matched_outcomes = []
        except: # Catch any error during parsing
            matched_outcomes = [str(matched_outcomes_str)] if matched_outcomes_str else []

        try:
            if isinstance(mandatory_terms_str, list):
                mandatory_terms = mandatory_terms_str
            elif mandatory_terms_str:
                mandatory_terms = [s.strip() for s in mandatory_terms_str.split(',') if s.strip()]
            else:
                mandatory_terms = []
        except: # Catch any error during parsing
            mandatory_terms = [str(mandatory_terms_str)] if mandatory_terms_str else []

        # syllabus_references would ideally be structured in metadata too
        syllabus_references = metadata.get("references", []) 
        if isinstance(syllabus_references, str): # if it's a string, try to parse as JSON list or keep as string
            try:
                syllabus_references = json.loads(syllabus_references)
                if not isinstance(syllabus_references, list): syllabus_references = [syllabus_references]
            except json.JSONDecodeError:
                 syllabus_references = [{"doc_id": syllabus_references, "page": None, "section": None}]


        return {
            "is_in_syllabus": True,
            "alignment_score": score if isinstance(score, (float, int)) else 0.0,
            "matched_outcomes": matched_outcomes,
            "mandatory_terms": mandatory_terms,
            "syllabus_references": syllabus_references, # e.g., [{"doc_id": "math_syllabus.pdf", "page": 10, "section": "2.3"}]
            "identified_subject": subject_hint, # Use the provided subject_hint
            "identified_topic": identified_topic,
            "identified_subtopic": identified_subtopic,
            "identified_form": identified_form, # Add the identified form
            "gaps": [], # Placeholder
            "notes_for_orchestrator": f"Syllabus match: {identified_topic} - {identified_subtopic} (Score: {score:.2f})",
            "raw_metadata_preview": metadata # For debugging
        }

    except Exception as e:
        print(f"Error in get_syllabus_alignment_from_rag: {e}")
        return {
            "is_in_syllabus": False,
            "error": str(e),
            "notes_for_orchestrator": f"Error during RAG syllabus alignment: {e}"
        }


def get_knowledge_content_from_rag(topic: str, subtopic: str, form: str) -> str:
    """
    Uses the KnowledgeBaseRetriever from rag_oo_pipeline to fetch content
    for a given topic, subtopic, and form.
    Returns a string of knowledge content.
    """
    if not kb_retriever_instance:
        return "Error: KnowledgeBaseRetriever not initialized. Cannot fetch content."

    try:
        # Form is expected by retrieve_direct_match
        matched_docs: List[Document] = kb_retriever_instance.retrieve_direct_match(
            classified_form=form,
            classified_topic=topic,
            classified_subtopic=subtopic
        )

        if not matched_docs:
            # Fallback: If no direct match, could try a broader Qdrant search on content collection
            # For now, just indicate no specific content from JSON KB.
            # Placeholder: A Qdrant search for content would be:
            # if syllabus_processor_instance and syllabus_processor_instance.qdrant_service:
            #   content_docs = syllabus_processor_instance.qdrant_service.search(query=f"{topic} {subtopic}", collection_name="content_collection_name", form=form, k=1)
            #   if content_docs: return content_docs[0].page_content
            return f"No specific knowledge content found via direct JSON lookup for Form '{form}', Topic '{topic}', Subtopic '{subtopic}'."

        # Assuming retrieve_direct_match returns one comprehensive document
        # The page_content is already formatted by KnowledgeBaseRetriever
        return matched_docs[0].page_content

    except Exception as e:
        print(f"Error in get_knowledge_content_from_rag: {e}")
        return f"Error retrieving knowledge content from RAG: {e}"

# Example test (can be run if this file is executed directly, for basic checks)
if __name__ == "__main__":
    print("--- Testing RAG Integration Module ---")
    if not _initialized:
        print("RAG components failed to initialize. Cannot run tests.")
    else:
        print("\\n--- Testing Syllabus Alignment ---")
        # Note: Qdrant must be running and populated for this to work.
        # The form hint should match forms available in your syllabus Qdrant collection.
        # Example: "Form 4", "Form 3", etc.
        test_query = "median from grouped data"
        test_form_hint = "Form 4" 
        alignment = get_syllabus_alignment_from_rag(test_query, "Mathematics", test_form_hint)
        print(f"Alignment for '{test_query}' (Form Hint: {test_form_hint}):")
        print(json.dumps(alignment, indent=2))

        print("\\n--- Testing Knowledge Retrieval ---")
        # These topic/subtopic/form values should correspond to an entry in your math_knowledge_bank.json
        # For example, if you have an entry with title "Form 4: Statistics - Measures of Central Tendency"
        test_topic = alignment.get("identified_topic", "Statistics") 
        test_subtopic = alignment.get("identified_subtopic", "Measures of Central Tendency (Grouped Data)")
        test_form = alignment.get("identified_form", "Form 4")

        if alignment.get("is_in_syllabus"):
            print(f"Retrieving content for Form '{test_form}', Topic '{test_topic}', Subtopic '{test_subtopic}'")
            content = get_knowledge_content_from_rag(test_topic, test_subtopic, test_form)
            print("Retrieved Content (first 500 chars):")
            print(content[:500] + "..." if content else "No content found.")
        else:
            print("Skipping knowledge retrieval test as syllabus alignment failed or returned no match.")

    print("\\n--- RAG Integration Module Test Complete ---") 