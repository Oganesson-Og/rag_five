"""
Handles direct retrieval from the structured JSON knowledge base file.

This module provides the `KnowledgeBaseRetriever` class, which is responsible for:
- Loading the `math_knowledge_bank.json` file specified in `config.py` upon initialization.
- Parsing the titles within the JSON data to extract structured information (form, topic, subtopic)
  using the `_parse_knowledge_bank_title` helper function.
- Offering a `retrieve_direct_match` method that attempts to find an exact entry
  in the loaded JSON data based on the classified form, topic, and subtopic.
- If a match is found, it constructs a comprehensive Langchain `Document` containing
  the notes, worked examples, and practice questions from that specific JSON entry.
This allows for precise content retrieval when the syllabus classification is highly confident.
"""
import json
from typing import List, Optional, Tuple
from langchain.docstore.document import Document
from rich.console import Console

from ..config import KNOWLEDGE_BANK_PATH

console = Console()

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
            topic_part = topic_subtopic_part.strip()
            subtopic_part = "" # Default to empty if no '-' found for subtopic
            
        if not form_part or not topic_part: # Basic validation that form and topic were found
            # console.print(f"[yellow]Warning: Could not fully parse title: '{title}'. Missing form or topic.[/yellow]")
            return None
            
        return form_part, topic_part, subtopic_part
    except ValueError: # Handles errors from split if format is unexpected
        # console.print(f"[yellow]Warning: Title '{title}' does not match expected format for parsing.[/yellow]")
        return None
    except Exception as e:
        # console.print(f"[yellow]Unexpected error parsing title '{title}': {e}[/yellow]")
        return None

class KnowledgeBaseRetriever:
    """Retrieves content directly from the JSON knowledge bank file."""

    def __init__(self, knowledge_bank_path: str = KNOWLEDGE_BANK_PATH):
        self.knowledge_bank_path = knowledge_bank_path
        self.knowledge_data = self._load_knowledge_bank()

    def _load_knowledge_bank(self) -> List[dict]:
        """Loads the knowledge bank from the JSON file."""
        try:
            with open(self.knowledge_bank_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                console.print(f"[dim]Knowledge bank '{self.knowledge_bank_path}' loaded successfully. Contains {len(data)} items.[/dim]")
                return data
        except FileNotFoundError:
            console.print(f"[bold red]Error: Knowledge bank file '{self.knowledge_bank_path}' not found.[/bold red]")
            return []
        except json.JSONDecodeError:
            console.print(f"[bold red]Error: Could not decode JSON from '{self.knowledge_bank_path}'. Check its format.[/bold red]")
            return []
        except Exception as e:
            console.print(f"[bold red]Unexpected error loading knowledge bank: {e}[/bold red]")
            return []

    def retrieve_direct_match(
        self,
        classified_form: str,
        classified_topic: str,
        classified_subtopic: str
    ) -> List[Document]:
        """Attempts to find an exact match in the loaded knowledge bank."""
        if not self.knowledge_data:
            console.print("[yellow]Knowledge bank is empty or failed to load. Cannot perform direct match.[/yellow]")
            return []
        
        if not (classified_form and classified_topic and classified_subtopic):
            # console.print("[dim]Skipping direct JSON lookup in KnowledgeBaseRetriever: Incomplete classification details provided.[/dim]")
            return []

        # console.print(f"[dim]KnowledgeBaseRetriever searching for: Form='{classified_form}', Topic='{classified_topic}', Subtopic='{classified_subtopic}'[/dim]")
        matched_docs = []
        for item_index, item in enumerate(self.knowledge_data):
            if "notes" in item and isinstance(item["notes"], dict) and "title" in item["notes"]:
                parsed_title_tuple = _parse_knowledge_bank_title(item["notes"]["title"])
                if parsed_title_tuple:
                    kb_form, kb_topic, kb_subtopic = parsed_title_tuple
                    if (
                        kb_form.strip().lower() == classified_form.strip().lower() and
                        kb_topic.strip().lower() == classified_topic.strip().lower() and
                        kb_subtopic.strip().lower() == classified_subtopic.strip().lower()
                    ):
                        # console.print(f"[dim]Direct JSON match found by KnowledgeBaseRetriever: '{item['notes']['title']}'[/dim]")
                        page_content_parts = [f"Title: {item['notes']['title']}\n"]
                        metadata = {
                            "form": kb_form,
                            "topic": kb_topic,
                            "subtopic": kb_subtopic,
                            "source": "direct_json_match", # Important for downstream logic
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
                                    steps_content = ex.get('steps', [])
                                    steps_str = "\n".join([f"  - {s}" for s in steps_content]) if isinstance(steps_content, list) else str(steps_content)
                                    page_content_parts.append(f"Steps:\n{steps_str}")
                                    page_content_parts.append(f"Answer: {ex.get('answer', 'N/A')}")

                        if "questions" in item and isinstance(item["questions"], list):
                            page_content_parts.append("\n--- PRACTICE QUESTIONS ---")
                            for q_idx, q_item in enumerate(item["questions"]):
                                if isinstance(q_item, dict):
                                    page_content_parts.append(f"Question {q_idx+1} (Level: {q_item.get('level', 'N/A')}): {q_item.get('question_text', 'N/A')}")
                        
                        doc = Document(page_content="\n".join(page_content_parts), metadata=metadata)
                        matched_docs.append(doc)
                        # Found an exact match for the specific title, so we return this item.
                        # If multiple items could match (e.g., less specific title), logic might differ.
                        return matched_docs 
        return [] # No exact match found after checking all items

if __name__ == '__main__':
    print("Testing KnowledgeBaseRetriever...")
    # This test assumes KNOWLEDGE_BANK_PATH points to a valid JSON file with expected structure.
    kb_retriever = KnowledgeBaseRetriever()
    if kb_retriever.knowledge_data: # Check if data loaded
        test_form = "Form 4"
        test_topic = "Statistics"
        test_subtopic = "Measures of Central Tendency and Dispersion"
        print(f"\nAttempting direct match for: {test_form}, {test_topic}, {test_subtopic}")
        
        docs = kb_retriever.retrieve_direct_match(test_form, test_topic, test_subtopic)
        if docs:
            print(f"Successfully retrieved {len(docs)} document(s) via direct match.")
            for i, doc in enumerate(docs):
                print(f"  Doc {i+1} Source: {doc.metadata.get('source')}, Title: {doc.metadata.get('original_title')}")
                # print(f"    Content (first 200 chars): {doc.page_content[:200]}...")
        else:
            print("No direct match found for the test criteria.")
    else:
        print("Knowledge bank data not loaded, skipping direct match test.") 