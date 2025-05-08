"""
Provides services related to the Language Model (LLM).

This module includes:
- `LanguageModelService`: Loads and provides access to the main LLM instance
  (using a singleton pattern to avoid repeated loading) based on `config.py`.
- `QueryAnalyzer`: Utilizes an LLM instance to analyze user queries, extracting
  key information like topic, subtopic, form level, request type, and a
  rephrased query suitable for content retrieval.
"""
from langchain_ollama import OllamaLLM
from rich.console import Console
import json
import sys
from typing import Tuple

from ..config import LLM_MODEL #, other config if needed for prompts

console = Console()

class LanguageModelService:
    """Service to load and provide LLM instances (singleton pattern)."""
    _llm_instance = None

    @staticmethod
    def get_llm(model_name: str = LLM_MODEL):
        """Loads and returns the LLM model instance."""
        if LanguageModelService._llm_instance is None:
            try:
                console.print(f"[dim]Initializing LLM model: {model_name}... This may take a moment.[/dim]")
                LanguageModelService._llm_instance = OllamaLLM(model=model_name)
                # Optional: Test invocation to confirm model is working, e.g., llm.invoke("Hello!")
                console.print(f"[dim]LLM model {model_name} initialized successfully.[/dim]")
            except Exception as e:
                console.print(f"[bold red]Error loading LLM model '{model_name}': {e}[/bold red]")
                console.print(f"[bold yellow]Please make sure Ollama is running and model ('ollama pull {model_name}') is available.[/bold yellow]")
                sys.exit(1)
        return LanguageModelService._llm_instance

class QueryAnalyzer:
    """Uses a provided LLM instance to analyze a user's query."""
    def __init__(self, llm_instance: OllamaLLM):
        self.llm = llm_instance
        self.prompt_template = """
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

    def analyze_query(self, query: str, user_form: str) -> Tuple[str, str, str, str, str, bool]:
        """Extracts topic, subtopic, form, rephrased query, request type, and problem status."""
        try:
            formatted_prompt = self.prompt_template.format(query=query, user_form=user_form)
            response_text = self.llm.invoke(formatted_prompt).strip()

            start_index = response_text.find('{')
            end_index = response_text.rfind('}')

            if start_index != -1 and end_index != -1 and end_index > start_index:
                json_str = response_text[start_index : end_index + 1]
            else:
                json_str = response_text # Fallback if braces are missing
            
            result = json.loads(json_str)
            return (
                result.get("topic", ""),
                result.get("subtopic", ""),
                result.get("form", user_form),
                result.get("query_rephrased", query),
                result.get("request_type", "SOLVE"),
                result.get("is_specific_problem", False)
            )
        except json.JSONDecodeError as json_e:
            raw_response = response_text if 'response_text' in locals() else 'No response generated'
            console.print(f"[bold red]Error parsing JSON from LLM for query analysis: {json_e}[/bold red]")
            console.print(f"[bold yellow]Raw LLM response for analysis:\n---\n{raw_response}\n---[/bold yellow]")
            return "", "", user_form, query, "SOLVE", False # Defaults on error
        except Exception as e:
            console.print(f"[bold red]Unexpected error during query analysis: {e}[/bold red]")
            return "", "", user_form, query, "SOLVE", False # Defaults on error

if __name__ == '__main__':
    # Example Usage
    print("Testing LanguageModelService and QueryAnalyzer...")
    test_llm = LanguageModelService.get_llm() # Loads the model configured in config.py
    
    if test_llm:
        analyzer = QueryAnalyzer(llm_instance=test_llm)
        test_query = "What is the formula for the area of a circle if I am in Form 2?"
        test_user_form = "Form 2"
        
        print(f"\nAnalyzing query: '{test_query}' with default form '{test_user_form}'")
        topic, subtopic, form, rephrased, req_type, is_problem = analyzer.analyze_query(test_query, test_user_form)
        
        print(f"  Detected Topic: {topic}")
        print(f"  Detected Subtopic: {subtopic}")
        print(f"  Using Form: {form}")
        print(f"  Rephrased Query: {rephrased}")
        print(f"  Request Type: {req_type}")
        print(f"  Is Specific Problem: {is_problem}")

        # Test singleton behavior for LLM loader
        llm2 = LanguageModelService.get_llm()
        print(f"\nIs LLM instance the same? {test_llm is llm2}") 