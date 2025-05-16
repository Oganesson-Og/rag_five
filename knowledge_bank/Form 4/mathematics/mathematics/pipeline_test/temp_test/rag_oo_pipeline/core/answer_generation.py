"""
RAG Pipeline - Answer Generation
-----------------------------------

This module defines the `AnswerGenerator` class, responsible for generating
LLM-based responses using retrieved syllabus and content documents.

Key Features:
- Constructs context-aware prompts for the LLM based on user query, retrieved
  documents (syllabus and content), and request type (EXPLAIN, SOLVE, PRACTICE).
- Selects appropriate prompt templates based on request type and content source
  (direct JSON match vs. Qdrant search results).
- Prioritizes structured content from direct JSON matches if available.
- Invokes an LLM instance to generate the final textual answer.
- Includes basic prompt length truncation to prevent exceeding token limits.

Technical Details:
- Takes an initialized LLM instance (`OllamaLLM`) during construction.
- The `generate_llm_answer` method orchestrates prompt construction and LLM invocation.
- Different prompt templates are used for EXPLAIN, SOLVE, and PRACTICE scenarios.
- Handles cases where content is from a direct JSON match or Qdrant search.

Dependencies:
- typing (List)
- langchain.docstore.document (Document)
- langchain_ollama (OllamaLLM)
- rich.console (Console)
- ..config (MAX_PROMPT_CHARS)

Author: Keith Satuku
Version: 1.0.0
Created: 2025
License: MIT
"""
from typing import List
from langchain.docstore.document import Document
from langchain_ollama import OllamaLLM # Actual LLM type
from rich.console import Console

from ..config import MAX_PROMPT_CHARS # For prompt truncation

console = Console()

class AnswerGenerator:
    """Generates answers using an LLM, based on query, syllabus, and content documents."""

    def __init__(self, llm_instance: OllamaLLM):
        self.llm = llm_instance

    def generate_llm_answer(
        self,
        query: str, 
        syllabus_docs: List[Document], 
        content_docs: List[Document], 
        request_type: str = "SOLVE", 
        is_specific_problem: bool = False
    ) -> str:
        """Generates an answer using the provided LLM, contexts, and request type."""
        
        direct_json_content = None
        has_direct_notes = False
        has_direct_examples = False
        has_direct_questions = False
        if content_docs and content_docs[0].metadata.get("source") == "direct_json_match":
            direct_json_content = content_docs[0]
            if "--- NOTES SECTIONS ---" in direct_json_content.page_content: has_direct_notes = True
            if "--- WORKED EXAMPLES ---" in direct_json_content.page_content: has_direct_examples = True
            if "--- PRACTICE QUESTIONS ---" in direct_json_content.page_content: has_direct_questions = True
            # console.print(f"[dim]AnswerGenerator detected direct JSON content. Notes: {has_direct_notes}, Examples: {has_direct_examples}, Questions: {has_direct_questions}")

        syllabus_contexts_str = "\n\n".join([
            f"REFERENCE SYLLABUS {i+1} (Form: {doc.metadata.get('form', 'N/A')}, Topic: {doc.metadata.get('topic', 'N/A')}, Subtopic: {doc.metadata.get('subtopic', 'N/A')})\n---\n{doc.page_content}\n---"
            for i, doc in enumerate(syllabus_docs)
        ]) if syllabus_docs else "No relevant syllabus context found."

        if direct_json_content:
            main_content_context_str = direct_json_content.page_content
            # console.print("[dim]AnswerGenerator using directly matched JSON content for LLM context.[/dim]")
        else:
            qdrant_content_contexts_list = []
            for i, doc in enumerate(content_docs):
                metadata = doc.metadata
                doc_type = metadata.get('type', 'Generic Content')
                context_header = f"REFERENCE DOCUMENT {i+1} (Type: {doc_type}, Form: {metadata.get('form', 'N/A')}, Topic: {metadata.get('topic', 'N/A')}, Subtopic: {metadata.get('subtopic', 'N/A')})"
                if 'notes_title' in metadata and metadata['notes_title']:
                    context_header += f" (Section Topic: {metadata.get('notes_title')})"
                qdrant_content_contexts_list.append(f"{context_header}\n---\n{doc.page_content}\n---")
            main_content_context_str = "\n\n".join(qdrant_content_contexts_list) if qdrant_content_contexts_list else "No relevant Qdrant document context found."
            # console.print("[dim]AnswerGenerator using Qdrant retrieved content for LLM context.[/dim]")

        prompt_template = ""
        if request_type == "EXPLAIN":
            prompt_template = """
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
            prompt_template = """
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
            if has_direct_questions:
                prompt_template = """
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
                prompt_template = """
                You are a math teacher generating practice questions for a student.
                The student needs practice on a topic related to their query.
                No pre-defined questions were found for this specific request, or the directly matched content did not contain a practice questions section. You need to create suitable ones based on the available notes and examples.

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
        else: # Default
            prompt_template = """You are a helpful AI assistant. Please answer the query based on the provided context.
            Query: {query}
            Syllabus: {syllabus_contexts}
            Content: {main_content_context}
            Answer:"""

        final_prompt_str = prompt_template.format(
            query=query,
            syllabus_contexts=syllabus_contexts_str,
            main_content_context=main_content_context_str
        )
        
        if len(final_prompt_str) > MAX_PROMPT_CHARS:
            # console.print(f"[yellow]Warning: AnswerGenerator prompt length ({len(final_prompt_str)} chars) exceeds {MAX_PROMPT_CHARS} chars. Truncating.[/yellow]")
            final_prompt_str = final_prompt_str[:MAX_PROMPT_CHARS] + "... [PROMPT TRUNCATED]"

        try:
            response = self.llm.invoke(final_prompt_str)
            return response
        except Exception as e:
            console.print(f"[bold red]Error during LLM invocation in AnswerGenerator: {e}[/bold red]")
            # console.print(f"[bold yellow]Failed prompt for AnswerGenerator (first 500 chars):\n{final_prompt_str[:500]}...[/bold yellow]")
            return "I encountered an error while trying to generate the final answer. Please check logs."

if __name__ == '__main__':
    # Example Usage (requires other core components and config)
    from ..core.llm import LanguageModelService # For getting an LLM instance
    print("Testing AnswerGenerator...")
    
    try:
        test_llm_instance = LanguageModelService.get_llm()
        answer_gen = AnswerGenerator(llm_instance=test_llm_instance)

        # Mock data for testing
        mock_query = "Explain Pythagoras Theorem."
        mock_syllabus_docs = [
            Document(page_content="Syllabus: Geometry - Triangles - Pythagoras Theorem. Objectives: State and apply the theorem.", metadata={'form':'Form 2', 'topic':'Geometry'})
        ]
        mock_content_docs_direct = [
            Document(
                page_content='''Title: Form 2: Geometry - Pythagoras Theorem

--- NOTES SECTIONS ---
Section 1: Heading: Statement
Content: In a right-angled triangle, the square of the hypotenuse is equal to the sum of the squares of the other two sides (a^2 + b^2 = c^2).

--- WORKED EXAMPLES ---
Example 1: Problem: Find c if a=3, b=4.
Steps:
  - c^2 = 3^2 + 4^2
  - c^2 = 9 + 16
  - c^2 = 25
  - c = 5
Answer: c = 5

--- PRACTICE QUESTIONS ---
Question 1 (Level: Easy): If legs are 6 and 8, what is hypotenuse?''', 
                metadata={'source': 'direct_json_match', 'form':'Form 2', 'topic':'Geometry', 'subtopic':'Pythagoras Theorem'}
            )
        ]
        mock_content_docs_qdrant = [
            Document(page_content="Pythagoras: a^2 + b^2 = c^2. Used for right triangles.", metadata={'type':'notes snippet'})
        ]

        print(f"\n--- Test Case 1: EXPLAIN with Direct JSON Content ---")
        explain_answer_direct = answer_gen.generate_llm_answer(mock_query, mock_syllabus_docs, mock_content_docs_direct, "EXPLAIN")
        print(f"Explain (Direct JSON) - First 100 chars: {explain_answer_direct[:100]}...")

        print(f"\n--- Test Case 2: SOLVE with Qdrant Content ---")
        solve_query = "Hypotenuse is 10, one leg is 6. Find other leg."
        solve_answer_qdrant = answer_gen.generate_llm_answer(solve_query, mock_syllabus_docs, mock_content_docs_qdrant, "SOLVE")
        print(f"Solve (Qdrant) - First 100 chars: {solve_answer_qdrant[:100]}...")

        print(f"\n--- Test Case 3: PRACTICE with Direct JSON Questions ---")
        practice_answer_direct = answer_gen.generate_llm_answer("Practice Pythagoras", mock_syllabus_docs, mock_content_docs_direct, "PRACTICE")
        print(f"Practice (Direct Questions) - First 100 chars: {practice_answer_direct[:100]}...")

        print(f"\n--- Test Case 4: PRACTICE with Qdrant (Generate New) ---")
        practice_answer_qdrant = answer_gen.generate_llm_answer("Practice Pythagoras", mock_syllabus_docs, mock_content_docs_qdrant, "PRACTICE")
        print(f"Practice (Generate New from Qdrant) - First 100 chars: {practice_answer_qdrant[:100]}...")

    except Exception as e:
        print(f"Error in AnswerGenerator test: {e}") 