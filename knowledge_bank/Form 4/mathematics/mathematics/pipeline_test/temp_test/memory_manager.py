import datetime
import logging
from typing import List, Dict, Optional, Tuple
from pydantic import BaseModel, Field
# from openai import OpenAI # No longer needed for local embeddings
import litellm # For interacting with local Ollama models

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
MAX_MEMORY_RETENTION_HOURS = 1
SEMANTIC_SIMILARITY_THRESHOLD = 0.3
# EMBEDDING_MODEL = "text-embedding-ada-002" # OpenAI embedding model - Will be passed or defaulted
DEFAULT_LOCAL_EMBEDDING_MODEL = "nomic-embed-text" # Default local model
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434" # Default Ollama URL

class Interaction(BaseModel):
    """Represents a single turn in a conversation."""
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.now)
    user_query: str
    retrieved_context: Optional[str] = None # Context from RAG for this query
    agent_response: Optional[str] = None

class ConversationMemory:
    """Manages conversation history for multiple students."""
    def __init__(self, 
                 embedding_model_name: str = DEFAULT_LOCAL_EMBEDDING_MODEL, 
                 ollama_base_url: Optional[str] = DEFAULT_OLLAMA_BASE_URL):
        self.history: Dict[str, List[Interaction]] = {}
        self.embedding_model_name = embedding_model_name
        self.ollama_base_url = ollama_base_url

        if not self.ollama_base_url:
            logging.warning(
                "Ollama base URL not provided for ConversationMemory. "
                "Semantic similarity checks will be disabled if local embeddings are needed and URL is missing."
            )
        else:
            logging.info(f"ConversationMemory initialized to use local embedding model '{self.embedding_model_name}' via '{self.ollama_base_url}'")

    async def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Generates an embedding for the given text using a local model via litellm, asynchronously."""
        if not self.ollama_base_url or not self.embedding_model_name:
            logging.warning("Ollama base URL or embedding model name not configured. Cannot generate embedding.")
            return None
        try:
            text = text.replace("\\n", " ") # Sanitize input
            model_identifier = f"ollama/{self.embedding_model_name}"
            
            # Use asynchronous call to litellm.aembedding
            response = await litellm.aembedding(
                model=model_identifier,
                input=[text],
                api_base=self.ollama_base_url
            )
            return response.data[0]["embedding"]
        except Exception as e:
            logging.error(f"Error getting local embedding for model '{self.embedding_model_name}': {e}")
            if "Model not found" in str(e) or "not found" in str(e).lower():
                 logging.error(f"Ensure the model '{self.embedding_model_name}' is available in Ollama at '{self.ollama_base_url}' and spelled correctly.")
                 logging.error(f"You might need to run 'ollama pull {self.embedding_model_name}'.")
            return None

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Computes cosine similarity between two embedding vectors."""
        # This method requires numpy. Ensure it's installed.
        try:
            import numpy as np
        except ImportError:
            logging.error("Numpy is required for cosine similarity. Please install it.")
            return 0.0
            
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0
        return dot_product / (norm_vec1 * norm_vec2)

    def add_interaction(self, student_id: str, user_query: str, 
                        retrieved_context: Optional[str] = None, 
                        agent_response: Optional[str] = None):
        """Adds a new interaction to the student's history."""
        if student_id not in self.history:
            self.history[student_id] = []
        
        interaction = Interaction(
            user_query=user_query,
            retrieved_context=retrieved_context,
            agent_response=agent_response
        )
        self.history[student_id].append(interaction)
        logging.info(f"Added interaction for student {student_id}: Query - '{user_query[:50]}...'")

    def get_last_interaction(self, student_id: str) -> Optional[Interaction]:
        """Retrieves the most recent interaction for a student."""
        if student_id in self.history and self.history[student_id]:
            return self.history[student_id][-1]
        return None

    def clear_history(self, student_id: str):
        """Clears all conversation history for a student."""
        if student_id in self.history:
            self.history[student_id] = []
            logging.info(f"Cleared conversation history for student {student_id}.")

    def _is_new_topic_signal(self, query: str) -> bool:
        """Checks if the query contains keywords indicating a new topic."""
        new_topic_keywords = ["new question", "change topic", "different subject", "start over", "unrelated question"]
        return any(keyword in query.lower() for keyword in new_topic_keywords)

    async def should_reset_context(self, student_id: str, current_query: str) -> Tuple[bool, str]:
        """
        Determines if the context for a student should be reset based on configured logic, asynchronously.
        Returns a tuple (should_reset, reason_for_decision).
        """
        last_interaction = self.get_last_interaction(student_id)

        if not last_interaction:
            return True, "No previous interaction found for this student."

        # 1. Explicit signal from user
        if self._is_new_topic_signal(current_query):
            reason = "Explicit new topic signal detected in user query."
            logging.info(f"Context reset for {student_id}: {reason}")
            return True, reason

        # 2. Time elapsed since last interaction
        time_since_last = datetime.datetime.now() - last_interaction.timestamp
        if time_since_last > datetime.timedelta(hours=MAX_MEMORY_RETENTION_HOURS):
            reason = f"Time elapsed (>{MAX_MEMORY_RETENTION_HOURS}h) since last interaction."
            logging.info(f"Context reset for {student_id}: {reason}")
            return True, reason

        # 2b. Generic follow-up detection: ensure follow-up queries don't reset context
        followup_keywords = ["explain", "further", "more", "detail", "continue", "next", "tell", "elaborate", "expand"]
        words = current_query.lower().split()
        if words and len(words) <= 12 and any(k in words for k in followup_keywords):
            reason = "Continuation query detected; continuing conversation."
            logging.info(f"Context not reset for {student_id}: {reason}")
            return False, reason

        # 3. Semantic similarity check (if embedding client is available)
        if self.embedding_model_name and last_interaction.user_query:
            current_query_embedding = await self._get_embedding(current_query)
            # Reuse embedding if already computed and stored in interaction, or recompute.
            # For simplicity here, recomputing for last_interaction.user_query
            last_query_embedding = await self._get_embedding(last_interaction.user_query)

            if current_query_embedding and last_query_embedding:
                similarity = self._cosine_similarity(current_query_embedding, last_query_embedding)
                logging.info(f"Semantic similarity for {student_id} (current vs. last query): {similarity:.2f}")
                if similarity < SEMANTIC_SIMILARITY_THRESHOLD:
                    reason = f"Low semantic similarity ({similarity:.2f} < {SEMANTIC_SIMILARITY_THRESHOLD}) to previous query."
                    logging.info(f"Context reset for {student_id}: {reason}")
                    return True, reason
            else:
                logging.warning(f"Could not compute semantic similarity for {student_id}; defaulting to continuation for this check.")
        
        return False, "Continuing conversation based on checks."

    def get_formatted_history_for_prompt(self, student_id: str, max_tokens_approx: Optional[int] = 200) -> str:
        """
        Formats the most recent interaction history for augmenting an LLM prompt.
        Manages token size by very basic truncation if needed.
        """
        print(f"Getting formatted history for student {student_id}")
        last_interaction = self.get_last_interaction(student_id)
        if not last_interaction:
            return ""
        print(f"Last interaction collection: \n {last_interaction}")
        prompt_parts = []
        if last_interaction.user_query:
            prompt_parts.append(f"Previous Question: {last_interaction.user_query}")
        if last_interaction.retrieved_context:
            context_preview = last_interaction.retrieved_context
            prompt_parts.append(f"Previous Context: {context_preview}")
        if last_interaction.agent_response:
            prompt_parts.append(f"Previous Response: {last_interaction.agent_response}")
        
        formatted_history = "\\n".join(prompt_parts)

        # Rudimentary token management (word count as proxy, very rough)
        # A proper tokenizer (e.g., tiktoken) is recommended for accurate counting.
        if max_tokens_approx:
            words = formatted_history.split()
            if len(words) > max_tokens_approx:
                truncated_words = words[:max_tokens_approx]
                formatted_history = " ".join(truncated_words) + "..."
                logging.warning(f"Truncated conversation history for student {student_id} due to token limit ({max_tokens_approx} words).")
        
        return formatted_history if formatted_history else "" 