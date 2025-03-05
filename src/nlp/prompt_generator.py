"""
Prompt Generator Module
----------------------

This module provides functionality for generating prompts for language models
in the RAG pipeline. It handles the creation of system, user, and context prompts
with appropriate formatting and content.

Key Features:
- Template-based prompt generation
- Context-aware prompting
- Dynamic prompt construction
- Customizable prompt templates
- Support for different LLM providers

Technical Details:
- Template rendering
- Context integration
- Token management
- Provider-specific formatting

Dependencies:
- typing
- string
- logging
- jinja2>=3.0.0

Example Usage:
    # Create a prompt generator with default templates
    generator = PromptGenerator()
    
    # Generate a system prompt with context
    system_prompt = generator.generate_system_prompt(context)
    
    # Generate a user prompt
    user_prompt = generator.generate_user_prompt(query)
    
    # Generate a complete prompt set
    prompts = generator.generate_prompts(query, context)

Performance Considerations:
- Template caching for efficiency
- Optimized context integration
- Token count management

Author: Keith Satuku
Version: 1.0.0
Created: 2023
License: MIT
"""

import logging
import string
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import json
import re

try:
    import jinja2
    JINJA_AVAILABLE = True
except ImportError:
    JINJA_AVAILABLE = False

logger = logging.getLogger(__name__)

class PromptGenerator:
    """
    Generator for creating prompts for language models in the RAG pipeline.
    
    This class handles the creation of system, user, and context prompts with
    appropriate formatting and content based on templates and configuration.
    """
    
    # Default templates
    DEFAULT_SYSTEM_TEMPLATE = """
    You are a helpful AI assistant with access to the following information:
    
    {{ context }}
    
    Use this information to answer the user's questions accurately and concisely.
    If you don't know the answer or the information is not in the provided context, 
    say so rather than making up information.
    """
    
    DEFAULT_USER_TEMPLATE = """
    {{ query }}
    """
    
    DEFAULT_CONTEXT_TEMPLATE = """
    {% for doc in documents %}
    [Document {{ loop.index }}]
    {{ doc.content }}
    {% if doc.metadata %}
    Metadata: {{ doc.metadata }}
    {% endif %}
    {% endfor %}
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the prompt generator.
        
        Args:
            config: Configuration dictionary with prompt templates and settings.
                   If None, default templates will be used.
        """
        self.config = config or {}
        
        # Set up templates
        self.system_template = self.config.get('system_template', self.DEFAULT_SYSTEM_TEMPLATE)
        self.user_template = self.config.get('user_template', self.DEFAULT_USER_TEMPLATE)
        self.context_template = self.config.get('context_template', self.DEFAULT_CONTEXT_TEMPLATE)
        
        # Set up Jinja environment if available
        if JINJA_AVAILABLE:
            self.jinja_env = jinja2.Environment(
                trim_blocks=True,
                lstrip_blocks=True,
                autoescape=False
            )
            # Compile templates
            self.system_template_compiled = self.jinja_env.from_string(self.system_template)
            self.user_template_compiled = self.jinja_env.from_string(self.user_template)
            self.context_template_compiled = self.jinja_env.from_string(self.context_template)
        else:
            logger.warning("Jinja2 not available. Using simple string templates.")
            self.jinja_env = None
        
        # Additional configuration
        self.max_context_length = self.config.get('max_context_length', 4000)
        self.provider = self.config.get('provider', 'openai')
        
        logger.info(f"Initialized PromptGenerator with provider={self.provider}")
    
    def generate_system_prompt(self, context: Union[str, Dict[str, Any], List[Dict[str, Any]]]) -> str:
        """
        Generate a system prompt with the given context.
        
        Args:
            context: Context information to include in the prompt.
                    Can be a string, a dictionary, or a list of documents.
            
        Returns:
            Formatted system prompt string.
        """
        # Process context if it's not already a string
        if not isinstance(context, str):
            context_str = self._process_context(context)
        else:
            context_str = context
        
        # Truncate context if needed
        if len(context_str) > self.max_context_length:
            logger.warning(f"Context exceeds max length ({len(context_str)} > {self.max_context_length}). Truncating.")
            context_str = context_str[:self.max_context_length] + "... [truncated]"
        
        # Generate prompt using template
        if JINJA_AVAILABLE and self.jinja_env:
            return self.system_template_compiled.render(context=context_str).strip()
        else:
            # Simple string template fallback
            return self.system_template.replace("{{ context }}", context_str).strip()
    
    def generate_user_prompt(self, query: str) -> str:
        """
        Generate a user prompt with the given query.
        
        Args:
            query: User query string.
            
        Returns:
            Formatted user prompt string.
        """
        if JINJA_AVAILABLE and self.jinja_env:
            return self.user_template_compiled.render(query=query).strip()
        else:
            # Simple string template fallback
            return self.user_template.replace("{{ query }}", query).strip()
    
    def generate_prompts(self, query: str, context: Union[str, Dict[str, Any], List[Dict[str, Any]]]) -> Dict[str, str]:
        """
        Generate a complete set of prompts for a conversation.
        
        Args:
            query: User query string.
            context: Context information to include in the prompt.
            
        Returns:
            Dictionary with 'system' and 'user' prompts.
        """
        return {
            'system': self.generate_system_prompt(context),
            'user': self.generate_user_prompt(query)
        }
    
    def _process_context(self, context: Union[Dict[str, Any], List[Dict[str, Any]]]) -> str:
        """
        Process context data into a string format.
        
        Args:
            context: Context data to process.
            
        Returns:
            Processed context string.
        """
        # Handle list of documents
        if isinstance(context, list):
            if JINJA_AVAILABLE and self.jinja_env:
                return self.context_template_compiled.render(documents=context).strip()
            else:
                # Simple processing for list of documents
                result = []
                for i, doc in enumerate(context, 1):
                    content = doc.get('content', '') if isinstance(doc, dict) else getattr(doc, 'content', '')
                    metadata = doc.get('metadata', {}) if isinstance(doc, dict) else getattr(doc, 'metadata', {})
                    
                    doc_str = f"[Document {i}]\n{content}"
                    if metadata:
                        doc_str += f"\nMetadata: {json.dumps(metadata, default=str)}"
                    
                    result.append(doc_str)
                
                return "\n\n".join(result)
        
        # Handle single document or dictionary
        elif isinstance(context, dict):
            content = context.get('content', '')
            metadata = context.get('metadata', {})
            
            result = content
            if metadata:
                result += f"\nMetadata: {json.dumps(metadata, default=str)}"
            
            return result
        
        # Fallback
        return str(context)
    
    def load_template_from_file(self, template_type: str, file_path: Union[str, Path]) -> None:
        """
        Load a template from a file.
        
        Args:
            template_type: Type of template to load ('system', 'user', or 'context').
            file_path: Path to the template file.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Template file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            template_content = f.read()
        
        if template_type == 'system':
            self.system_template = template_content
            if JINJA_AVAILABLE and self.jinja_env:
                self.system_template_compiled = self.jinja_env.from_string(template_content)
        elif template_type == 'user':
            self.user_template = template_content
            if JINJA_AVAILABLE and self.jinja_env:
                self.user_template_compiled = self.jinja_env.from_string(template_content)
        elif template_type == 'context':
            self.context_template = template_content
            if JINJA_AVAILABLE and self.jinja_env:
                self.context_template_compiled = self.jinja_env.from_string(template_content)
        else:
            raise ValueError(f"Unknown template type: {template_type}")
        
        logger.info(f"Loaded {template_type} template from {file_path}")
    
    def format_for_provider(self, prompts: Dict[str, str]) -> Dict[str, Any]:
        """
        Format prompts for the specific provider.
        
        Args:
            prompts: Dictionary with 'system' and 'user' prompts.
            
        Returns:
            Provider-specific formatted prompts.
        """
        if self.provider == 'openai':
            return {
                'messages': [
                    {'role': 'system', 'content': prompts['system']},
                    {'role': 'user', 'content': prompts['user']}
                ]
            }
        elif self.provider == 'anthropic':
            return {
                'prompt': f"System: {prompts['system']}\n\nHuman: {prompts['user']}\n\nAssistant:"
            }
        elif self.provider == 'google':
            return {
                'contents': [
                    {'role': 'system', 'parts': [{'text': prompts['system']}]},
                    {'role': 'user', 'parts': [{'text': prompts['user']}]}
                ]
            }
        else:
            # Default format
            return prompts 