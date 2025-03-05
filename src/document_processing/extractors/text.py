"""
Text Document Extractor Module
---------------------------

Specialized extractor for plain text documents with advanced processing.

Key Features:
- Text extraction
- Encoding detection
- Format detection
- Structure preservation
- Metadata parsing
- Character set handling
- Line ending normalization

Technical Details:
- Multiple encoding support
- Unicode handling
- Line ending management
- Whitespace normalization
- Character set detection
- Error handling
- Performance optimization

Dependencies:
- chardet>=4.0.0
- typing-extensions>=4.7.0

Author: Keith Satuku
Version: 2.0.0
Created: 2025
License: MIT
"""

from typing import Dict, Any, Optional, Union
from pathlib import Path
import chardet
from .models import Document
from .base import BaseExtractor, ExtractorResult, DocumentContent

class TextExtractor(BaseExtractor):
    """Handles plain text documents with advanced processing capabilities."""
    
    async def _read_file_content(self, file_path: Union[str, Path]) -> bytes:
        """Read file content as bytes."""
        if isinstance(file_path, str):
            file_path = Path(file_path)
            
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        with open(file_path, 'rb') as f:
            return f.read()
    
    async def extract(self, document: Document) -> Document:
        """Extract text content from document."""
        try:
            # If document already has content, use it
            if document.content:
                if isinstance(document.content, str):
                    document.content = self._process_text(document.content)
                    return document
                elif isinstance(document.content, bytes):
                    raw_content = document.content
                else:
                    raise ValueError(f"Unsupported content type: {type(document.content)}")
            # Otherwise, read from source
            elif document.source and isinstance(document.source, str):
                document.file_path = document.source
                raw_content = await self._read_file_content(document.file_path)
            else:
                raise ValueError("Document must have either content or a valid source file path")
            
            # Try multiple encodings in order of likelihood
            encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            text_content = None
            
            for encoding in encodings_to_try:
                try:
                    text_content = raw_content.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue
                
            if text_content is None:
                # If all standard encodings fail, use chardet as fallback
                encoding_info = self._detect_encoding(raw_content)
                try:
                    text_content = raw_content.decode(encoding_info['encoding'] or 'utf-8')
                except (UnicodeDecodeError, TypeError):
                    # Last resort: decode with errors ignored
                    text_content = raw_content.decode('utf-8', errors='ignore')
            
            document.content = text_content
            document.metadata['extracted_text_length'] = len(text_content)
            document.metadata['encoding_used'] = encoding_info.get('encoding', 'unknown')
            
            return document
            
        except Exception as e:
            self.logger.error(f"Text extraction error: {str(e)}")
            raise

    def _detect_encoding(self, content: bytes) -> Dict[str, Any]:
        """Detect text encoding."""
        try:
            result = chardet.detect(content)
            return {
                'encoding': result['encoding'] or 'utf-8',
                'confidence': result['confidence'],
                'language': result.get('language')
            }
        except Exception:
            return {
                'encoding': 'utf-8',
                'confidence': 1.0,
                'language': None
            }

    def _process_text(
        self,
        text: str,
        normalize_whitespace: bool = True,
        normalize_endings: bool = True
    ) -> str:
        """Process text content."""
        if normalize_endings:
            # Normalize line endings to \n
            text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        if normalize_whitespace:
            # Normalize whitespace while preserving line breaks
            lines = text.split('\n')
            lines = [self._clean_text(line) for line in lines]
            text = '\n'.join(lines)
        
        return text.strip() 