#!/usr/bin/env python
"""
Test script to verify the content_hash property of the Document class.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def main():
    """Test the content_hash property of the Document class."""
    try:
        from src.rag.models import Document, ContentModality
        
        # Test with different content types
        test_contents = [
            "This is a simple text document",
            b"This is a binary document",
            {"title": "This is a JSON document", "content": "Some content"},
            Path(__file__).read_bytes() if Path(__file__).exists() else b"fallback content"
        ]
        
        print("Testing content_hash property with different content types:\n")
        
        for i, content in enumerate(test_contents):
            try:
                # Create a document with this content
                doc = Document(
                    content=content,
                    source=f"test-source-{i}",
                    modality=ContentModality.TEXT
                )
                
                # Test getting the content_hash
                hash_value = doc.content_hash
                
                # Print results
                content_preview = str(content)[:50] + "..." if len(str(content)) > 50 else str(content)
                print(f"Content type: {type(content).__name__}")
                print(f"Content preview: {content_preview}")
                print(f"Hash value: {hash_value}")
                print(f"Hash length: {len(hash_value)}")
                print("-" * 80)
                
            except Exception as e:
                print(f"Error with content type {type(content).__name__}: {str(e)}")
                print("-" * 80)
                
    except ImportError as e:
        print(f"Import error: {str(e)}")
        print("Make sure you're running this script from the project root directory.")
        return

if __name__ == "__main__":
    main() 