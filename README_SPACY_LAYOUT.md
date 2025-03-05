# spaCy Layout Integration for RAG Pipeline

This document explains how to use the spaCy layout integration in the RAG pipeline for document layout analysis.

## Overview

The RAG pipeline now supports three different layout recognition engines:

1. **spaCy Layout** - Uses the `spacy-layout` library for document structure analysis without requiring API keys
2. **Gemini Vision API** - Uses Google's Gemini Vision API for layout recognition (requires API key)
3. **LayoutLMv3** - Uses Hugging Face's LayoutLMv3 model for document layout analysis (requires API key)

The spaCy layout integration provides a robust alternative that doesn't require external API keys, making it suitable for offline use and privacy-sensitive applications.

## Installation

To use the spaCy layout integration, you need to install the required packages:

```bash
pip install spacy spacy-layout
```

You also need to download a spaCy model:

```bash
python -m spacy download en_core_web_sm
```

## Configuration

To use spaCy layout in your RAG pipeline, update your configuration file (`config/rag_config.yaml`) with the following settings:

```yaml
document_processing:
  pdf:
    layout:
      engine: "spacy"
      spacy_model: "en_core_web_sm"
      confidence: 0.5
      merge_boxes: true
```

### Configuration Options

- `engine`: Set to "spacy" to use the spaCy layout engine
- `spacy_model`: The spaCy model to use (default: "en_core_web_sm")
- `confidence`: Minimum confidence threshold for layout elements (default: 0.5)
- `merge_boxes`: Whether to merge adjacent layout elements (default: true)

## Features

The spaCy layout integration provides the following features:

1. **Document Structure Analysis**: Identifies different elements in the document such as titles, paragraphs, lists, tables, etc.
2. **Reading Order Detection**: Determines the logical reading order of elements in the document
3. **Hierarchical Structure**: Builds a hierarchical representation of the document structure
4. **Markdown Conversion**: Converts the document to markdown format for easier processing
5. **Table Detection**: Identifies tables in the document

## Output Format

The layout analysis results are stored in the `document.doc_info['layout']` field with the following structure:

```json
{
  "text": "Full document text",
  "pages": [
    {
      "page_no": 0,
      "elements": [
        {
          "type": "title",
          "text": "Document Title",
          "bbox": [x1, y1, x2, y2],
          "confidence": 0.95
        },
        {
          "type": "text",
          "text": "Paragraph text...",
          "bbox": [x1, y1, x2, y2],
          "confidence": 0.98
        }
      ]
    }
  ],
  "tables": [
    {
      "id": "table_0_1",
      "page": 0,
      "extraction_method": "spacy_layout",
      "confidence": 0.92,
      "bbox": [x1, y1, x2, y2],
      "text": "Table content..."
    }
  ],
  "markdown": "# Document Title\n\nParagraph text...\n\n"
}
```

## Comparison with Other Engines

| Feature | spaCy Layout | Gemini Vision | LayoutLMv3 |
|---------|-------------|---------------|------------|
| API Key Required | No | Yes | Yes |
| Offline Use | Yes | No | Yes* |
| Table Detection | Yes | Yes | Yes |
| Reading Order | Yes | Yes | Yes |
| Markdown Output | Yes | No | No |
| Language Support | Multiple | Multiple | Multiple |
| Processing Speed | Fast | Medium | Slow |
| Accuracy | Good | Very Good | Excellent |

*LayoutLMv3 requires downloading models but can run offline after that

## Example Usage

Here's an example of how to use the spaCy layout integration in your code:

```python
from src.rag.pipeline import Pipeline

# Initialize the pipeline with spaCy layout configuration
config = {
    "document_processing": {
        "pdf": {
            "layout": {
                "engine": "spacy",
                "spacy_model": "en_core_web_sm"
            }
        }
    }
}

pipeline = Pipeline(config)

# Process a document
document = await pipeline.process_document("path/to/document.pdf")

# Access the layout information
layout_info = document.doc_info.get('layout', {})
markdown = document.doc_info.get('markdown', '')

print(f"Document has {len(layout_info.get('pages', []))} pages")
print(f"Document has {len(layout_info.get('tables', []))} tables")
print(f"Markdown preview: {markdown[:200]}...")
```

## Troubleshooting

If you encounter issues with the spaCy layout integration, check the following:

1. Make sure you have installed the required packages: `spacy` and `spacy-layout`
2. Ensure you have downloaded the spaCy model you specified in the configuration
3. Check the logs for any error messages related to spaCy layout
4. Try using a different spaCy model (e.g., "en_core_web_md" for better accuracy)

## Contributing

If you want to improve the spaCy layout integration, consider the following areas:

1. Adding support for more languages
2. Improving table extraction accuracy
3. Enhancing the reading order detection algorithm
4. Adding support for more document element types
5. Optimizing performance for large documents 