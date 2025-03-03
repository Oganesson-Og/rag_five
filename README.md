# PDF Document Recognition Pipeline

A robust document processing pipeline for extracting structured information from PDF documents, leveraging spaCyLayout and vision-based recognition.

## Features

- **PDF Document Processing**: Extract text, layout, and structure from PDF documents
- **Table Extraction**: Identify and extract tables as structured data
- **Layout Analysis**: Recognize document layout elements (headings, paragraphs, lists, etc.)
- **Metadata Extraction**: Extract document metadata (title, author, creation date, etc.)
- **Chunking for RAG**: Create optimized text chunks for Retrieval-Augmented Generation
- **Fallback Mechanism**: Vision-based recognition as fallback when text extraction fails
- **Multi-page Support**: Process multi-page documents with page tracking
- **Visualization**: Visualize document layout and recognition results

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/rag_pipeline.git
   cd rag_pipeline
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download spaCy model:
   ```bash
   python -m spacy download en_core_web_sm
   ```

## Usage

### Basic Usage

```python
from src.document_processing.core.vision.pdf_recognizer import PDFRecognizer

# Initialize the PDF recognizer
pdf_recognizer = PDFRecognizer(
    spacy_model="en_core_web_sm",
    device="cuda"  # or "cpu" if GPU is not available
)

# Process a PDF document
result = pdf_recognizer.process_document("path/to/document.pdf")

# Extract text blocks
text_blocks = pdf_recognizer.extract_text_blocks(result)

# Extract tables
tables = pdf_recognizer.extract_tables(result)

# Get document text
text = pdf_recognizer.get_document_text(result)

# Get document chunks for RAG
chunks = pdf_recognizer.get_document_chunks(result)
```

### Command Line Example

The repository includes an example script for processing PDF documents from the command line:

```bash
python examples/pdf_recognition_example.py path/to/document.pdf --output results.json --visualize
```

Options:
- `--output`, `-o`: Output JSON file path
- `--model`, `-m`: spaCy model to use (default: en_core_web_sm)
- `--device`, `-d`: Device to use (cuda, cpu, or auto)
- `--chunk-size`, `-c`: Size of text chunks for RAG
- `--visualize`, `-v`: Visualize the results

## Architecture

The PDF recognition pipeline consists of the following components:

1. **Base Recognizer**: Core vision-based recognition functionality
2. **PDF Recognizer**: PDF-specific processing using spaCyLayout
3. **Document Processing**: Text extraction, layout analysis, and structure recognition
4. **Table Extraction**: Identification and extraction of tables
5. **Chunking**: Creation of optimized text chunks for RAG

## Dependencies

- **spaCy**: Core NLP functionality
- **spaCyLayout**: PDF processing and layout analysis
- **ONNX Runtime**: Efficient model inference
- **pdf2image**: PDF to image conversion
- **PyTesseract**: OCR for fallback mode
- **PyMuPDF**: Alternative PDF parser

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [spaCyLayout](https://github.com/explosion/spacy-layout) for PDF processing
- [ONNX Runtime](https://onnxruntime.ai/) for efficient model inference
- [pdf2image](https://github.com/Belval/pdf2image) for PDF to image conversion 