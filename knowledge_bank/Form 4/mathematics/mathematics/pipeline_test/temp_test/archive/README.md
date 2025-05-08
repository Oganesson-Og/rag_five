# Mathematics RAG Pipeline

This repository contains a Retrieval-Augmented Generation (RAG) pipeline for mathematics education content. The system allows students to ask questions about mathematics topics and receive tailored answers based on the syllabus, notes, and worked examples.

## Features

- Uses a dual retrieval approach:
  - Syllabus retrieval based on topic/subtopic classification
  - Content retrieval from worked examples and notes
- Form-level filtering to ensure appropriate content
- Interactive CLI interface
- Utilizes local models for embedding and generation
- Full document display for debugging and understanding retrieved sources

## Requirements

To run this system, you'll need:

1. **Python 3.9+**
2. **Ollama** installed and running locally with:
   - `phi4` model for text generation
   - `nomic-embed-text` model for embeddings (dimension: 768)
3. **Python libraries** listed in `requirements.txt`

## Installation

1. Install required Python packages:
```bash
pip install -r requirements.txt
```

2. Make sure Ollama is running with the required models:
```bash
ollama pull phi4
ollama pull nomic-embed-text
```

## Usage

1. **Setup the RAG pipeline**:
```bash
python setup_rag.py
```
This will:
- Load and process the mathematics syllabus, worked examples, and notes
- Create Qdrant vector stores for efficient retrieval
- Embed all documents using Ollama embeddings

2. **Query the RAG pipeline interactively**:
```bash
python query_rag.py interactive
```
This will start an interactive session where you can:
- Select your form level (Form 1-4)
- Choose to show summaries or full content of retrieved documents
- Ask questions about mathematics topics
- Get tailored answers based on the syllabus and content

3. **Query with a single question**:
```bash
python query_rag.py query --question "What is quadratic equation?" --form "Form 4" --show-full-documents
```

4. **Command options**:
   - `--question` or `-q`: Your math question
   - `--form` or `-f`: Your form level (Form 1-4)
   - `--show-sources` or `-s`: Show a summary of retrieved sources
   - `--show-full-documents` or `-d`: Show the full content of retrieved documents

## Understanding the Source Documents

When using the `--show-full-documents` option, the system will display:

1. **Syllabus Documents**: These show the curriculum topics, subtopics, and relevant learning objectives.
2. **Content Documents**: These show the worked examples, explanations, and notes related to the query.

This feature is useful for:
- Understanding why certain answers were generated
- Debugging the retrieval process
- Verifying that form-level filtering is working correctly
- Ensuring the answers are based on relevant course material

## Troubleshooting

If you encounter any issues:

1. **Embedding Model Issues**:
   - If `nomic-embed-text` isn't working, ensure Ollama is running
   - You can modify `setup_rag.py` to set `USE_OLLAMA_EMBEDDINGS = False` to use sentence-transformers instead

2. **Database Issues**:
   - If the vector database becomes corrupted, delete the `qdrant_db` directory and run setup again:
     ```bash
     rm -rf qdrant_db
     python setup_rag.py
     ```

3. **Known Issues**:
   - If you see `could not broadcast input array from shape...`, ensure that the `EMBEDDING_DIMENSION` in `setup_rag.py` matches the actual dimension of your embedding model (768 for nomic-embed-text)

## Customization

You can customize the system by:

1. **Changing the embedding model**:
   - Edit `EMBEDDING_MODEL` in `setup_rag.py` for sentence-transformers
   - Edit `OLLAMA_EMBEDDING_MODEL` to use a different Ollama model
   - Ensure you update `EMBEDDING_DIMENSION` to match your model

2. **Changing the LLM**:
   - Edit `LLM_MODEL` in `query_rag.py` to use a different Ollama model
   - You may need to adjust the prompts for different models

## Data Sources

The system uses three JSON files as sources:
- `mathematics_syllabus_chunked.json`: Mathematics curriculum syllabus
- `worked_examples_chunked.json`: Worked examples for various topics
- `first_maths_notes_chunked_cleaned.json`: Notes on mathematics topics

## Cleanup

To remove the vector database:
```
rm -rf ./qdrant_db
``` 