# RAG Pipeline

A Retrieval-Augmented Generation (RAG) pipeline for processing and analyzing documents.

## Setup

### Virtual Environment

Create and activate a virtual environment:

```bash
python -m venv rag_env
source rag_env/bin/activate  # On Windows: rag_env\Scripts\activate
```

### Install Dependencies

Install the required packages:

```bash
pip install -r requirements.txt
```

### Configuration

The project uses a YAML configuration file to store settings and API keys. For security reasons, this file is not included in the repository.

1. Copy the example configuration file:

```bash
cp config/rag_config.example.yaml config/rag_config.yaml
```

2. Edit `config/rag_config.yaml` and add your API keys and other settings.

**Important**: Never commit your configuration file with real API keys to the repository. The `.gitignore` file is set up to exclude the `config/` directory and YAML files to prevent accidental exposure of sensitive information.

## Features

- Document processing (PDF, DOCX, Excel, CSV)
- Table extraction from PDFs using multiple libraries (Camelot, Tabula, PDFPlumber)
- OCR for scanned documents
- Image analysis and annotation
- Educational content processing
- Cross-modal analysis (text and images)

## Usage

[Add usage examples here]

## Development

### Git Workflow

The repository is set up with a `.gitignore` file that excludes:

- Virtual environments (`rag_env/`, `venv/`, etc.)
- Configuration files with sensitive information (`config/`, `*.yaml`, etc.)
- Python cache files and build artifacts
- IDE-specific files
- Large model files
- Temporary files and logs

When adding new features that require configuration, add example settings to `config/rag_config.example.yaml` with placeholder values.

### Adding New Dependencies

When adding new dependencies:

1. Install the package: `pip install package-name`
2. Add it to `requirements.txt`: `pip freeze > requirements.txt`

## Troubleshooting

### Ghostscript Issues

If you encounter issues with Ghostscript detection:

1. Ensure Ghostscript is installed on your system:
   - macOS: `brew install ghostscript`
   - Ubuntu: `apt-get install ghostscript`
   - Windows: Download from [Ghostscript website](https://www.ghostscript.com/download.html)

2. Create symbolic links if needed:
   ```bash
   # On macOS
   sudo ln -sf /opt/homebrew/lib/libgs.dylib /usr/local/lib/libgs.so
   sudo ln -sf /opt/homebrew/lib/libgs.dylib /usr/local/lib/libgs.dylib
   ```

### TableStructureRecognizer Issues

If you encounter issues with the TableStructureRecognizer:

1. Check your configuration file to ensure the model settings are correct
2. Ensure you have the necessary API keys for the models you're using
3. Check the logs for specific error messages

## License

[Add license information here] 