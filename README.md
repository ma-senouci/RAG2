# RAG2

A production-ready Retrieval-Augmented Generation (RAG) system built with LangChain, ChromaDB, and Gradio. Ingest documents, build a searchable vector index, and chat with an AI that answers based on your content.

## Tech Stack

- **LangChain** – RAG pipeline orchestration
- **ChromaDB** – Vector storage with disk persistence
- **sentence-transformers** – Local embeddings (zero API cost)
- **Gradio** – Chat interface
- **OpenAI** – Response generation

## Setup

### Prerequisites

- Python 3.11 or 3.12

### Installation

```bash
# Clone the repository
git clone https://github.com/ma-senouci/RAG2.git
cd RAG2

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your OpenAI API key
```

## Usage

### 1. Synchronize Documents
Before querying, you must populate the vector index. Keep your documents in the `docs/` folder and run:

```bash
python app.py --sync
```

This command triggers:
- **Discovery**: Automatically finds `.pdf`, `.txt`, and `.md` files in `docs/`.
- **Chunking**: Splits documents into manageable pieces with overlap.
- **Indexing**: Generates embeddings and saves them to the local `chroma_db/` storage.

### 2. Start Chat Interface (Coming Soon)
The Gradio UI is currently under development. To start the interface in the future:

```bash
python app.py
```

## Configuration

The system behavior can be tuned via environment variables in `.env`:

- `LLM_API_KEY`: API key for your chosen provider (e.g., `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`, etc.).
- `RAG_TOP_K`: Number of relevant chunks to retrieve for each query (default: `5`).

## Project Structure

```
RAG2/
├── app.py              # Gradio interface + CLI entry point
├── rag_logic.py        # RAG pipeline: ingestion, retrieval, augmentation
├── docs/               # Source documents (PDF, TXT, MD)
├── chroma_db/          # Persisted vector index (auto-generated)
├── requirements.txt    # Pinned dependencies
├── .env.example        # Environment variable template
└── README.md
```

## License

MIT
