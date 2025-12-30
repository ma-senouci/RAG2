# RAG2 — Portfolio RAG Chatbot

A production-ready **Retrieval-Augmented Generation** system that lets users interact with an AI grounded in specific source documents. Built with LangChain, ChromaDB, and Gradio.

> Ask questions grounded in your document collection — every answer is backed by verifiable evidence from your uploaded source material.

## ✨ Key Features

- **Semantic Search** — Queries are matched against document embeddings using cosine similarity, not keyword matching
- **Evidence-Grounded Responses** — The LLM cites specific portfolio content; no hallucinated claims
- **Multi-Format Ingestion** — Supports PDF, TXT, and Markdown documents out of the box
- **Tool Calling** — Collects user contact information and flags unanswered questions via Pushover notifications
- **Persistent Index** — ChromaDB stores vectors on disk; no re-indexing on restart
- **Local Embeddings** — Uses `all-MiniLM-L6-v2` for zero-cost, offline vector generation

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│                   app.py                        │
│  ┌───────────┐  ┌───────────────┐  ┌──────────────┐ │
│  │  Gradio   │  │   Persona     │  │ Tool Calling │ │
│  │ ChatUI    │→ │   Handler     │→ │ (Pushover)   │ │
│  └───────────┘  └──────┬────────┘  └──────────────┘ │
│                      │                          │
│              ┌───────▼───────┐                  │
│              │   LLM Service │                  │
│              │ (OpenAI-Compat)│                  │
│              └───────────────┘                  │
└──────────────────────┬──────────────────────────┘
                       │ query_documents()
┌──────────────────────▼──────────────────────────┐
│                 rag_logic.py                    │
│  ┌────────────┐  ┌───────────┐  ┌───────────┐  │
│  │ Document   │  │ Chunking  │  │ Semantic   │  │
│  │ Discovery  │→ │ (500/50)  │→ │ Search     │  │
│  └────────────┘  └───────────┘  └─────┬─────┘  │
│                                       │         │
│              ┌────────────────────────▼───────┐ │
│              │  ChromaDB (cosine similarity)  │ │
│              │  + HuggingFace Embeddings      │ │
│              └────────────────────────────────┘ │
└─────────────────────────────────────────────────┘
```

**Two-file design:**
| File | Responsibility |
|------|---------------|
| `app.py` | UI, LLM orchestration, persona handler, tool calling |
| `rag_logic.py` | Document ingestion, chunking, indexing, retrieval |

## 🛠️ Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| RAG Pipeline | LangChain | Modular chain orchestration |
| Vector Store | ChromaDB | Persistent, serverless, cosine similarity |
| Embeddings | `all-MiniLM-L6-v2` | Local, free, fast (~80MB) |
| LLM Provider | OpenAI-Compatible API | Support for DeepSeek, OpenAI, Anthropic, etc. |
| Interface | Gradio | Chat UI with message history |
| Document Loading | LangChain Loaders | PDF, TXT, Markdown support |

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or 3.12
- A provider API key (e.g., [DeepSeek](https://platform.deepseek.com/) or [OpenAI](https://platform.openai.com/))

### Installation

```bash
git clone https://github.com/ma-senouci/RAG2.git
cd RAG2

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# Edit .env with your API keys
```

### Configuration

Create a `.env` file from the template:

```env
LLM_API_KEY=your-api-key-here
LLM_BASE_URL=https://api.deepseek.com/v1   # Or your preferred provider endpoint

PUSHOVER_TOKEN=your-pushover-token               # optional, for notifications
PUSHOVER_USER=your-pushover-user                 # optional, for notifications

RAG_TOP_K=5                                      # optional, default: 5
```

### Usage

**Step 1 — Index your documents**

Place your PDF, TXT, or MD files in the `docs/` folder, then run:

```bash
python app.py --sync
```

**Step 2 — Chat**

```bash
python app.py
```

This launches the Gradio chat interface at `http://localhost:7860`.

## 📁 Project Structure

```
RAG2/
├── app.py                # Chat UI, RAG augmentation + generation, tool calling
├── rag_logic.py          # Knowledge base: ingestion, chunking, indexing, RAG retrieval
├── docs/                 # Source documents (PDF, TXT, MD)
├── chroma_db/            # Persisted vector index (auto-generated)
├── tests/
│   ├── test_rag_logic.py # RAG pipeline unit tests
│   ├── test_app_chat.py  # Chat integration tests
│   └── test_app_cli.py   # CLI sync tests
├── requirements.txt      # Pinned dependencies
├── .env.example          # Environment variable template
└── README.md
```

## 🧪 Testing

```bash
python -m pytest tests/ -v
```

Tests cover:
- Document discovery and multi-format loading
- Text chunking with metadata tracking
- ChromaDB indexing and persistence verification
- Semantic search with top-k retrieval
- Context formatting and prompt injection
- Conversation history passing
- Tool call handling
- Error handling and graceful degradation

## 📝 How It Works

1. **Sync** — Documents in `docs/` are loaded, split into 500-char chunks (50 overlap), embedded with `all-MiniLM-L6-v2`, and stored in ChromaDB.
2. **Query** — When a user asks a question, the query is embedded and the top-5 most similar chunks are retrieved via cosine similarity.
3. **Augment** — Retrieved chunks (with source citations) are injected into the system prompt alongside the defined AI persona.
4. **Generate** — The LLM produces a grounded response based on the combined information. If it can't answer, it can be extended via tool calling to log the unknown question.

## 🌐 Deployment

This application can be easily deployed to [HuggingFace Spaces](https://huggingface.co/spaces) using the Gradio SDK.

### Steps to Deploy:

1. **Create a New Space:** On HuggingFace, create a new Space and select **Gradio** as the SDK.
2. **Upload Files:** Upload the following files to the Space repository:
   - `app.py`
   - `rag_logic.py`
   - `requirements.txt`
   - `docs/` (or the `chroma_db/` folder to bundle a pre-synced index)
3. **Configure Secrets:** In your Space's **Settings** tab, add the following as "Variables" or "Secrets":
   - `LLM_API_KEY`
   - `LLM_BASE_URL` (optional)
   - `PUSHOVER_TOKEN` / `PUSHOVER_USER` (optional)

The Space will automatically build and launch the interface, providing a public URL for your RAG chatbot.

## License

MIT
