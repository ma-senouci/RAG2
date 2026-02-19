import os
import pytest
from rag_logic import RAGManager
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfWriter

def test_rag_manager_initialization(tmp_path):
    """Verify that RAGManager initializes with correct components."""
    persist_dir = str(tmp_path / "chroma_db")
    manager = RAGManager(persist_directory=persist_dir)
    
    # Assert embeddings initialization
    assert isinstance(manager.embeddings, HuggingFaceEmbeddings)
    assert manager.embeddings.model_name == "all-MiniLM-L6-v2"
    
    # Assert vector store initialization
    assert isinstance(manager.vector_store, Chroma)
    
def test_persistence_directory_creation(tmp_path):
    """Verify that ChromaDB persistence directory is created"""
    persist_dir = str(tmp_path / "chroma_db")
    
    # Ensure directory does not exist
    assert not os.path.exists(persist_dir)
    
    manager = RAGManager(persist_directory=persist_dir) 
    
    # Should create directory
    assert os.path.isdir(persist_dir)

def test_text_splitter_functional(tmp_path):
    """Verify get_text_splitter returns a splitter that chunks correctly."""
    persist_dir = str(tmp_path / "chroma_db")
    manager = RAGManager(persist_directory=persist_dir)
    splitter = manager.get_text_splitter(chunk_size=100, chunk_overlap=0)
    
    doc = Document(page_content="A" * 250)
    chunks = splitter.split_documents([doc])
    
    assert isinstance(splitter, RecursiveCharacterTextSplitter)
    assert len(chunks) == 3
    assert len(chunks[0].page_content) == 100

def test_split_documents_logic(tmp_path):
    """Verify document splitting and global metadata injection."""
    persist_dir = str(tmp_path / "chroma_db")
    manager = RAGManager(persist_directory=persist_dir)
    
    # Create mock documents
    doc1 = Document(page_content="A" * 600, metadata={"source": "doc1.txt"})
    doc2 = Document(page_content="B" * 100, metadata={"source": "doc2.txt"})
    
    chunks = manager.split_documents([doc1, doc2])
    
    # doc1 (600 chars) should split into 2 chunks (500 size, 50 overlap)
    # doc2 (100 chars) should be 1 chunk
    # Total: 3 chunks
    assert len(chunks) == 3
    
    # Verify metadata preservation and GLOBAL index injection
    assert chunks[0].metadata["source"] == "doc1.txt"
    assert chunks[0].metadata["chunk_index"] == 0
    
    assert chunks[1].metadata["source"] == "doc1.txt"
    assert chunks[1].metadata["chunk_index"] == 1
    
    # doc2 chunk should follow sequence: 2, not reset to 0
    assert chunks[2].metadata["source"] == "doc2.txt"
    assert chunks[2].metadata["chunk_index"] == 2

def test_rag_manager_persistence_loading(tmp_path):
    """Verify that RAGManager loads an existing index from disk."""
    persist_dir = str(tmp_path / "chroma_db")
    collection_name = "test_persistence"
    
    # 1. First initialization: Create index and add a document
    manager1 = RAGManager(persist_directory=persist_dir, collection_name=collection_name)
    doc = Document(page_content="Persistence Test Content", metadata={"id": "test_1"})
    manager1.vector_store.add_documents([doc])
    
    # Ensure it's there
    results1 = manager1.vector_store.similarity_search("Persistence", k=1)
    assert len(results1) == 1
    assert results1[0].page_content == "Persistence Test Content"
    
    # 2. Second initialization: Load existing index
    # We expect RAGManager to load the data from persist_dir
    manager2 = RAGManager(persist_directory=persist_dir, collection_name=collection_name)
    
    # Search WITHOUT adding documents - if loaded correctly, it should find the previous doc
    results2 = manager2.vector_store.similarity_search("Persistence", k=1)
    assert len(results2) == 1
    assert results2[0].page_content == "Persistence Test Content"

def test_rag_manager_graceful_missing_index(tmp_path):
    """Verify RAGManager initializes an empty collection when no index exists."""
    persist_dir = str(tmp_path / "chroma_db")
    collection_name = "test_missing"
    
    # Ensure directory does not exist
    assert not os.path.exists(persist_dir)
    
    manager = RAGManager(persist_directory=persist_dir, collection_name=collection_name)
    
    # Should create directory
    assert os.path.isdir(persist_dir)
    
    # Should be empty
    results = manager.vector_store.similarity_search("Anything", k=1)
    assert len(results) == 0
    
def test_discover_documents_mixed_types(tmp_path):
    """Verify loading of PDF, TXT, and MD files from a directory."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    
    # Create sample files
    (docs_dir / "test1.txt").write_text("Text content", encoding="utf-8")
    (docs_dir / "test2.md").write_text("# Markdown content", encoding="utf-8")
    
    # Simple 1-page PDF
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    with open(docs_dir / "test3.pdf", "wb") as f:
        writer.write(f)
    
    manager = RAGManager(persist_directory=str(tmp_path / "chroma_db"))
    docs = manager.discover_documents(directory_path=str(docs_dir))
    filenames = [os.path.basename(doc.metadata.get("source", "")) for doc in docs]
    assert "test1.txt" in filenames
    assert "test2.md" in filenames
    assert "test3.pdf" in filenames


def test_pdf_multipage_ingestion(tmp_path):
    """Verify that multi-page PDFs are handled correctly (resolving false claim)."""
    docs_dir = tmp_path / "docs_multipage"
    docs_dir.mkdir()
    
    # Simple 2-page PDF
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.add_blank_page(width=612, height=792)
    with open(docs_dir / "multipage.pdf", "wb") as f:
        writer.write(f)
    
    manager = RAGManager(persist_directory=str(tmp_path / "chroma_db"))
    docs = manager.discover_documents(directory_path=str(docs_dir))
    
    # Verify that we get 2 separate document objects (one per page)
    # This proves the loader is initialized correctly for multi-page extraction
    pdf_docs = [d for d in docs if "multipage.pdf" in d.metadata.get("source", "")]
    assert len(pdf_docs) == 2, f"Expected 2 pages, found {len(pdf_docs)}"
    assert pdf_docs[0].metadata.get("page") == 0
    assert pdf_docs[1].metadata.get("page") == 1

def test_md_ingestion_content(tmp_path):
    """Verify content extraction and header handling from a markdown file."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    md_content = "# Header 1\n\nThis is a test.\n\n## Header 2\n\n- Item 1\n- Item 2"
    (docs_dir / "sample.md").write_text(md_content, encoding="utf-8")
    
    manager = RAGManager(persist_directory=str(tmp_path / "chroma_db"))
    docs = manager.discover_documents(directory_path=str(docs_dir))
    
    md_docs = [d for d in docs if d.metadata["source"].endswith(".md")]
    assert len(md_docs) > 0
    assert "Header 1" in md_docs[0].page_content
    assert "Item 1" in md_docs[0].page_content

def test_txt_ingestion_content(tmp_path):
    """Verify content extraction from a text file."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    txt_content = "Plain text content for testing multi-format ingestion. "
    (docs_dir / "sample.txt").write_text(txt_content, encoding="utf-8") # Test encoding detection
    
    manager = RAGManager(persist_directory=str(tmp_path / "chroma_db"))
    docs = manager.discover_documents(directory_path=str(docs_dir))
    
    txt_docs = [d for d in docs if d.metadata["source"].endswith(".txt")]
    assert len(txt_docs) > 0
    assert "Plain text content" in txt_docs[0].page_content

def test_discover_documents_empty_folder(tmp_path):
    """Verify handling of empty directories."""
    docs_dir = tmp_path / "empty_docs"
    docs_dir.mkdir()
    
    manager = RAGManager(persist_directory=str(tmp_path / "chroma_db"))
    
    docs = manager.discover_documents(directory_path=str(docs_dir))
    assert len(docs) == 0

def test_discover_documents_metadata(tmp_path):
    """Verify that 'source' metadata is preserved."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "meta_test.txt").write_text("Metadata test content", encoding="utf-8")
    
    manager = RAGManager(persist_directory=str(tmp_path / "chroma_db"))
    
    docs = manager.discover_documents(directory_path=str(docs_dir))
    assert len(docs) == 1
    assert "meta_test.txt" in docs[0].metadata["source"]
    
def test_index_documents_integration(tmp_path):
    """Verify the full indexing pipeline: split -> add -> retrieve."""
    persist_dir = str(tmp_path / "chroma_db")
    manager = RAGManager(persist_directory=persist_dir)
    
    # Create sample document
    doc = Document(page_content="Integration test for the indexing pipeline.", metadata={"source": "test.txt"})
    
    # Run indexing pipeline
    result = manager.index_documents([doc])
    
    assert result["status"] == "success"
    assert result["documents_processed"] == 1
    assert result["chunks_created"] > 0
    
    # Verify retrieval
    search_results = manager.vector_store.similarity_search("indexing pipeline", k=1)
    assert len(search_results) > 0
    assert "Integration test" in search_results[0].page_content
    assert search_results[0].metadata["source"] == "test.txt"
    assert "chunk_index" in search_results[0].metadata

def test_incremental_indexing(tmp_path):
    """Verify that we can add documents to an existing index without loss."""
    persist_dir = str(tmp_path / "chroma_db")
    manager = RAGManager(persist_directory=persist_dir)
    
    # 1. Index first document
    doc1 = Document(page_content="First document content.", metadata={"source": "doc1.txt"})
    manager.index_documents([doc1])
    
    # 2. Index second document
    doc2 = Document(page_content="Second document content.", metadata={"source": "doc2.txt"})
    manager.index_documents([doc2])
    
    # 3. Verify both are present
    results = manager.vector_store.similarity_search("document", k=10)
    sources = [res.metadata["source"] for res in results]
    assert "doc1.txt" in sources
    assert "doc2.txt" in sources

def test_index_documents_verification_logic(tmp_path):
    """Verify that index_documents correctly reports chunk counts and verification status."""
    persist_dir = str(tmp_path / "chroma_db")
    manager = RAGManager(persist_directory=persist_dir)
    
    doc = Document(page_content="Test document for verification.", metadata={"source": "verify.txt"})
    result = manager.index_documents([doc])
    
    assert result["status"] == "success"
    assert result["documents_processed"] == 1
    assert result["chunks_indexed"] == result["chunks_created"]
    assert result["chunks_indexed"] > 0
    assert result["verification"] == "verified"

class TestRAGQuery:
    """Tests for the query_documents method in RAGManager."""

    @pytest.fixture
    def manager(self, tmp_path):
        persist_dir = str(tmp_path / "chroma_db")
        return RAGManager(persist_directory=persist_dir)

    def test_query_documents_success(self, manager, monkeypatch):
        """Verify successful query returns List[Document] with metadata."""
        # Create mock documents to return
        mock_docs = [
            Document(page_content="Result 1", metadata={"source": "doc1.txt"}),
            Document(page_content="Result 2", metadata={"source": "doc2.txt"})
        ]
        
        # Mock similarity_search on the vector store
        def mock_search(query, k):
            return mock_docs
            
        monkeypatch.setattr(manager.vector_store, "similarity_search", mock_search)
        
        results = manager.query_documents("test query")
        
        assert len(results) == 2
        assert isinstance(results[0], Document)
        assert results[0].page_content == "Result 1"
        assert results[0].metadata["source"] == "doc1.txt"

    def test_query_documents_top_k_param(self, manager, monkeypatch):
        """Verify top_k parameter is passed correctly to similarity_search."""
        captured_k = None
        
        def mock_search(query, k):
            nonlocal captured_k
            captured_k = k
            return []
            
        monkeypatch.setattr(manager.vector_store, "similarity_search", mock_search)
        
        manager.query_documents("test query", top_k=10)
        assert captured_k == 10

    def test_query_documents_default_top_k(self, manager, monkeypatch):
        """Verify default top_k=5 is used when not specified."""
        captured_k = None
        
        def mock_search(query, k):
            nonlocal captured_k
            captured_k = k
            return []
            
        monkeypatch.setattr(manager.vector_store, "similarity_search", mock_search)
        
        manager.query_documents("test query")
        assert captured_k == 5

    def test_query_documents_env_override(self, manager, monkeypatch):
        """Verify RAG_TOP_K environment variable overrides default."""
        monkeypatch.setenv("RAG_TOP_K", "7")
        captured_k = None
        
        def mock_search(query, k):
            nonlocal captured_k
            captured_k = k
            return []
            
        monkeypatch.setattr(manager.vector_store, "similarity_search", mock_search)
        
        manager.query_documents("test query")
        assert captured_k == 7

    def test_query_documents_empty_query(self, manager):
        """Verify empty or whitespace-only query returns empty list."""
        assert manager.query_documents("") == []
        assert manager.query_documents("   ") == []
        assert manager.query_documents(None) == []

        # Should return empty list and not crash
        results = manager.query_documents("test query")
        assert results == []

    def test_query_documents_integration_top_k(self, manager):
        """Integration test verifying top_k limits results from real Chroma collection."""
        # Create some real documents
        docs = [
            Document(page_content="Common theme here", metadata={"source": "doc1.txt"}),
            Document(page_content="Common theme there", metadata={"source": "doc2.txt"}),
            Document(page_content="Unique content", metadata={"source": "doc3.txt"})
        ]
        manager.index_documents(docs)
        
        # Test limit k=1
        results_k1 = manager.query_documents("Common theme", top_k=1)
        assert len(results_k1) == 1
        
        # Test limit k=2
        results_k2 = manager.query_documents("Common theme", top_k=2)
        assert len(results_k2) == 2

    def test_query_documents_invalid_top_k_handling(self, manager, monkeypatch):
        """Verify that negative or zero top_k defaults to 1."""
        captured_k = None
        def mock_search(query, k):
            nonlocal captured_k
            captured_k = k
            return []
        monkeypatch.setattr(manager.vector_store, "similarity_search", mock_search)
        
        manager.query_documents("test", top_k=0)
        assert captured_k == 1
        
        manager.query_documents("test", top_k=-5)
        assert captured_k == 1
