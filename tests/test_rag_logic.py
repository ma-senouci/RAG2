import os
import pytest
from rag_logic import RAGManager
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
    persist_dir = str(tmp_path / "new_chroma_db")
    
    # Ensure it doesn't exist
    assert not os.path.exists(persist_dir)
    
    manager = RAGManager(persist_directory=persist_dir) 
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
    persist_dir = str(tmp_path / "missing_db")
    collection_name = "test_missing"
    
    # Ensure directory does not exist
    assert not os.path.exists(persist_dir)
    
    manager = RAGManager(persist_directory=persist_dir, collection_name=collection_name)
    
    # Should create directory
    assert os.path.exists(persist_dir)
    
    # Should be empty
    results = manager.vector_store.similarity_search("Anything", k=1)
    assert len(results) == 0
