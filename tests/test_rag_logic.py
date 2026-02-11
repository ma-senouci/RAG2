import os
import pytest
from rag_logic import RAGManager
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

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
