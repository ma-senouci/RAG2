import os
import sys
import logging
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List

# Configure logging
PROJECT_LOGGER_NAME = "RAG2"
logger = logging.getLogger(PROJECT_LOGGER_NAME)
logger.setLevel(logging.INFO)

# Handler console
console_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Prevent inheritance issues
logger.propagate = False

# Suppress noise from third-party libraries
for name in logging.root.manager.loggerDict:
    if not name.startswith(PROJECT_LOGGER_NAME):
        logging.getLogger(name).setLevel(logging.WARNING)

class RAGManager:
    """
    Manages the RAG pipeline components including embeddings and vector store.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", persist_directory: str = "./chroma_db", collection_name: str = "portfolio_documents"):
        """
        Initialize RAGManager with embedding model and ChromaDB.
        
        Args:
            model_name (str): The name of the HuggingFace embedding model.
            persist_directory (str): The directory to persist ChromaDB data.
            collection_name (str): The name of the ChromaDB collection.
        """
        self.embedding_model_name = model_name
        
        # Initialize Embeddings
        logger.info(f"Initializing HuggingFaceEmbeddings with model: {self.embedding_model_name}")
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
        
        # ChromaDB configuration
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Check if index exists
        index_exists = False
        if os.path.exists(self.persist_directory):
            # Chroma typically has a chroma.sqlite3 file or multiple files in the directory
            items = os.listdir(self.persist_directory)
            if any(item.endswith('.sqlite3') or item == 'chroma.sqlite3' for item in items) or len(items) > 0:
                index_exists = True
        
        if index_exists:
            logger.info(f"Loading existing index from: {self.persist_directory}")
        else:
            logger.info(f"No existing index found at {self.persist_directory}. Initializing empty collection.")
            
        # Initialize Chroma vector store
        self.vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name=self.collection_name
        )
        
        if index_exists:
            logger.info(f"RAGManager successfully loaded existing index: {self.collection_name}")
        else:
            logger.info(f"RAGManager initialized empty collection: {self.collection_name}")
        
        # Initialize default splitter
        self._splitter = self.get_text_splitter()

    def get_text_splitter(self, chunk_size: int = 500, chunk_overlap: int = 50) -> RecursiveCharacterTextSplitter:
        """
        Create a RecursiveCharacterTextSplitter instance.
        
        Args:
            chunk_size (int): Max size of chunks.
            chunk_overlap (int): Overlap between chunks.
            
        Returns:
            RecursiveCharacterTextSplitter: Configured splitter instance.
        """
        logger.info(f"Creating text splitter with chunk_size={chunk_size} and chunk_overlap={chunk_overlap}")
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    def split_documents(self, documents: List[Document], chunk_size: int = None, chunk_overlap: int = None) -> List[Document]:
        """
        Split documents into chunks and inject sequence tracking metadata.
        
        Args:
            documents (List[Document]): List of documents to split.
            chunk_size (int, optional): Override default chunk size.
            chunk_overlap (int, optional): Override default chunk overlap.
            
        Returns:
            List[Document]: List of chunked documents with global chunk_index.
        """
        # Get appropriate splitter
        if chunk_size is not None or chunk_overlap is not None:
            # Create transient splitter for specific overrides
            size = chunk_size if chunk_size is not None else 500
            overlap = chunk_overlap if chunk_overlap is not None else 50
            splitter = self.get_text_splitter(chunk_size=size, chunk_overlap=overlap)
        else:
            splitter = self._splitter

        logger.info(f"Splitting {len(documents)} documents using {type(splitter).__name__}")
        
        all_chunks = []
        global_index = 0
        for doc in documents:
            doc_chunks = splitter.split_documents([doc])
            # Inject chunk_index globally across all documents in this batch
            for chunk in doc_chunks:
                chunk.metadata["chunk_index"] = global_index
                global_index += 1
            all_chunks.extend(doc_chunks)
            
        logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks
