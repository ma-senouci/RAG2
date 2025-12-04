import os
import sys
import logging
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

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
        logging.getLogger(name).setLevel(logging.ERROR)

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
        
        # Ensure persistence directory exists
        if not os.path.exists(persist_directory):
            logger.info(f"Creating persistence directory: {persist_directory}")
            os.makedirs(persist_directory)
            
        # Initialize Chroma vector store
        self.vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings,
            collection_name=collection_name
        )
        
        logger.info(f"RAGManager initialized and ChromaDB persisted at: {persist_directory}")
