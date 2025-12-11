import time
import os
import sys

# Add current dir to path
sys.path.append(os.getcwd())

from rag_logic import RAGManager

start = time.time()
print("Starting profiling...")

t0 = time.time()
from langchain_huggingface import HuggingFaceEmbeddings
print(f"Import HuggingFaceEmbeddings: {time.time() - t0:.2f}s")

t1 = time.time()
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print(f"Initialize Embeddings: {time.time() - t1:.2f}s")

t2 = time.time()
manager = RAGManager()
print(f"Total RAGManager init: {time.time() - t2:.2f}s")

print(f"Total script time: {time.time() - start:.2f}s")
