import sys
import subprocess

required_packages = [
    "langchain",
    "langchain-community",
    "langchain-openai",
    "chromadb",
    "sentence-transformers",
    "langchain-huggingface",
    "gradio",
    "pypdf",
    "unstructured",
    "python-dotenv",
    "openai",
    "pytest"
]

def verify_imports():
    print(f"Python version: {sys.version}")
    
    missing = []
    for package in required_packages:
        try:
            # Handle cases where package name != import name
            if package == "python-dotenv":
                import dotenv
            elif package == "langchain-community":
                import langchain_community
            elif package == "langchain-openai":
                import langchain_openai
            elif package == "langchain-huggingface":
                import langchain_huggingface
            elif package == "sentence-transformers":
                import sentence_transformers
            else:
                __import__(package.replace("-", "_"))
            print(f"✅ Successfully imported {package}")
        except ImportError as e:
            print(f"❌ Failed to import {package}: {e}")
            missing.append(package)
            
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        sys.exit(1)
    else:
        print("\nAll required packages verified successfully.")

if __name__ == "__main__":
    verify_imports()
