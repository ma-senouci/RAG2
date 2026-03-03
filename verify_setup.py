import sys

required_packages = [
    "langchain",
    "langchain-community",
    "langchain-openai",
    "langchain-huggingface",
    "langchain-chroma",
    "chromadb",
    "sentence-transformers",
    "gradio",
    "openai",
    "pypdf",
    "unstructured",
    "python-dotenv",
    "markdown",
    "nltk",
    "pytest",
    "requests"
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
            elif package == "langchain-chroma":
                import langchain_chroma
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
