import argparse
import sys
from rag_logic import RAGManager, logger

def main():
    parser = argparse.ArgumentParser(description="RAG2 Application CLI")
    parser.add_argument("--sync", action="store_true", help="Trigger document discovery and indexing")
    
    args = parser.parse_args()
    
    if args.sync:
        logger.info("Starting RAG Sync Pipeline...")
        try:
            manager = RAGManager()
            docs = manager.discover_documents()
            if not docs:
                logger.warning("No documents found in 'docs/' folder. Index remains unchanged.")
                return
            
            result = manager.index_documents(docs)
            
            # Formatted Summary
            logger.info("=" * 40)
            logger.info("SYNC OPERATION SUMMARY")
            logger.info("-" * 40)
            logger.info(f"Documents Processed: {result.get('documents_processed')}")
            logger.info(f"Chunks Created:      {result.get('chunks_created')}")
            logger.info(f"Chunks Indexed:      {result.get('chunks_indexed')}")
            logger.info(f"Verification:        {result.get('verification', 'N/A').upper()}")
            logger.info("=" * 40)
            logger.info("Sync complete. System is ready for retrieval.")
        except Exception as e:
            logger.error(f"Sync failed: {str(e)}")
            sys.exit(1)
    else:
        logger.info("Starting Gradio UI... (Interface will use the persistence-optimized index)")
        # Future: gr.ChatInterface(...)

if __name__ == "__main__":
    main()
