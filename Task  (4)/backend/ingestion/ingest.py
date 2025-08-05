"""
Main ingestion script for processing Chris Voss's book and other negotiation materials
"""
import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List

from dual_chunking_pipeline import DualChunkingPipeline
from qdrant_collections import QdrantCollectionManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IngestionService:
    """Main service for ingesting negotiation documents"""
    
    def __init__(self, dual_mode: bool = True, openai_api_key: str = None):
        self.dual_mode = dual_mode
        self.pipeline = DualChunkingPipeline(openai_api_key=openai_api_key)
        self.collection_manager = QdrantCollectionManager()
        
    async def ingest_documents(self, data_dir: str):
        """Ingest all documents from the data directory"""
        data_path = Path(data_dir)
        
        # Check for markdown version of Chris Voss book
        book_path = data_path / "output.md"
        if book_path.exists():
            logger.info(f"Found book: {book_path}")
            await self.ingest_file(str(book_path), is_primary=True)
        else:
            logger.warning("Book 'output.md' not found in data directory")
        
        # Process other markdown files
        for md_file in data_path.glob("*.md"):
            if md_file != book_path:
                logger.info(f"Processing additional material: {md_file}")
                await self.ingest_file(str(md_file), is_primary=False)
        
        # Also process PDFs if any exist
        for pdf_file in data_path.glob("*.pdf"):
            logger.info(f"Processing PDF material: {pdf_file}")
            await self.ingest_file(str(pdf_file), is_primary=False)
    
    async def ingest_file(self, file_path: str, is_primary: bool = False):
        """Ingest a single document file (PDF or Markdown)"""
        try:
            logger.info(f"Starting ingestion of {file_path}")
            
            # Process with dual pipeline
            result = await self.pipeline.process_document(file_path)
            
            if not result.get("success", False):
                logger.error(f"Failed to process document: {result.get('error', 'Unknown error')}")
                return
            
            # Extract chunks from result
            naive_chunks = result.get("naive_chunks", [])
            advanced_chunks = result.get("advanced_chunks", [])
            
            # Store in collections
            if self.dual_mode:
                # The storage is already handled in the pipeline
                logger.info(f"Stored {len(naive_chunks)} naive chunks and {len(advanced_chunks)} advanced chunks")
            else:
                # Single mode - only store in one collection based on config
                logger.info("Single mode ingestion not implemented")
            
            # Generate comparison metrics (already done in pipeline)
            metrics = result.get("comparison", {})
            logger.info(f"Comparison metrics: {metrics}")
            
        except Exception as e:
            logger.error(f"Failed to ingest {file_path}: {e}")
            raise

async def main():
    parser = argparse.ArgumentParser(description="Ingest negotiation documents")
    parser.add_argument("--dual-mode", action="store_true", 
                       help="Process documents with both naive and advanced strategies")
    parser.add_argument("--data-dir", type=str, default="/data",
                       help="Directory containing documents to ingest")
    parser.add_argument("--openai-api-key", type=str, 
                       help="OpenAI API key (optional, will use env var if not provided)")
    
    args = parser.parse_args()
    
    # Initialize service with optional API key
    service = IngestionService(dual_mode=args.dual_mode, openai_api_key=args.openai_api_key)
    
    # Ensure collections exist
    await service.collection_manager.create_collections()
    
    # Run ingestion
    await service.ingest_documents(args.data_dir)
    
    logger.info("Ingestion complete!")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())