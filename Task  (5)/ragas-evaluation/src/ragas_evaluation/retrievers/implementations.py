"""Implementation of all 7 retrieval methods."""
import os
from typing import List, Any

from langchain.retrievers import (
    ContextualCompressionRetriever,
    EnsembleRetriever,
    MultiQueryRetriever,
    ParentDocumentRetriever,
)
from langchain.storage import InMemoryStore
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rich.console import Console

from ..config import config
from .base import RetrieverInfo, create_vector_store

console = Console()


def create_naive_retriever(documents: List[Document]) -> RetrieverInfo:
    """Create naive dense vector retriever."""
    vector_store = create_vector_store(documents)
    retriever = vector_store.as_retriever(
        search_kwargs={"k": config.retrieval_k}
    )
    return RetrieverInfo(
        name="Naive Retrieval",
        retriever=retriever,
        description="Standard dense vector similarity search using cosine similarity"
    )


def create_bm25_retriever(documents: List[Document]) -> RetrieverInfo:
    """Create BM25 sparse retriever."""
    retriever = BM25Retriever.from_documents(
        documents,
        k=config.retrieval_k
    )
    return RetrieverInfo(
        name="BM25",
        retriever=retriever,
        description="Sparse retrieval using bag-of-words keyword matching"
    )


def create_contextual_compression_retriever(documents: List[Document]) -> RetrieverInfo:
    """Create contextual compression retriever with Cohere reranking."""
    try:
        from langchain_cohere import CohereRerank
        from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
        
        # Create base retriever - get more initial results for reranking
        vector_store = create_vector_store(documents)
        base_retriever = vector_store.as_retriever(
            search_kwargs={"k": config.retrieval_k * 2}  # Get 2x candidates for reranking
        )
        
        # Check if Cohere API key is available
        cohere_api_key = os.getenv("COHERE_API_KEY")
        if not cohere_api_key:
            console.print("[yellow]Warning: COHERE_API_KEY not found, using simplified version[/yellow]")
            return RetrieverInfo(
                name="Contextual Compression (No API Key)",
                retriever=base_retriever,
                description="Dense retrieval (Cohere API key not configured)"
            )
        
        # Create Cohere reranker with latest model
        compressor = CohereRerank(
            cohere_api_key=cohere_api_key,
            model="rerank-english-v3.0",  # Latest model as of 2024
            top_n=config.retrieval_k      # Return top K after reranking
        )
        
        # Create contextual compression retriever
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )
        
        console.print("[green]✓ Created Contextual Compression with Cohere Rerank v3.0[/green]")
        
        return RetrieverInfo(
            name="Contextual Compression",
            retriever=compression_retriever,
            description="Two-stage retrieval with Cohere rerank-english-v3.0 for improved relevance"
        )
        
    except ImportError:
        console.print("[yellow]Warning: langchain-cohere not installed, using simplified version[/yellow]")
        console.print("[dim]Install with: pip install langchain-cohere[/dim]")
        base_retriever = create_naive_retriever(documents).retriever
        
        return RetrieverInfo(
            name="Contextual Compression (Simplified)",
            retriever=base_retriever,
            description="Dense retrieval (install langchain-cohere for full functionality)"
        )


def create_multi_query_retriever(documents: List[Document]) -> RetrieverInfo:
    """Create multi-query retriever."""
    # Create base retriever
    base_retriever = create_naive_retriever(documents).retriever
    
    # Create LLM for query generation
    llm = ChatOpenAI(
        model=config.openai_model,
        temperature=0,
        api_key=config.openai_api_key
    )
    
    # Create multi-query retriever
    retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm
    )
    
    return RetrieverInfo(
        name="Multi-Query",
        retriever=retriever,
        description="Generates multiple query variations for comprehensive retrieval"
    )


def create_parent_document_retriever(documents: List[Document]) -> RetrieverInfo:
    """Create parent document retriever."""
    # Create vector store
    vector_store = create_vector_store(documents)
    
    # Create parent splitter (larger chunks)
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=0
    )
    
    # Create child splitter (smaller chunks for search)
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=200
    )
    
    # Create document store
    store = InMemoryStore()
    
    # Create retriever
    retriever = ParentDocumentRetriever(
        vectorstore=vector_store,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        search_kwargs={"k": config.retrieval_k}
    )
    
    # Add documents
    retriever.add_documents(documents)
    
    return RetrieverInfo(
        name="Parent Document",
        retriever=retriever,
        description="Small-to-big retrieval: searches on small chunks, returns full parents"
    )


def create_ensemble_retriever(documents: List[Document]) -> RetrieverInfo:
    """Create ensemble retriever combining multiple methods."""
    # Create individual retrievers
    bm25 = create_bm25_retriever(documents).retriever
    dense = create_naive_retriever(documents).retriever
    
    # Create ensemble with equal weights
    retriever = EnsembleRetriever(
        retrievers=[bm25, dense],
        weights=[0.5, 0.5]
    )
    
    return RetrieverInfo(
        name="Ensemble",
        retriever=retriever,
        description="Combines BM25 and dense retrieval using Reciprocal Rank Fusion"
    )


def create_semantic_chunking_retriever(documents: List[Document]) -> RetrieverInfo:
    """Create retriever with semantic chunking."""
    from langchain_experimental.text_splitter import SemanticChunker
    from langchain_openai import OpenAIEmbeddings
    
    # Create embeddings
    embeddings = OpenAIEmbeddings(
        model=config.openai_embedding_model,
        api_key=config.openai_api_key
    )
    
    # Create semantic chunker
    text_splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile"
    )
    
    # IMPORTANT: Limit documents to avoid hanging (as shown in the notebook)
    # The notebook only processes 20 documents: semantic_chunker.split_documents(loan_complaint_data[:20])
    limited_docs = documents[:20]  # Same as notebook to avoid hanging
    
    console.print(f"[yellow]Semantic chunking: Processing {len(limited_docs)} documents (limited to avoid hanging)[/yellow]")
    
    # Use split_documents method directly like the notebook
    semantic_docs = text_splitter.split_documents(limited_docs)
    
    # Create vector store with semantic chunks
    vector_store = create_vector_store(semantic_docs)
    retriever = vector_store.as_retriever(
        search_kwargs={"k": config.retrieval_k}
    )
    
    return RetrieverInfo(
        name="Semantic Chunking",
        retriever=retriever,
        description="Splits documents based on semantic similarity for coherent chunks"
    )


def create_parent_child_advanced_retriever(documents: List[Document]) -> RetrieverInfo:
    """Create advanced Parent-Child retriever with Context Corridor."""
    from .parent_child_chunker import ParentChildChunker, ChildChunk
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    import numpy as np
    from pathlib import Path
    
    # Check if Markdown version exists and use it for better structure
    markdown_path = Path("data/Never_split_the_diff.md")
    if markdown_path.exists():
        console.print("[green]Using Markdown version for better structure detection[/green]")
        from ..utils.markdown_loader import load_markdown
        markdown_docs = load_markdown(markdown_path)
        documents_to_use = markdown_docs
    else:
        console.print("[yellow]Using PDF documents (may have poor structure detection)[/yellow]")
        documents_to_use = documents
    
    # Create the chunker with smaller parent chunks for better precision
    # Reduced from 1200 to 600 max tokens to create more focused chunks
    chunker = ParentChildChunker(
        parent_max_tokens=600,    # Reduced from 1200
        parent_min_tokens=200,    # Reduced from 300
        child_min_tokens=100,
        child_max_tokens=140
    )
    
    # Process documents into child chunks
    console.print("[yellow]Creating Parent-Child chunks with Context Corridor...[/yellow]")
    child_chunks = chunker.chunk_documents(documents_to_use)
    
    # Convert child chunks to Documents for vector store
    # IMPROVEMENT: Use hybrid embeddings that include parent context
    child_docs = []
    for child in child_chunks:
        # Create enhanced content for embedding that includes parent context
        parent_preview = child.parent_content[:150] if child.parent_content else ""
        enhanced_content = f"Section: {child.section_heading}\n{child.content}\nParent context: {parent_preview}"
        
        doc = Document(
            page_content=enhanced_content,  # Embed this enhanced content
            metadata={
                'parent_id': child.parent_id,
                'child_idx': child.child_idx,
                'section_heading': child.section_heading,
                'parent_content': child.parent_content,  # Store FULL parent content
                'tokens': child.tokens,
                'original_child_content': child.content  # Keep original for reference
            }
        )
        child_docs.append(doc)
    
    # Create vector store from child documents
    vector_store = create_vector_store(child_docs)
    
    # Create custom retriever with Context Corridor logic
    class ParentChildRetriever(BaseRetriever):
        """Custom retriever implementing Context Corridor enhancement"""
        
        def _get_relevant_documents(
            self,
            query: str,
            *,
            run_manager: CallbackManagerForRetrieverRun = None
        ) -> List[Document]:
            # Get more candidates than needed for deduplication
            # Increased multiplier for better coverage with smaller chunks
            candidates = vector_store.similarity_search_with_score(
                query, 
                k=config.retrieval_k * 5  # Increased from 3x to 5x
            )
            
            # Deduplicate by parent_id, keeping best scoring child
            # Note: In Qdrant, LOWER scores = MORE similar (it's distance, not similarity)
            seen_parents = {}
            for doc, score in candidates:
                parent_id = doc.metadata.get('parent_id')
                if parent_id not in seen_parents or score < seen_parents[parent_id][1]:
                    seen_parents[parent_id] = (doc, score)
            
            # Sort by score (ascending for Qdrant - lower is better)
            sorted_results = sorted(seen_parents.values(), key=lambda x: x[1])
            final_results = []
            
            # Apply Context Corridor logic and return parent content
            for i, (child_doc, score) in enumerate(sorted_results[:config.retrieval_k]):
                # IMPROVEMENT: Return focused excerpt instead of full parent
                parent_content = child_doc.metadata.get('parent_content', child_doc.page_content)
                
                # Find the most relevant 800-char window in parent
                if len(parent_content) > 1000:
                    # Get original child content to find its position
                    child_content = child_doc.metadata.get('original_child_content', '')
                    
                    # Try to find child content in parent
                    child_pos = parent_content.find(child_content)
                    if child_pos >= 0:
                        # Center window around child content
                        start = max(0, child_pos - 200)
                        end = min(len(parent_content), child_pos + len(child_content) + 600)
                        focused_content = parent_content[start:end]
                    else:
                        # Fallback: use first 800 chars
                        focused_content = parent_content[:800]
                else:
                    focused_content = parent_content
                
                parent_doc = Document(
                    page_content=focused_content,
                    metadata={
                        **child_doc.metadata,
                        'child_content': child_doc.metadata.get('original_child_content', ''),
                        'relevance_score': score,
                        'is_focused_excerpt': len(parent_content) > 1000
                    }
                )
                final_results.append(parent_doc)
                
                # Context Corridor: Check if this is the last child of its parent
                # Convert Qdrant distance to similarity (1 - distance)
                similarity = 1 - score
                
                if similarity >= 0.85 and i > 0:
                    # Check if we should add preceding parent for context
                    # This happens when transitioning between related sections
                    curr_parent_id = child_doc.metadata.get('parent_id')
                    
                    # Look for a preceding parent that might provide context
                    for j in range(i):
                        prev_doc, prev_score = sorted_results[j]
                        prev_parent_id = prev_doc.metadata.get('parent_id')
                        
                        # If different parent and high similarity, add it for context
                        if prev_parent_id != curr_parent_id:
                            context_doc = Document(
                                page_content=prev_doc.metadata.get('parent_content', prev_doc.page_content),
                                metadata={
                                    **prev_doc.metadata,
                                    'context_corridor': True,
                                    'added_for_context': True,
                                    'for_parent': curr_parent_id
                                }
                            )
                            final_results.append(context_doc)
                            break
            
            return final_results[:config.retrieval_k]
    
    retriever = ParentChildRetriever()
    
    return RetrieverInfo(
        name="Parent-Child Advanced",
        retriever=retriever,
        description="Advanced Parent-Child chunking with Context Corridor for enhanced retrieval"
    )


def get_all_retrievers(documents: List[Document]) -> List[RetrieverInfo]:
    """Create and return all retriever implementations."""
    retrievers = [
        create_naive_retriever(documents),
        create_bm25_retriever(documents),
        create_contextual_compression_retriever(documents),
        create_multi_query_retriever(documents),
        create_parent_document_retriever(documents),
        create_ensemble_retriever(documents),
        create_semantic_chunking_retriever(documents),
        create_parent_child_advanced_retriever(documents),  # New advanced method
    ]
    return retrievers