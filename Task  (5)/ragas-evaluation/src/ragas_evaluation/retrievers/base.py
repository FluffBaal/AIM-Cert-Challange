"""Base retriever interface and common utilities."""
from abc import ABC, abstractmethod
from typing import List, Dict, Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from ..config import config


class RetrieverInfo:
    """Container for retriever information."""
    
    def __init__(self, name: str, retriever: BaseRetriever, description: str):
        self.name = name
        self.retriever = retriever
        self.description = description


def create_vector_store(documents: List[Document]) -> QdrantVectorStore:
    """Create Qdrant vector store from documents."""
    embeddings = OpenAIEmbeddings(
        model=config.openai_embedding_model,
        api_key=config.openai_api_key
    )
    
    # Create in-memory Qdrant client
    client = QdrantClient(location=":memory:")
    
    # Create collection
    collection_name = "evaluation_docs"
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=1536,  # text-embedding-3-small dimension
            distance=Distance.COSINE
        )
    )
    
    # Create vector store with existing client
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )
    
    # Add documents
    vector_store.add_documents(documents)
    
    return vector_store