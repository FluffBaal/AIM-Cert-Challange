"""Document loading utilities for PDF processing."""
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..config import config


def load_pdf(pdf_path: Path) -> List[Document]:
    """Load PDF and return documents."""
    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()
    return documents


def split_documents(
    documents: List[Document], 
    chunk_size: int = config.chunk_size,
    chunk_overlap: int = 200
) -> List[Document]:
    """Split documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    splits = text_splitter.split_documents(documents)
    return splits


def load_and_split_pdf(
    pdf_path: Path = config.pdf_path,
    chunk_size: int = config.chunk_size,
    chunk_overlap: int = 200
) -> List[Document]:
    """Load PDF and split into chunks."""
    documents = load_pdf(pdf_path)
    splits = split_documents(documents, chunk_size, chunk_overlap)
    return splits