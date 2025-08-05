"""Markdown document loader for structured documents."""
from pathlib import Path
from typing import List
from langchain_core.documents import Document


def load_markdown(file_path: Path) -> List[Document]:
    """Load a markdown file and return as a single document."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Create a single document with the full content
    # This preserves the markdown structure for the Parent-Child chunker
    document = Document(
        page_content=content,
        metadata={
            'source': str(file_path),
            'file_type': 'markdown'
        }
    )
    
    return [document]