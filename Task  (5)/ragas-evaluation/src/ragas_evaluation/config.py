"""Configuration module for RAGAS evaluation pipeline."""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class Config:
    """Central configuration for the evaluation pipeline."""
    
    # API Keys
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    cohere_api_key: str = os.getenv("COHERE_API_KEY", "")
    
    # Model Configuration
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o")  # Changed to RAGAS recommended model
    openai_embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    cohere_rerank_model: str = os.getenv("COHERE_RERANK_MODEL", "rerank-v3.5")
    
    # Evaluation Settings
    test_dataset_size: int = int(os.getenv("TEST_DATASET_SIZE", "20"))
    retrieval_k: int = int(os.getenv("RETRIEVAL_K", "10"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "750"))
    max_tokens: int = int(os.getenv("MAX_TOKENS", "32000"))
    temperature: float = float(os.getenv("TEMPERATURE", "0"))
    
    # Paths
    project_root: Path = Path(__file__).parent.parent.parent
    data_dir: Path = project_root / "data"
    output_dir: Path = project_root / os.getenv("OUTPUT_DIR", "outputs")
    golden_data_dir: Path = project_root / os.getenv("GOLDEN_DATA_DIR", "the_gold")
    
    # File paths
    pdf_path: Path = data_dir / "Never split the diff clean.pdf"
    
    def __post_init__(self):
        """Validate configuration and create directories."""
        # Create directories if they don't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.golden_data_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "diagrams").mkdir(exist_ok=True)
        (self.output_dir / "tables").mkdir(exist_ok=True)
        
        # Validate API keys
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        if not self.cohere_api_key:
            raise ValueError("COHERE_API_KEY not found in environment variables")
        
        # Validate data file exists
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found at {self.pdf_path}")


# Global config instance
config = Config()