"""Golden test dataset generation using RAGAS."""
import json
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from rich.console import Console
from rich.progress import track

from ..config import config
from ..utils.document_loader import load_and_split_pdf

console = Console()


def generate_golden_dataset(
    documents: List[Document],
    test_size: int = config.test_dataset_size,
    save_path: Path = None
) -> pd.DataFrame:
    """Generate golden test dataset using RAGAS TestsetGenerator."""
    console.print(f"[bold blue]Generating golden test dataset with {test_size} samples...[/bold blue]")
    
    # Initialize LLM and embeddings with wrappers
    generator_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=config.openai_model,
            temperature=0,
            api_key=config.openai_api_key
        )
    )
    generator_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            model=config.openai_embedding_model,
            api_key=config.openai_api_key
        )
    )
    
    # Create generator
    generator = TestsetGenerator(
        llm=generator_llm,
        embedding_model=generator_embeddings
    )
    
    # Generate testset
    testset = generator.generate_with_langchain_docs(
        documents,
        testset_size=test_size
    )
    
    # Convert to pandas DataFrame
    df = testset.to_pandas()
    
    # Save if path provided
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as CSV for human readability
        csv_path = save_path.with_suffix('.csv')
        df.to_csv(csv_path, index=False)
        console.print(f"[green]Saved golden dataset CSV to: {csv_path}[/green]")
        
        # Save as JSON for programmatic access
        json_path = save_path.with_suffix('.json')
        df.to_json(json_path, orient='records', indent=2)
        console.print(f"[green]Saved golden dataset JSON to: {json_path}[/green]")
        
        # Create human-readable markdown version
        markdown_path = save_path.with_suffix('.md')
        create_readable_markdown(df, markdown_path)
        console.print(f"[green]Saved human-readable version to: {markdown_path}[/green]")
    
    return df


def create_readable_markdown(df: pd.DataFrame, output_path: Path):
    """Create a human-readable markdown version of the golden dataset."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Golden Test Dataset\n\n")
        f.write(f"Total samples: {len(df)}\n\n")
        
        for idx, row in df.iterrows():
            f.write(f"## Sample {idx + 1}\n\n")
            f.write(f"**Question**: {row['question']}\n\n")
            
            if 'contexts' in row and row['contexts']:
                f.write("**Contexts**:\n")
                for i, context in enumerate(row['contexts']):
                    f.write(f"\n*Context {i + 1}*:\n")
                    f.write(f"```\n{context[:500]}{'...' if len(context) > 500 else ''}\n```\n")
            
            if 'ground_truth' in row and row['ground_truth']:
                f.write(f"\n**Ground Truth Answer**: {row['ground_truth']}\n")
            
            if 'evolution_type' in row:
                f.write(f"\n**Question Type**: {row['evolution_type']}\n")
            
            f.write("\n---\n\n")


def load_golden_dataset(path: Path) -> pd.DataFrame:
    """Load golden dataset from file."""
    if path.suffix == '.csv':
        return pd.read_csv(path)
    elif path.suffix == '.json':
        return pd.read_json(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def generate_or_load_golden_dataset(force_regenerate: bool = False) -> pd.DataFrame:
    """Generate golden dataset if it doesn't exist, otherwise load it."""
    golden_path = config.golden_data_dir / "golden_dataset"
    csv_path = golden_path.with_suffix('.csv')
    
    if csv_path.exists() and not force_regenerate:
        console.print("[yellow]Loading existing golden dataset...[/yellow]")
        return load_golden_dataset(csv_path)
    else:
        console.print("[bold blue]Generating new golden dataset...[/bold blue]")
        documents = load_and_split_pdf()
        return generate_golden_dataset(documents, save_path=golden_path)