#!/usr/bin/env python
"""
Evaluate retrievers for downstream generation tasks using RAGAS generation metrics.
This measures how well retrieved contexts support answer generation.
"""
import asyncio
from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from rich.console import Console
from rich.table import Table

from src.ragas_evaluation.evaluation.generation_evaluator import (
    evaluate_generation_performance,
    display_generation_results,
    save_generation_results
)
from src.ragas_evaluation.utils.document_loader import load_pdf
from src.ragas_evaluation.retrievers.implementations import get_all_retrievers
from src.ragas_evaluation.config import config

console = Console()


async def main():
    console.print("[bold]RAGAS Generation-Focused Evaluation[/bold]")
    console.print("Evaluating how well retrievers support answer generation\n")
    
    console.print("[yellow]Generation vs Retrieval Metrics:[/yellow]")
    console.print("• Generation metrics evaluate the complete RAG pipeline")
    console.print("• They measure if retrieved context enables good answers")
    console.print("• Parent-Child may perform better here due to fuller context\n")
    
    # Load documents
    documents = load_pdf(config.data_dir / "Never split the diff clean.pdf")
    console.print(f"✓ Loaded {len(documents)} documents")
    
    # Load golden dataset
    golden_path = config.data_dir.parent / "the_gold" / "ragas_golden_dataset.csv"
    if not golden_path.exists():
        console.print("[red]Golden dataset not found! Run generation first.[/red]")
        return
    
    test_dataset = pd.read_csv(golden_path)
    console.print(f"✓ Loaded {len(test_dataset)} test questions")
    
    # Get retrievers
    retrievers = get_all_retrievers(documents)
    console.print(f"✓ Initialized {len(retrievers)} retrievers")
    
    # Show metrics being used
    console.print("\n[bold]Generation Metrics Being Evaluated:[/bold]")
    console.print("1. [cyan]Response Relevancy[/cyan] - How relevant is the generated answer")
    console.print("2. [cyan]Faithfulness[/cyan] - Is the answer grounded in retrieved context")
    console.print("3. [cyan]Semantic Similarity[/cyan] - Semantic similarity to expected answer\n")
    
    # Run evaluation
    console.print("[bold]Starting generation evaluation...[/bold]")
    results = []
    
    for retriever_info in retrievers:
        try:
            result = await evaluate_generation_performance(retriever_info, test_dataset)
            results.append(result)
        except Exception as e:
            console.print(f"[red]Failed to evaluate {retriever_info.name}: {e}[/red]")
    
    # Display results
    if results:
        display_generation_results(results)
        save_generation_results(results)
        
        # Key insights
        console.print("\n[bold]Key Insights:[/bold]")
        console.print("• Higher scores = better support for answer generation")
        console.print("• Faithfulness shows if answers are grounded in context")
        console.print("• Parent-Child may rank higher here than in retrieval metrics")
        console.print("• Consider both retrieval and generation scores for full picture")
    
    console.print("\n[green]✓ Generation evaluation complete![/green]")


if __name__ == "__main__":
    asyncio.run(main())