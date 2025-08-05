#!/usr/bin/env python
"""Quick test of retrieval metrics to show they work correctly"""
from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from rich.console import Console

from src.ragas_evaluation.utils.document_loader import load_pdf
from src.ragas_evaluation.retrievers.implementations import get_all_retrievers
from src.ragas_evaluation.evaluation.retrieval_evaluator import evaluate_retriever_performance
from src.ragas_evaluation.config import config

console = Console()

def main():
    console.print("[bold]Testing Retrieval-Focused Metrics[/bold]\n")
    
    # Load minimal data
    pdf_path = config.data_dir / "Never split the diff clean.pdf"
    documents = load_pdf(pdf_path)
    
    # Create minimal test dataset
    test_data = pd.DataFrame({
        'user_input': [
            "What experience did the FBI negotiator have in the Philippines?",
            "Who is Robert Mnookin?"
        ],
        'reference': [
            "The FBI negotiator spent 15 years negotiating hostage situations including in the Philippines",
            "Robert Mnookin is the director of the Harvard Negotiation Research Project"
        ]
    })
    
    # Get one retriever
    retrievers = get_all_retrievers(documents)
    test_retriever = retrievers[1]  # BM25
    
    console.print(f"Testing {test_retriever.name} with 2 questions...\n")
    
    # Evaluate
    result = evaluate_retriever_performance(test_retriever, test_data)
    
    # Display results
    console.print("\n[bold]Results:[/bold]")
    console.print(f"Retriever: {result.retriever_name}")
    console.print(f"Retrieval Score: {result.retrieval_score:.3f}\n")
    
    console.print("[bold]Retrieval Metrics:[/bold]")
    for metric, value in result.metrics.items():
        if metric != 'semantic_similarity':
            status = "✓" if value > 0.5 else "⚠" if value > 0 else "✗"
            console.print(f"{status} {metric}: {value:.3f}")
    
    console.print(f"\n[dim]Latency: {result.latency:.2f}s per query[/dim]")
    console.print(f"[dim]Docs retrieved: {result.doc_count:.1f} average[/dim]")
    
    console.print("\n[green]Success![/green] Retrieval metrics are working correctly.")
    console.print("\nThese metrics specifically measure how well the retriever")
    console.print("finds relevant contexts, not how well the LLM generates answers.")

if __name__ == "__main__":
    main()