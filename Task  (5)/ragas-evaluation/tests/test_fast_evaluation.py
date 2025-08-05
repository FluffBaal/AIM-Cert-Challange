#!/usr/bin/env python
"""Fast test evaluation - subset of data for quick testing"""
from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from pathlib import Path
from rich.console import Console
import time

from src.ragas_evaluation.utils.document_loader import load_pdf
from src.ragas_evaluation.retrievers.implementations import get_all_retrievers
from src.ragas_evaluation.evaluation.evaluator import evaluate_all_retrievers, display_results_table
from src.ragas_evaluation.config import config

console = Console()

def main():
    console.print("[bold cyan]RAGAS Fast Test Evaluation[/bold cyan]")
    console.print("Running with optimized settings for speed\n")
    
    start_time = time.time()
    
    # Load documents
    pdf_path = config.data_dir / "Never split the diff clean.pdf"
    documents = load_pdf(pdf_path)
    console.print(f"✓ Loaded {len(documents)} documents")
    
    # Load golden dataset - use only 3 questions
    golden_path = config.data_dir.parent / "the_gold" / "ragas_golden_dataset.csv"
    test_dataset = pd.read_csv(golden_path).head(3)
    console.print(f"✓ Using {len(test_dataset)} questions (subset for speed)")
    
    # Get only 3 fastest retrievers
    all_retrievers = get_all_retrievers(documents)
    fast_retrievers = [r for r in all_retrievers if r.name in ["BM25", "Parent Document", "Naive Retrieval"]]
    console.print(f"✓ Testing {len(fast_retrievers)} retrievers (fastest ones)")
    
    # Show optimizations
    console.print("\n[yellow]Optimizations applied:[/yellow]")
    console.print("- Timeout: 60s (was 180s)")
    console.print("- Max retries: 3 (was 15)")
    console.print("- Parallel workers: 4 (was 1)")
    console.print("- Questions: 3 (was 12)")
    console.print("- Retrievers: 3 (was 7)")
    console.print("- Metrics: 7 (all metrics included)")
    
    expected_time = (3 * 3 * 7 * 5) / 60  # questions × retrievers × metrics × 5s / 60
    console.print(f"\n[green]Expected time: ~{expected_time:.1f} minutes[/green]")
    console.print("[dim](Full evaluation would take 3-4 hours)[/dim]\n")
    
    # Run evaluation
    console.print("[bold]Starting evaluation...[/bold]")
    results = evaluate_all_retrievers(fast_retrievers, test_dataset)
    
    # Display results
    display_results_table(results)
    
    # Time taken
    elapsed = (time.time() - start_time) / 60
    console.print(f"\n[green]✓ Completed in {elapsed:.1f} minutes[/green]")
    
    # Extrapolate to full evaluation
    full_time_estimate = elapsed * (7/3) * (12/3)  # scale up for all retrievers and questions
    console.print(f"[dim]Full evaluation estimated: {full_time_estimate:.1f} minutes[/dim]")
    
    console.print("\n[bold]To run full evaluation with optimized settings:[/bold]")
    console.print("uv run python run_evaluation.py")

if __name__ == "__main__":
    main()