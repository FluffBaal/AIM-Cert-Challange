#!/usr/bin/env python
"""Run evaluation focused on retrieval metrics - the correct approach for comparing retrieval methods"""
from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from pathlib import Path
from rich.console import Console

from src.ragas_evaluation.utils.document_loader import load_pdf
from src.ragas_evaluation.retrievers.implementations import get_all_retrievers
from src.ragas_evaluation.evaluation.retrieval_evaluator import (
    evaluate_retriever_performance, 
    display_retrieval_results
)
from src.ragas_evaluation.config import config

console = Console()

def main():
    console.print("[bold cyan]RAGAS Retrieval-Focused Evaluation[/bold cyan]")
    console.print("Evaluating retrieval methods with appropriate metrics\n")
    
    # Explain the change
    console.print("[yellow]Why this is better:[/yellow]")
    console.print("• Previous evaluation mixed generation and retrieval metrics")
    console.print("• Generation metrics (Faithfulness, Answer Correctness) evaluate LLM quality")
    console.print("• Retrieval metrics (Context Precision, Recall) evaluate retriever quality")
    console.print("• For comparing retrievers, we need retrieval-specific metrics\n")
    
    # Load documents
    pdf_path = config.data_dir / "Never split the diff clean.pdf"
    documents = load_pdf(pdf_path)
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
    console.print("\n[bold]Retrieval Metrics Being Evaluated:[/bold]")
    console.print("1. [cyan]Context Precision[/cyan] - How precisely contexts match the query")
    console.print("2. [cyan]Context Recall[/cyan] - Coverage of relevant information")
    console.print("3. [cyan]Context Relevance[/cyan] - Overall relevance of retrieved contexts")
    console.print("4. [cyan]Context Entity Recall[/cyan] - Coverage of key entities")
    console.print("5. [cyan]Context Utilization[/cyan] - How well contexts are used")
    console.print("6. [dim]Semantic Similarity[/dim] - Reference metric for comparison\n")
    
    # Run evaluation
    console.print("[bold]Starting retrieval evaluation...[/bold]")
    results = []
    
    for retriever_info in retrievers:
        try:
            result = evaluate_retriever_performance(retriever_info, test_dataset)
            results.append(result)
        except Exception as e:
            console.print(f"[red]Failed to evaluate {retriever_info.name}: {e}[/red]")
    
    # Display results
    if results:
        display_retrieval_results(results)
        
        # Save results
        save_retrieval_results(results)
        
        # Key insights
        console.print("\n[bold]Key Insights:[/bold]")
        console.print("• Retrieval Score is weighted average of retrieval metrics")
        console.print("• Higher scores indicate better context retrieval")
        console.print("• Low scores may indicate poor chunking or indexing")
        console.print("• Consider both quality (score) and efficiency (latency)")
    
    console.print("\n[green]✓ Retrieval evaluation complete![/green]")


def save_retrieval_results(results):
    """Save retrieval results to CSV and generate visualizations"""
    data = []
    for result in results:
        row = {
            'retriever': result.retriever_name,
            'retrieval_score': result.retrieval_score,
            'latency': result.latency,
            'avg_docs': result.doc_count,
            'cost': result.estimated_cost,
            **result.metrics
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Save CSV
    output_path = config.output_dir / "retrieval_evaluation_results.csv"
    df.to_csv(output_path, index=False)
    console.print(f"\nResults saved to: {output_path}")
    
    # Save JSON
    json_path = config.output_dir / "retrieval_evaluation_results.json"
    df.to_json(json_path, orient='records', indent=2)
    console.print(f"Results saved to: {json_path}")
    
    # Generate visualizations
    console.print("\n[bold]Generating retrieval-specific visualizations...[/bold]")
    from src.ragas_evaluation.visualizations.retrieval_plots import create_all_retrieval_visualizations
    create_all_retrieval_visualizations(df, config.output_dir)
    console.print("[green]✓ Visualizations created in outputs/retrieval_diagrams/[/green]")
    
    # Create comparison with old results
    old_results_path = config.output_dir / "ragas_evaluation_results.csv"
    if old_results_path.exists():
        console.print("\n[yellow]Comparison with previous (mixed metrics) evaluation:[/yellow]")
        old_df = pd.read_csv(old_results_path)
        
        comparison = pd.merge(
            df[['retriever', 'retrieval_score']],
            old_df[['retriever', 'ragas_score']],
            on='retriever',
            how='inner'
        )
        
        comparison['score_diff'] = comparison['retrieval_score'] - comparison['ragas_score']
        comparison = comparison.sort_values('retrieval_score', ascending=False)
        
        console.print("\nRetriever | Retrieval Score | Old RAGAS Score | Difference")
        console.print("-" * 60)
        for _, row in comparison.iterrows():
            diff_color = "green" if row['score_diff'] > 0 else "red"
            console.print(
                f"{row['retriever']:<25} | "
                f"{row['retrieval_score']:.3f} | "
                f"{row['ragas_score']:.3f} | "
                f"[{diff_color}]{row['score_diff']:+.3f}[/{diff_color}]"
            )


if __name__ == "__main__":
    main()