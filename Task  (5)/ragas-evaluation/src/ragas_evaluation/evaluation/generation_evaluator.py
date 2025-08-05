"""
Generation-focused evaluation using RAGAS metrics.
Evaluates how well retrievers support downstream answer generation.
"""
import time
from typing import List, Dict, Any
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    ResponseRelevancy,  # Correct name for answer relevancy
    Faithfulness,
    SemanticSimilarity  # For answer similarity
)
from langchain_openai import ChatOpenAI
from rich.console import Console
from rich.table import Table

from ..retrievers.base import RetrieverInfo
from ..config import config

console = Console()


async def evaluate_generation_performance(
    retriever_info: RetrieverInfo,
    test_data: pd.DataFrame
) -> Dict[str, Any]:
    """Evaluate retriever for generation tasks."""
    console.print(f"\nEvaluating {retriever_info.name} (Generation Metrics)...")
    
    retriever = retriever_info.retriever
    llm = ChatOpenAI(
        model=config.openai_model,
        temperature=0,
        api_key=config.openai_api_key
    )
    
    # Create QA chain
    from langchain.chains import RetrievalQA
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    
    # Generate answers
    questions = []
    answers = []
    contexts = []
    ground_truths = []
    latencies = []
    
    for _, row in test_data.iterrows():
        question = row['user_input']  # Changed from 'question'
        ground_truth = row['reference']  # Changed from 'ground_truth'
        
        # Time the retrieval + generation
        start_time = time.time()
        result = qa_chain.invoke({"query": question})
        latency = time.time() - start_time
        
        # Extract answer and contexts
        answer = result['result']
        source_docs = result['source_documents']
        context_list = [doc.page_content for doc in source_docs]
        
        questions.append(question)
        answers.append(answer)
        contexts.append(context_list)
        ground_truths.append(ground_truth)
        latencies.append(latency)
    
    # Create dataset for RAGAS
    eval_dataset = Dataset.from_dict({
        'question': questions,
        'answer': answers,
        'contexts': contexts,
        'ground_truth': ground_truths
    })
    
    # Evaluate with generation metrics
    metrics = [
        ResponseRelevancy(),
        Faithfulness(),
        SemanticSimilarity()
    ]
    
    results = evaluate(eval_dataset, metrics=metrics)
    
    # Calculate average latency and scores
    avg_latency = sum(latencies) / len(latencies)
    
    # Extract scores - handle different result formats
    if hasattr(results, 'to_pandas'):
        # Convert to pandas and get mean scores
        df = results.to_pandas()
        scores = {
            'retriever': retriever_info.name,
            'answer_relevancy': float(df['answer_relevancy'].mean()),
            'faithfulness': float(df['faithfulness'].mean()),
            'semantic_similarity': float(df['semantic_similarity'].mean()),
            'latency': avg_latency,
            'avg_contexts': sum(len(c) for c in contexts) / len(contexts)
        }
    else:
        # Try dict-like access
        scores = {
            'retriever': retriever_info.name,
            'answer_relevancy': float(results['answer_relevancy']),
            'faithfulness': float(results['faithfulness']),
            'semantic_similarity': float(results['semantic_similarity']),
            'latency': avg_latency,
            'avg_contexts': sum(len(c) for c in contexts) / len(contexts)
        }
    
    # Calculate overall generation score
    metric_scores = [
        scores['answer_relevancy'],
        scores['faithfulness'],
        scores['semantic_similarity']
    ]
    scores['generation_score'] = sum(metric_scores) / len(metric_scores)
    
    return scores


def display_generation_results(results: List[Dict[str, Any]]):
    """Display generation evaluation results."""
    # Sort by generation score
    results.sort(key=lambda x: x['generation_score'], reverse=True)
    
    # Create table
    table = Table(title="Generation Task Evaluation Results")
    table.add_column("Retriever", style="cyan")
    table.add_column("Generation Score", justify="right", style="green")
    table.add_column("Answer Relevancy", justify="right")
    table.add_column("Faithfulness", justify="right")
    table.add_column("Semantic Similarity", justify="right")
    table.add_column("Latency (s)", justify="right")
    
    for result in results:
        table.add_row(
            result['retriever'],
            f"{result['generation_score']:.3f}",
            f"{result['answer_relevancy']:.3f}",
            f"{result['faithfulness']:.3f}",
            f"{result['semantic_similarity']:.3f}",
            f"{result['latency']:.2f}"
        )
    
    console.print(table)
    
    # Winner announcement
    winner = results[0]
    console.print(f"\n[bold green]Best for Generation: {winner['retriever']} "
                  f"(Score: {winner['generation_score']:.3f})[/bold green]")


def save_generation_results(results: List[Dict[str, Any]]):
    """Save generation evaluation results with visualizations."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.graph_objects as go
    from pathlib import Path
    
    # Create output directories
    output_dir = config.output_dir / "generation_diagrams"
    output_dir.mkdir(exist_ok=True)
    
    # Save as CSV
    df = pd.DataFrame(results)
    output_path = config.output_dir / "generation_evaluation_results.csv"
    df.to_csv(output_path, index=False)
    console.print(f"\n✓ Results saved to {output_path}")
    
    # Sort by generation score
    df = df.sort_values('generation_score', ascending=False)
    
    # 1. Bar Chart - Generation Scores
    plt.figure(figsize=(10, 6))
    colors = ['green' if score > 0.9 else 'orange' if score > 0.85 else 'red' 
              for score in df['generation_score']]
    plt.bar(df['retriever'], df['generation_score'], color=colors)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Generation Score')
    plt.title('Generation Task Performance by Retriever')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_dir / 'generation_scores_bar.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Radar Chart - Metric Comparison
    fig = go.Figure()
    
    metrics = ['answer_relevancy', 'faithfulness', 'semantic_similarity']
    
    for _, row in df.iterrows():
        values = [row[m] for m in metrics]
        values.append(values[0])  # Complete the circle
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics + [metrics[0]],
            fill='toself',
            name=row['retriever'],
            opacity=0.6
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Generation Metrics Comparison"
    )
    
    fig.write_html(output_dir / 'generation_radar_chart.html')
    
    # 3. Heatmap - All Metrics
    plt.figure(figsize=(8, 10))
    
    # Prepare data for heatmap
    heatmap_data = df.set_index('retriever')[metrics].T
    
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn', 
                vmin=0, vmax=1, cbar_kws={'label': 'Score'})
    plt.title('Generation Metrics Heatmap')
    plt.tight_layout()
    plt.savefig(output_dir / 'generation_metrics_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Scatter Plot - Faithfulness vs Relevancy
    plt.figure(figsize=(10, 8))
    
    for _, row in df.iterrows():
        plt.scatter(row['faithfulness'], row['answer_relevancy'], 
                   s=200, alpha=0.7)
        plt.annotate(row['retriever'], 
                    (row['faithfulness'], row['answer_relevancy']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    plt.xlabel('Faithfulness')
    plt.ylabel('Answer Relevancy')
    plt.title('Faithfulness vs Answer Relevancy')
    plt.xlim(0.7, 1.0)
    plt.ylim(0.8, 1.0)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'faithfulness_vs_relevancy.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Latency Comparison
    plt.figure(figsize=(10, 6))
    colors = ['green' if lat < 6 else 'orange' if lat < 8 else 'red' 
              for lat in df['latency']]
    plt.bar(df['retriever'], df['latency'], color=colors)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Latency (seconds)')
    plt.title('Generation Latency by Retriever')
    plt.tight_layout()
    plt.savefig(output_dir / 'generation_latency.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    console.print(f"✓ Visualizations saved to {output_dir}/")
    
    # 6. Create summary table
    summary_path = config.output_dir / "generation_tables"
    summary_path.mkdir(exist_ok=True)
    
    summary_df = df[['retriever', 'generation_score', 'answer_relevancy', 
                     'faithfulness', 'semantic_similarity', 'latency']].round(3)
    summary_df.to_csv(summary_path / 'generation_summary_table.csv', index=False)
    summary_df.to_html(summary_path / 'generation_summary_table.html', index=False)
    
    # Create comparison with retrieval results
    retrieval_path = config.output_dir / "retrieval_evaluation_results.csv"
    if retrieval_path.exists():
        retrieval_df = pd.read_csv(retrieval_path)
        
        # Merge results
        comparison = pd.merge(
            df[['retriever', 'generation_score']],
            retrieval_df[['retriever', 'retrieval_score']],
            on='retriever'
        )
        
        # Add ranking columns
        comparison['generation_rank'] = comparison['generation_score'].rank(ascending=False)
        comparison['retrieval_rank'] = comparison['retrieval_score'].rank(ascending=False)
        comparison['rank_change'] = comparison['retrieval_rank'] - comparison['generation_rank']
        
        # Save comparison
        comparison_path = config.output_dir / "retrieval_vs_generation_comparison.csv"
        comparison.to_csv(comparison_path, index=False)
        
        # Show methods that improved
        improved = comparison[comparison['rank_change'] > 0].sort_values('rank_change', ascending=False)
        if not improved.empty:
            console.print("\n[bold]Methods that perform better for generation:[/bold]")
            for _, row in improved.iterrows():
                console.print(f"• {row['retriever']}: ↑{int(row['rank_change'])} positions")
    
    # Generate markdown report
    _create_generation_report(results)


def _create_generation_report(results: List[Dict[str, Any]]):
    """Create a detailed markdown report for generation evaluation"""
    df = pd.DataFrame(results)
    df = df.sort_values('generation_score', ascending=False)
    
    report = []
    report.append("# Generation Task Evaluation Report\n")
    report.append("## Executive Summary\n")
    report.append(f"**Best Method for Generation**: {df.iloc[0]['retriever']} (Score: {df.iloc[0]['generation_score']:.3f})\n")
    
    report.append("### Key Findings:\n")
    report.append(f"1. **{len(results)} retrieval methods evaluated** for answer generation quality")
    report.append("2. **Top 3 Methods**:")
    for i, row in df.head(3).iterrows():
        report.append(f"   - {row['retriever']}: {row['generation_score']:.3f}")
    report.append("")
    
    # Metric Analysis
    report.append("## Metric Analysis\n")
    report.append("### Generation Metrics Performance:")
    for metric in ['answer_relevancy', 'faithfulness', 'semantic_similarity']:
        mean_val = df[metric].mean()
        std_val = df[metric].std()
        report.append(f"- **{metric.replace('_', ' ').title()}**: μ={mean_val:.3f}, σ={std_val:.3f}")
    report.append("")
    
    # Performance Analysis
    report.append("## Performance Analysis\n")
    fastest = df.loc[df['latency'].idxmin()]
    most_faithful = df.loc[df['faithfulness'].idxmax()]
    most_relevant = df.loc[df['answer_relevancy'].idxmax()]
    
    report.append(f"**Fastest Generation**: {fastest['retriever']} ({fastest['latency']:.2f}s avg)")
    report.append(f"**Most Faithful**: {most_faithful['retriever']} ({most_faithful['faithfulness']:.3f})")
    report.append(f"**Most Relevant**: {most_relevant['retriever']} ({most_relevant['answer_relevancy']:.3f})")
    report.append("")
    
    # Method-Specific Insights
    report.append("## Method-Specific Insights\n")
    for _, row in df.iterrows():
        report.append(f"### {row['retriever']}")
        report.append("- **Scores**:")
        report.append(f"  - Answer Relevancy: {row['answer_relevancy']:.3f}")
        report.append(f"  - Faithfulness: {row['faithfulness']:.3f}")
        report.append(f"  - Semantic Similarity: {row['semantic_similarity']:.3f}")
        report.append(f"- **Performance**: {row['latency']:.2f}s latency, {row['avg_contexts']:.1f} contexts/query")
        report.append("")
    
    # Recommendations
    report.append("## Recommendations\n")
    report.append("1. **For High-Quality Answers**: " + df.iloc[0]['retriever'])
    report.append("   - Best overall generation quality")
    report.append("   - Balanced performance across metrics\n")
    
    report.append("2. **For Trusted Answers**: " + most_faithful['retriever'])
    report.append("   - Highest faithfulness to retrieved context")
    report.append("   - Best when accuracy is critical\n")
    
    report.append("3. **For Fast Response**: " + fastest['retriever'])
    report.append("   - Fastest generation time")
    report.append("   - Good for real-time applications\n")
    
    # Technical Details
    report.append("## Technical Details\n")
    report.append("This evaluation used generation-specific metrics from RAGAS:")
    report.append("- Answer Relevancy: How relevant the answer is to the question")
    report.append("- Faithfulness: Whether the answer is grounded in retrieved context")
    report.append("- Semantic Similarity: How similar the answer is to ground truth\n")
    report.append("These metrics evaluate the complete RAG pipeline, not just retrieval quality.")
    
    # Save report
    report_path = config.output_dir / "generation_evaluation_report.md"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    
    console.print(f"✓ Report saved to {report_path}")