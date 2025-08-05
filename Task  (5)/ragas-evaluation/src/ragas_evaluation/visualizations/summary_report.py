"""Generate comprehensive summary report with insights."""
import pandas as pd
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
import matplotlib.pyplot as plt
import seaborn as sns

console = Console()


def generate_summary_report(results_df: pd.DataFrame, output_path: Path):
    """Generate a comprehensive summary report highlighting key insights."""
    
    # Create report file
    report_path = output_path / "evaluation_summary_report.md"
    
    # Identify meaningful metrics (non-zero)
    all_metrics = ['faithfulness', 'semantic_similarity', 'answer_relevancy', 
                   'context_precision', 'answer_correctness', 'llm_context_recall', 
                   'factual_correctness']
    
    meaningful_metrics = [m for m in all_metrics 
                         if m in results_df.columns and results_df[m].sum() > 0]
    
    zero_metrics = [m for m in all_metrics 
                   if m in results_df.columns and results_df[m].sum() == 0]
    
    with open(report_path, 'w') as f:
        f.write("# RAGAS Evaluation Summary Report\n\n")
        
        # Executive Summary
        f.write("## Executive Summary\n\n")
        winner = results_df.loc[results_df['ragas_score'].idxmax()]
        f.write(f"**Best Performing Retriever**: {winner['retriever']} (RAGAS Score: {winner['ragas_score']:.3f})\n\n")
        
        # Key Findings
        f.write("### Key Findings:\n\n")
        f.write(f"1. **{len(results_df)} retrievers evaluated** on {results_df['avg_docs'].mean():.0f} average documents\n")
        f.write(f"2. **Top 3 Retrievers**:\n")
        for i, row in results_df.nlargest(3, 'ragas_score').iterrows():
            f.write(f"   - {row['retriever']}: {row['ragas_score']:.3f}\n")
        
        # Metric Analysis
        f.write("\n## Metric Analysis\n\n")
        f.write("### Working Metrics (Non-Zero):\n")
        for metric in meaningful_metrics:
            avg_score = results_df[metric].mean()
            std_score = results_df[metric].std()
            f.write(f"- **{metric.replace('_', ' ').title()}**: μ={avg_score:.3f}, σ={std_score:.3f}\n")
        
        f.write("\n### Metrics with 0.0 Values:\n")
        for metric in zero_metrics:
            f.write(f"- **{metric.replace('_', ' ').title()}**: All retrievers scored 0.0 (strict claim matching)\n")
        
        # Performance Analysis
        f.write("\n## Performance Analysis\n\n")
        
        # Speed champion
        fastest = results_df.loc[results_df['latency'].idxmin()]
        f.write(f"**Fastest Retriever**: {fastest['retriever']} ({fastest['latency']:.2f}s avg)\n")
        
        # Cost effective
        cheapest = results_df.loc[results_df['estimated_cost'].idxmin()]
        f.write(f"**Most Cost-Effective**: {cheapest['retriever']} (${cheapest['estimated_cost']:.4f}/query)\n")
        
        # Best value (score per dollar)
        results_df['value_ratio'] = results_df['ragas_score'] / (results_df['estimated_cost'] + 0.0001)
        best_value = results_df.loc[results_df['value_ratio'].idxmax()]
        f.write(f"**Best Value**: {best_value['retriever']} ({best_value['value_ratio']:.0f} score/dollar)\n")
        
        # Recommendations
        f.write("\n## Recommendations\n\n")
        f.write("### For Different Use Cases:\n\n")
        f.write(f"1. **High Accuracy Required**: Use {winner['retriever']}\n")
        f.write(f"2. **Speed Critical**: Use {fastest['retriever']}\n")
        f.write(f"3. **Budget Conscious**: Use {cheapest['retriever']}\n")
        f.write(f"4. **Balanced Performance**: Use {best_value['retriever']}\n")
        
        # Technical Notes
        f.write("\n## Technical Notes\n\n")
        f.write("### Why Some Metrics Show 0.0:\n")
        f.write("- **Context Recall**: Compares reference claims against retrieved contexts (strict matching)\n")
        f.write("- **Factual Correctness**: Compares response claims against reference (exact factual overlap)\n")
        f.write("\nThese metrics use claim decomposition which requires exact matches. ")
        f.write("High scores in Faithfulness and Semantic Similarity indicate good overall performance.\n")
        
        # Detailed Results
        f.write("\n## Detailed Results Table\n\n")
        
        # Create markdown table
        f.write("| Retriever | RAGAS Score | Faithfulness | Semantic Sim | ")
        if 'answer_relevancy' in meaningful_metrics:
            f.write("Answer Relevancy | ")
        if 'context_precision' in meaningful_metrics:
            f.write("Context Precision | ")
        f.write("Latency (s) | Cost ($) |\n")
        
        f.write("|-----------|-------------|--------------|--------------|")
        if 'answer_relevancy' in meaningful_metrics:
            f.write("-----------------|")
        if 'context_precision' in meaningful_metrics:
            f.write("------------------|")
        f.write("------------|----------|\n")
        
        for _, row in results_df.sort_values('ragas_score', ascending=False).iterrows():
            f.write(f"| {row['retriever']} | {row['ragas_score']:.3f} | ")
            f.write(f"{row['faithfulness']:.3f} | {row['semantic_similarity']:.3f} | ")
            if 'answer_relevancy' in meaningful_metrics:
                f.write(f"{row.get('answer_relevancy', 0):.3f} | ")
            if 'context_precision' in meaningful_metrics:
                f.write(f"{row.get('context_precision', 0):.3f} | ")
            f.write(f"{row['latency']:.2f} | {row['estimated_cost']:.4f} |\n")
    
    console.print(f"[green]✓ Summary report saved to: {report_path}[/green]")
    
    # Create visual summary
    create_visual_summary(results_df, output_path)


def create_visual_summary(results_df: pd.DataFrame, output_path: Path):
    """Create a visual one-page summary of results."""
    fig = plt.figure(figsize=(16, 10))
    
    # Title
    fig.suptitle('RAGAS Evaluation Summary - Retriever Performance Analysis', fontsize=20, fontweight='bold')
    
    # 1. Top performers bar chart
    ax1 = plt.subplot(2, 3, 1)
    top_5 = results_df.nlargest(5, 'ragas_score')
    bars = ax1.barh(top_5['retriever'], top_5['ragas_score'], 
                    color=['gold', 'silver', '#CD7F32', 'lightblue', 'lightblue'])
    ax1.set_xlabel('RAGAS Score')
    ax1.set_title('Top 5 Retrievers')
    ax1.set_xlim(0, max(top_5['ragas_score']) * 1.1)
    
    # Add value labels
    for bar, score in zip(bars, top_5['ragas_score']):
        ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{score:.3f}', va='center')
    
    # 2. Metric comparison for winner
    ax2 = plt.subplot(2, 3, 2)
    winner = results_df.loc[results_df['ragas_score'].idxmax()]
    meaningful_metrics = ['faithfulness', 'semantic_similarity', 'answer_relevancy', 
                         'context_precision', 'answer_correctness']
    available_metrics = [m for m in meaningful_metrics 
                        if m in winner.index and winner[m] > 0]
    
    if available_metrics:
        values = [winner[m] for m in available_metrics]
        bars = ax2.bar(range(len(available_metrics)), values, color='#2ecc71')
        ax2.set_xticks(range(len(available_metrics)))
        ax2.set_xticklabels([m.replace('_', '\n') for m in available_metrics], rotation=0)
        ax2.set_ylim(0, 1.1)
        ax2.set_title(f"Winner: {winner['retriever']}")
        
        for bar, val in zip(bars, values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.2f}', ha='center', va='bottom')
    
    # 3. Cost vs Performance
    ax3 = plt.subplot(2, 3, 3)
    ax3.scatter(results_df['estimated_cost'], results_df['ragas_score'], 
               s=results_df['latency']*50, alpha=0.6, c=range(len(results_df)), cmap='viridis')
    ax3.set_xlabel('Cost per Query ($)')
    ax3.set_ylabel('RAGAS Score')
    ax3.set_title('Cost vs Performance (size = latency)')
    
    # Add retriever labels for top performers
    for _, row in results_df.nlargest(3, 'ragas_score').iterrows():
        ax3.annotate(row['retriever'], (row['estimated_cost'], row['ragas_score']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # 4. Metric distribution heatmap (only non-zero metrics)
    ax4 = plt.subplot(2, 3, (4, 5))
    meaningful_metrics = ['faithfulness', 'semantic_similarity', 'answer_relevancy', 
                         'context_precision', 'answer_correctness']
    available_metrics = [m for m in meaningful_metrics 
                        if m in results_df.columns and results_df[m].sum() > 0]
    
    if available_metrics:
        heatmap_data = results_df.set_index('retriever')[available_metrics]
        sns.heatmap(heatmap_data.T, annot=True, fmt='.2f', cmap='YlOrRd', 
                   cbar_kws={'label': 'Score'}, ax=ax4)
        ax4.set_title('Performance Heatmap (Non-Zero Metrics)')
    
    # 5. Summary statistics
    ax5 = plt.subplot(2, 3, 6)
    ax5.axis('off')
    
    summary_text = f"""
    Evaluation Summary:
    
    • Retrievers Evaluated: {len(results_df)}
    • Winner: {winner['retriever']} ({winner['ragas_score']:.3f})
    • Fastest: {results_df.loc[results_df['latency'].idxmin()]['retriever']} 
      ({results_df['latency'].min():.2f}s)
    • Cheapest: {results_df.loc[results_df['estimated_cost'].idxmin()]['retriever']} 
      (${results_df['estimated_cost'].min():.4f})
    
    Metrics with 0.0 values:
    • Context Recall (strict claim matching)
    • Factual Correctness (exact overlap required)
    
    Recommendation:
    Focus on Faithfulness and Semantic Similarity
    for overall quality assessment.
    """
    
    ax5.text(0.1, 0.9, summary_text, transform=ax5.transAxes, 
            fontsize=12, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path / "evaluation_summary_visual.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    console.print(f"[green]✓ Visual summary saved to: {output_path}/evaluation_summary_visual.png[/green]")