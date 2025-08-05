"""Visualization module for retrieval evaluation results."""
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Any

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def create_retrieval_radar_chart(results_df: pd.DataFrame, output_path: Path):
    """Create radar chart for retrieval metrics comparison."""
    # Retrieval-specific metrics
    metrics = ['context_precision', 'context_recall', 'context_relevance', 
               'context_entity_recall', 'context_utilization']
    
    # Filter to available metrics
    available_metrics = [m for m in metrics if m in results_df.columns]
    
    fig = go.Figure()
    
    # Add traces for top 5 retrievers
    top_retrievers = results_df.nlargest(5, 'retrieval_score')
    
    for _, row in top_retrievers.iterrows():
        values = [row[metric] for metric in available_metrics]
        values.append(values[0])  # Complete the circle
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=available_metrics + [available_metrics[0]],
            fill='toself',
            name=row['retriever'],
            opacity=0.7
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickmode='linear',
                tick0=0,
                dtick=0.2
            )),
        showlegend=True,
        title="Retrieval Performance Comparison - Top 5 Methods",
        font=dict(size=14)
    )
    
    fig.write_html(str(output_path / "retrieval_radar_chart.html"))
    try:
        fig.write_image(str(output_path / "retrieval_radar_chart.png"))
    except:
        pass


def create_retrieval_heatmap(results_df: pd.DataFrame, output_path: Path):
    """Create heatmap specifically for retrieval metrics."""
    # Retrieval metrics
    metrics = ['retrieval_score', 'context_precision', 'context_recall', 
               'context_relevance', 'context_entity_recall', 'context_utilization']
    
    # Filter available metrics
    available_metrics = [m for m in metrics if m in results_df.columns]
    
    # Prepare data
    heatmap_data = results_df.set_index('retriever')[available_metrics]
    
    # Create figure with better size
    plt.figure(figsize=(12, 8))
    
    # Create mask for zero values (especially context_relevance)
    mask = heatmap_data == 0.0
    
    # Create heatmap
    ax = sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn', 
                     vmin=0, vmax=1, cbar_kws={'label': 'Score'},
                     linewidths=0.5, linecolor='gray')
    
    # Highlight zero values
    for i in range(len(heatmap_data.index)):
        for j in range(len(heatmap_data.columns)):
            if mask.iloc[i, j]:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True, 
                                         facecolor='lightgray', alpha=0.3,
                                         edgecolor='red', lw=2))
    
    plt.title('Retrieval Method Performance Heatmap\n(Red borders indicate metrics that need investigation)', 
              fontsize=14)
    plt.xlabel('Metrics', fontsize=12)
    plt.ylabel('Retrievers', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path / "retrieval_performance_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()


def create_retrieval_ranking_chart(results_df: pd.DataFrame, output_path: Path):
    """Create ranking chart for retrieval methods."""
    # Sort by retrieval score
    sorted_df = results_df.sort_values('retrieval_score', ascending=True)
    
    # Create color gradient
    colors = plt.cm.RdYlGn(sorted_df['retrieval_score'] / sorted_df['retrieval_score'].max())
    
    plt.figure(figsize=(12, 8))
    bars = plt.barh(sorted_df['retriever'], sorted_df['retrieval_score'], color=colors)
    
    # Add value labels
    for bar, score in zip(bars, sorted_df['retrieval_score']):
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left', va='center', fontweight='bold')
    
    # Highlight winner
    plt.gca().get_children()[-len(sorted_df)].set_edgecolor('gold')
    plt.gca().get_children()[-len(sorted_df)].set_linewidth(3)
    
    plt.xlabel('Retrieval Score', fontsize=12)
    plt.title('Retrieval Method Ranking\n(Based on Context Finding Quality)', fontsize=14)
    plt.xlim(0, 1)
    
    # Add grid
    plt.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / "retrieval_ranking.png", dpi=300, bbox_inches='tight')
    plt.close()


def create_metric_breakdown_chart(results_df: pd.DataFrame, output_path: Path):
    """Create stacked bar chart showing metric breakdown."""
    metrics = ['context_precision', 'context_recall', 'context_entity_recall', 'context_utilization']
    available_metrics = [m for m in metrics if m in results_df.columns and results_df[m].sum() > 0]
    
    if not available_metrics:
        return
    
    # Sort by retrieval score
    sorted_df = results_df.sort_values('retrieval_score', ascending=False)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create stacked bar chart
    bottom = pd.Series([0] * len(sorted_df), index=sorted_df.index)
    colors = plt.cm.Set3(range(len(available_metrics)))
    
    for i, metric in enumerate(available_metrics):
        values = sorted_df[metric]
        ax.bar(sorted_df['retriever'], values, bottom=bottom, 
               label=metric.replace('_', ' ').title(), color=colors[i], alpha=0.8)
        bottom += values
    
    # Add retrieval score line
    ax2 = ax.twinx()
    ax2.plot(sorted_df['retriever'], sorted_df['retrieval_score'], 
             'ko-', linewidth=2, markersize=8, label='Retrieval Score')
    ax2.set_ylabel('Retrieval Score', fontsize=12)
    ax2.set_ylim(0, 1)
    
    ax.set_xlabel('Retriever', fontsize=12)
    ax.set_ylabel('Metric Values (Stacked)', fontsize=12)
    ax.set_title('Retrieval Metrics Breakdown by Method', fontsize=14)
    ax.legend(loc='upper left', bbox_to_anchor=(0, 0.95))
    ax2.legend(loc='upper right', bbox_to_anchor=(1, 0.95))
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path / "retrieval_metric_breakdown.png", dpi=300, bbox_inches='tight')
    plt.close()


def create_efficiency_analysis(results_df: pd.DataFrame, output_path: Path):
    """Create efficiency analysis comparing performance vs resources."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Performance vs Latency
    scatter1 = ax1.scatter(results_df['latency'], results_df['retrieval_score'], 
                          s=results_df['avg_docs']*20, alpha=0.6, c=results_df['retrieval_score'],
                          cmap='RdYlGn', edgecolors='black', linewidth=1)
    
    # Add labels
    for _, row in results_df.iterrows():
        ax1.annotate(row['retriever'], (row['latency'], row['retrieval_score']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax1.set_xlabel('Latency (seconds)', fontsize=12)
    ax1.set_ylabel('Retrieval Score', fontsize=12)
    ax1.set_title('Performance vs Latency\n(Bubble size = Avg docs retrieved)', fontsize=14)
    ax1.grid(alpha=0.3)
    
    # Performance vs Document Count
    scatter2 = ax2.scatter(results_df['avg_docs'], results_df['retrieval_score'],
                          s=200, alpha=0.6, c=results_df['retrieval_score'],
                          cmap='RdYlGn', edgecolors='black', linewidth=1)
    
    # Add labels
    for _, row in results_df.iterrows():
        ax2.annotate(row['retriever'], (row['avg_docs'], row['retrieval_score']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax2.set_xlabel('Average Documents Retrieved', fontsize=12)
    ax2.set_ylabel('Retrieval Score', fontsize=12)
    ax2.set_title('Performance vs Document Count', fontsize=14)
    ax2.grid(alpha=0.3)
    
    # Add colorbars
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('Retrieval Score', fontsize=10)
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('Retrieval Score', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path / "retrieval_efficiency_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()


def create_retrieval_summary_table(results_df: pd.DataFrame, output_path: Path):
    """Create comprehensive summary table."""
    # Add rankings
    results_df['score_rank'] = results_df['retrieval_score'].rank(ascending=False)
    results_df['speed_rank'] = results_df['latency'].rank(ascending=True)
    results_df['efficiency_rank'] = (results_df['retrieval_score'] / results_df['latency']).rank(ascending=False)
    
    # Sort by retrieval score
    table_df = results_df.sort_values('retrieval_score', ascending=False)
    
    # Select columns for display
    display_cols = ['retriever', 'retrieval_score', 'context_precision', 'context_recall',
                   'context_entity_recall', 'context_utilization', 'latency', 'avg_docs',
                   'score_rank', 'speed_rank', 'efficiency_rank']
    
    # Filter available columns
    display_cols = [col for col in display_cols if col in table_df.columns]
    table_df = table_df[display_cols]
    
    # Save as CSV
    table_df.to_csv(output_path / "retrieval_summary_table.csv", index=False)
    
    # Create styled HTML
    html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th {{ background-color: #4CAF50; color: white; padding: 12px; text-align: left; }}
            td {{ padding: 8px; border-bottom: 1px solid #ddd; }}
            tr:hover {{ background-color: #f5f5f5; }}
            .winner {{ background-color: #FFD700; font-weight: bold; }}
            .zero {{ color: #888; background-color: #f9f9f9; }}
            h2 {{ color: #333; }}
            .note {{ color: #666; font-style: italic; margin-top: 10px; }}
        </style>
    </head>
    <body>
        <h2>Retrieval Method Evaluation Summary</h2>
        <table>
            <tr>
    """
    
    # Add headers
    for col in table_df.columns:
        header = col.replace('_', ' ').title()
        html += f"<th>{header}</th>"
    html += "</tr>"
    
    # Add rows
    for idx, row in table_df.iterrows():
        row_class = "winner" if idx == 0 else ""
        html += f'<tr class="{row_class}">'
        
        for col, val in row.items():
            if isinstance(val, float):
                cell_class = "zero" if val == 0.0 and 'relevance' in col else ""
                html += f'<td class="{cell_class}">{val:.3f}</td>'
            else:
                html += f'<td>{val}</td>'
        html += "</tr>"
    
    html += """
        </table>
        <p class="note">Note: Context Relevance showing 0.0 may indicate a metric configuration issue being investigated.</p>
        <p class="note">Rankings: Score (by retrieval quality), Speed (by latency), Efficiency (score/latency ratio)</p>
    </body>
    </html>
    """
    
    with open(output_path / "retrieval_summary_table.html", 'w') as f:
        f.write(html)
    
    return table_df


def create_all_retrieval_visualizations(results_df: pd.DataFrame, output_dir: Path):
    """Create all retrieval-specific visualizations."""
    output_path = output_dir / "retrieval_diagrams"
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Create all visualizations
    create_retrieval_radar_chart(results_df, output_path)
    create_retrieval_heatmap(results_df, output_path)
    create_retrieval_ranking_chart(results_df, output_path)
    create_metric_breakdown_chart(results_df, output_path)
    create_efficiency_analysis(results_df, output_path)
    
    # Create summary table
    tables_path = output_dir / "retrieval_tables"
    tables_path.mkdir(exist_ok=True, parents=True)
    summary_df = create_retrieval_summary_table(results_df, tables_path)
    
    # Generate retrieval-specific report
    generate_retrieval_report(results_df, output_dir)
    
    return summary_df


def generate_retrieval_report(results_df: pd.DataFrame, output_dir: Path):
    """Generate comprehensive retrieval evaluation report."""
    # Sort by score
    sorted_df = results_df.sort_values('retrieval_score', ascending=False)
    winner = sorted_df.iloc[0]
    
    report = f"""# Retrieval Method Evaluation Report

## Executive Summary

**Best Retrieval Method**: {winner['retriever']} (Score: {winner['retrieval_score']:.3f})

### Key Findings:

1. **{len(results_df)} retrieval methods evaluated** using retrieval-specific metrics
2. **Top 3 Methods**:
   - {sorted_df.iloc[0]['retriever']}: {sorted_df.iloc[0]['retrieval_score']:.3f}
   - {sorted_df.iloc[1]['retriever']}: {sorted_df.iloc[1]['retrieval_score']:.3f}
   - {sorted_df.iloc[2]['retriever']}: {sorted_df.iloc[2]['retrieval_score']:.3f}

## Metric Analysis

### Retrieval Metrics Performance:
"""
    
    # Add metric averages
    metrics = ['context_precision', 'context_recall', 'context_relevance', 
               'context_entity_recall', 'context_utilization']
    
    for metric in metrics:
        if metric in results_df.columns:
            mean_val = results_df[metric].mean()
            std_val = results_df[metric].std()
            if mean_val > 0:
                report += f"- **{metric.replace('_', ' ').title()}**: μ={mean_val:.3f}, σ={std_val:.3f}\n"
    
    # Note about context relevance if all zeros
    if 'context_relevance' in results_df.columns and results_df['context_relevance'].sum() == 0:
        report += "\n**Note**: Context Relevance shows 0.0 for all methods - investigating metric configuration.\n"
    
    report += f"""
## Performance Analysis

**Fastest Method**: {results_df.loc[results_df['latency'].idxmin()]['retriever']} ({results_df['latency'].min():.2f}s avg)
**Most Efficient**: {results_df.loc[(results_df['retrieval_score'] / results_df['latency']).idxmax()]['retriever']}
**Highest Precision**: {results_df.loc[results_df['context_precision'].idxmax()]['retriever']} ({results_df['context_precision'].max():.3f})
**Highest Recall**: {results_df.loc[results_df['context_recall'].idxmax()]['retriever']} ({results_df['context_recall'].max():.3f})

## Method-Specific Insights

### {winner['retriever']} (Winner)
- **Strengths**: """
    
    # Find winner's strengths
    winner_metrics = {}
    for metric in metrics:
        if metric in results_df.columns:
            winner_metrics[metric] = winner[metric]
    
    top_metrics = sorted(winner_metrics.items(), key=lambda x: x[1], reverse=True)[:3]
    for metric, value in top_metrics:
        if value > 0:
            report += f"\n  - {metric.replace('_', ' ').title()}: {value:.3f}"
    
    report += f"""
- **Performance**: {winner['latency']:.2f}s latency, {winner['avg_docs']:.1f} docs/query

## Recommendations

1. **For Production Use**: {winner['retriever']}
   - Best overall retrieval quality
   - Balanced performance across metrics

2. **For Speed-Critical Applications**: {results_df.loc[results_df['latency'].idxmin()]['retriever']}
   - Fastest response time
   - Good for real-time applications

3. **For High-Precision Needs**: {results_df.loc[results_df['context_precision'].idxmax()]['retriever']}
   - Most precise context selection
   - Best when accuracy is critical

## Technical Details

This evaluation used retrieval-specific metrics from RAGAS:
- Context Precision: How precisely contexts match the query
- Context Recall: Coverage of relevant information
- Context Entity Recall: Coverage of key entities
- Context Utilization: How well contexts support answering

These metrics specifically evaluate retrieval quality, not generation quality.
"""
    
    # Save report
    with open(output_dir / "retrieval_evaluation_report.md", 'w') as f:
        f.write(report)
    
    print(f"Report saved to: {output_dir / 'retrieval_evaluation_report.md'}")