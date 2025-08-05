"""Visualization module for creating plots and diagrams."""
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Any

from ..config import config
from .summary_report import generate_summary_report

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def create_radar_chart(results_df: pd.DataFrame, output_path: Path):
    """Create radar chart comparing retriever performance across metrics."""
    # Include all metrics but filter out those with all 0.0 values
    all_metrics = ['faithfulness', 'llm_context_recall', 'factual_correctness', 
                   'semantic_similarity', 'answer_relevancy', 'context_precision', 
                   'answer_correctness']
    
    # Filter out metrics where all values are 0.0
    metrics = [m for m in all_metrics if m in results_df.columns and results_df[m].sum() > 0]
    
    fig = go.Figure()
    
    for _, row in results_df.iterrows():
        values = [row[metric] for metric in metrics]
        values.append(values[0])  # Complete the circle
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics + [metrics[0]],
            fill='toself',
            name=row['retriever']
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Retriever Performance Comparison - Radar Chart"
    )
    
    fig.write_html(str(output_path / "radar_chart.html"))
    try:
        fig.write_image(str(output_path / "radar_chart.png"))
    except Exception as e:
        print(f"Warning: Could not export radar chart as PNG: {e}")


def create_performance_heatmap(results_df: pd.DataFrame, output_path: Path):
    """Create heatmap of all metrics for each retriever."""
    # Include all metrics available in the dataframe
    all_metrics = ['ragas_score', 'faithfulness', 'llm_context_recall', 
                   'factual_correctness', 'semantic_similarity', 'answer_relevancy', 
                   'context_precision', 'answer_correctness']
    
    # Filter to only include metrics that exist and have non-zero values
    metrics = [m for m in all_metrics if m in results_df.columns and results_df[m].sum() > 0]
    
    # Prepare data for heatmap
    heatmap_data = results_df.set_index('retriever')[metrics]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn', 
                vmin=0, vmax=1, cbar_kws={'label': 'Score'})
    plt.title('Retriever Performance Heatmap')
    plt.tight_layout()
    plt.savefig(output_path / "performance_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()


def create_cost_vs_performance_scatter(results_df: pd.DataFrame, output_path: Path):
    """Create scatter plot of cost vs performance."""
    fig = px.scatter(
        results_df,
        x='estimated_cost',
        y='ragas_score',
        size='latency',
        color='retriever',
        hover_data=['avg_docs'],
        labels={
            'estimated_cost': 'Estimated Cost ($)',
            'ragas_score': 'RAGAS Score',
            'latency': 'Latency (s)'
        },
        title='Cost vs Performance Analysis'
    )
    
    # Add quadrant lines
    fig.add_hline(y=results_df['ragas_score'].median(), line_dash="dash", 
                  line_color="gray", annotation_text="Median Performance")
    fig.add_vline(x=results_df['estimated_cost'].median(), line_dash="dash", 
                  line_color="gray", annotation_text="Median Cost")
    
    fig.write_html(str(output_path / "cost_vs_performance.html"))
    try:
        fig.write_image(str(output_path / "cost_vs_performance.png"))
    except Exception as e:
        print(f"Warning: Could not export cost vs performance chart as PNG: {e}")


def create_performance_bar_chart(results_df: pd.DataFrame, output_path: Path):
    """Create bar chart showing RAGAS scores with winner highlighted."""
    # Sort by RAGAS score
    sorted_df = results_df.sort_values('ragas_score', ascending=True)
    
    # Create color list (highlight winner)
    colors = ['lightblue'] * (len(sorted_df) - 1) + ['gold']
    
    plt.figure(figsize=(12, 8))
    bars = plt.barh(sorted_df['retriever'], sorted_df['ragas_score'], color=colors)
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left', va='center', fontweight='bold')
    
    plt.xlabel('RAGAS Score')
    plt.title('Retriever Performance Ranking')
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path / "performance_ranking.png", dpi=300, bbox_inches='tight')
    plt.close()


def create_latency_comparison(results_df: pd.DataFrame, output_path: Path):
    """Create latency comparison chart."""
    sorted_df = results_df.sort_values('latency')
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=sorted_df['retriever'],
        y=sorted_df['latency'],
        text=[f"{x:.2f}s" for x in sorted_df['latency']],
        textposition='auto',
        marker_color='lightgreen'
    ))
    
    fig.update_layout(
        title='Average Query Latency by Retriever',
        xaxis_title='Retriever',
        yaxis_title='Latency (seconds)',
        showlegend=False
    )
    
    fig.write_html(str(output_path / "latency_comparison.html"))
    try:
        fig.write_image(str(output_path / "latency_comparison.png"))
    except Exception as e:
        print(f"Warning: Could not export latency comparison as PNG: {e}")


def create_comprehensive_comparison_table(results_df: pd.DataFrame, output_path: Path):
    """Create comprehensive comparison table with rankings."""
    # Calculate rankings for all available metrics
    all_ranking_metrics = ['ragas_score', 'faithfulness', 'llm_context_recall', 
                          'factual_correctness', 'semantic_similarity', 'answer_relevancy', 
                          'context_precision', 'answer_correctness']
    
    # Only rank metrics that exist and have variation (not all 0.0)
    ranking_metrics = [m for m in all_ranking_metrics 
                      if m in results_df.columns and results_df[m].std() > 0]
    
    for metric in ranking_metrics:
        results_df[f'{metric}_rank'] = results_df[metric].rank(ascending=False)
    
    # Add inverse rankings for cost and latency (lower is better)
    results_df['cost_rank'] = results_df['estimated_cost'].rank(ascending=True)
    results_df['latency_rank'] = results_df['latency'].rank(ascending=True)
    
    # Calculate overall rank (average of all rankings)
    rank_columns = [col for col in results_df.columns if col.endswith('_rank')]
    results_df['overall_rank'] = results_df[rank_columns].mean(axis=1)
    
    # Sort by overall rank
    results_df = results_df.sort_values('overall_rank')
    
    # Create formatted table
    table_df = results_df[['retriever', 'ragas_score', 'faithfulness', 
                          'llm_context_recall', 'factual_correctness', 'semantic_similarity',
                          'latency', 'estimated_cost', 'avg_docs', 'overall_rank']]
    
    # Save as CSV
    table_df.to_csv(output_path / "comprehensive_comparison.csv", index=False)
    
    # Create HTML table with styling
    html = table_df.to_html(index=False, float_format=lambda x: f'{x:.3f}')
    
    # Add custom CSS for better styling
    styled_html = f"""
    <html>
    <head>
        <style>
            table {{
                border-collapse: collapse;
                width: 100%;
                font-family: Arial, sans-serif;
            }}
            th {{
                background-color: #4CAF50;
                color: white;
                padding: 12px;
                text-align: left;
            }}
            td {{
                padding: 8px;
                border-bottom: 1px solid #ddd;
            }}
            tr:hover {{
                background-color: #f5f5f5;
            }}
            tr:first-child {{
                background-color: #FFD700;
                font-weight: bold;
            }}
        </style>
    </head>
    <body>
        <h2>Comprehensive Retriever Comparison</h2>
        {html}
        <p><strong>Winner: {table_df.iloc[0]['retriever']}</strong> (Based on overall ranking)</p>
    </body>
    </html>
    """
    
    with open(output_path / "comprehensive_comparison.html", 'w') as f:
        f.write(styled_html)
    
    return table_df


def create_filtered_performance_heatmap(results_df: pd.DataFrame, output_path: Path):
    """Create comprehensive heatmap showing all metrics with annotations."""
    # Include ALL metrics
    all_metrics = ['ragas_score', 'faithfulness', 'llm_context_recall', 
                   'factual_correctness', 'semantic_similarity', 
                   'answer_relevancy', 'context_precision', 'answer_correctness']
    
    # Filter to only include metrics that exist
    available_metrics = [m for m in all_metrics if m in results_df.columns]
    
    # Prepare data for heatmap
    heatmap_data = results_df.set_index('retriever')[available_metrics]
    
    # Create custom colormap that makes 0.0 values distinct
    plt.figure(figsize=(14, 10))
    
    # Create mask for 0.0 values
    mask = heatmap_data == 0.0
    
    # Create heatmap with custom styling
    ax = sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn', 
                     vmin=0, vmax=1, cbar_kws={'label': 'Score'},
                     linewidths=0.5, linecolor='gray')
    
    # Highlight cells with 0.0 values
    for i in range(len(heatmap_data.index)):
        for j in range(len(heatmap_data.columns)):
            if mask.iloc[i, j]:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True, 
                                         facecolor='lightgray', alpha=0.3,
                                         edgecolor='red', lw=2))
    
    plt.title('Comprehensive Retriever Performance Heatmap\n(Red borders indicate 0.0 values - see UNDERSTANDING_ZERO_VALUES.md)', 
              fontsize=14)
    plt.xlabel('Metrics', fontsize=12)
    plt.ylabel('Retrievers', fontsize=12)
    
    # Add metric type annotations
    metric_types = {
        'ragas_score': 'Overall',
        'faithfulness': 'Quality',
        'llm_context_recall': 'Precision*',
        'factual_correctness': 'Precision*',
        'semantic_similarity': 'Quality',
        'answer_relevancy': 'Quality',
        'context_precision': 'Quality',
        'answer_correctness': 'Quality'
    }
    
    # Add secondary x-axis labels
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(range(len(available_metrics)))
    ax2.set_xticklabels([metric_types.get(m, 'New') for m in available_metrics], 
                        fontsize=10, style='italic')
    ax2.set_xlabel('Metric Type', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path / "comprehensive_performance_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()


def create_metric_distribution_plot(results_df: pd.DataFrame, output_path: Path):
    """Create box plots showing metric distribution across retrievers."""
    # Get non-zero metrics
    all_metrics = ['faithfulness', 'semantic_similarity', 'answer_relevancy', 
                   'context_precision', 'answer_correctness']
    available_metrics = [m for m in all_metrics if m in results_df.columns and results_df[m].sum() > 0]
    
    if not available_metrics:
        return
    
    fig, axes = plt.subplots(1, len(available_metrics), figsize=(4*len(available_metrics), 6))
    if len(available_metrics) == 1:
        axes = [axes]
    
    for idx, metric in enumerate(available_metrics):
        data = results_df[['retriever', metric]].copy()
        data = data[data[metric] > 0]  # Filter out zeros
        
        if len(data) > 0:
            sns.boxplot(data=data, y='retriever', x=metric, ax=axes[idx], orient='h')
            axes[idx].set_title(metric.replace('_', ' ').title())
            axes[idx].set_xlim(0, 1)
            axes[idx].set_ylabel('')
    
    plt.suptitle('Metric Distribution Across Retrievers', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path / "metric_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()


def create_enhanced_radar_chart(results_df: pd.DataFrame, output_path: Path):
    """Create enhanced radar chart with only meaningful metrics."""
    # Focus on metrics that work well
    meaningful_metrics = ['faithfulness', 'semantic_similarity', 'answer_relevancy', 
                         'context_precision', 'answer_correctness']
    
    # Filter to available non-zero metrics
    metrics = [m for m in meaningful_metrics 
              if m in results_df.columns and results_df[m].sum() > 0]
    
    if len(metrics) < 3:  # Need at least 3 metrics for radar chart
        print("Not enough non-zero metrics for radar chart")
        return
    
    fig = go.Figure()
    
    # Add traces for top 5 retrievers only (for clarity)
    top_retrievers = results_df.nlargest(5, 'ragas_score')
    
    for _, row in top_retrievers.iterrows():
        values = [row[metric] for metric in metrics]
        values.append(values[0])  # Complete the circle
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics + [metrics[0]],
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
        title="Top 5 Retrievers - Performance Comparison (Meaningful Metrics)",
        font=dict(size=14)
    )
    
    fig.write_html(str(output_path / "enhanced_radar_chart.html"))
    try:
        fig.write_image(str(output_path / "enhanced_radar_chart.png"))
    except:
        pass


def create_winner_analysis_plot(results_df: pd.DataFrame, output_path: Path):
    """Create detailed analysis plot for the winning retriever."""
    # Get winner
    winner = results_df.loc[results_df['ragas_score'].idxmax()]
    
    # Get meaningful metrics
    metric_categories = {
        'Quality Metrics': ['faithfulness', 'semantic_similarity', 'answer_correctness'],
        'Retrieval Metrics': ['context_precision', 'answer_relevancy'],
        'Performance Metrics': ['latency', 'estimated_cost']
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, (category, metrics) in enumerate(metric_categories.items()):
        available_metrics = [m for m in metrics if m in winner.index]
        if not available_metrics:
            continue
            
        values = [winner[m] for m in available_metrics]
        
        if category == 'Performance Metrics':
            # Normalize performance metrics for visualization
            if 'latency' in available_metrics:
                lat_idx = available_metrics.index('latency')
                values[lat_idx] = 1 - (values[lat_idx] / results_df['latency'].max())
            if 'estimated_cost' in available_metrics:
                cost_idx = available_metrics.index('estimated_cost')
                values[cost_idx] = 1 - (values[cost_idx] / results_df['estimated_cost'].max())
        
        bars = axes[idx].bar(available_metrics, values, color=['#2ecc71', '#3498db', '#e74c3c'])
        axes[idx].set_ylim(0, 1.1)
        axes[idx].set_title(f'{category}\n{winner["retriever"]}')
        axes[idx].set_xticklabels([m.replace('_', '\n') for m in available_metrics], rotation=0)
        
        # Add value labels
        for bar, val in zip(bars, values):
            axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                          f'{val:.3f}', ha='center', va='bottom')
    
    plt.suptitle(f'Winner Analysis: {winner["retriever"]} (RAGAS Score: {winner["ragas_score"]:.3f})', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path / "winner_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()


def create_all_visualizations(results_df: pd.DataFrame):
    """Create all visualizations and save to output directory."""
    output_path = config.output_dir / "diagrams"
    output_path.mkdir(exist_ok=True)
    
    # Create standard plots
    create_radar_chart(results_df, output_path)
    create_performance_heatmap(results_df, output_path)
    create_cost_vs_performance_scatter(results_df, output_path)
    create_performance_bar_chart(results_df, output_path)
    create_latency_comparison(results_df, output_path)
    
    # Create new enhanced visualizations
    create_filtered_performance_heatmap(results_df, output_path)
    create_metric_distribution_plot(results_df, output_path)
    create_enhanced_radar_chart(results_df, output_path)
    create_winner_analysis_plot(results_df, output_path)
    
    # Create comparison table
    tables_path = config.output_dir / "tables"
    tables_path.mkdir(exist_ok=True)
    comparison_df = create_comprehensive_comparison_table(results_df, tables_path)
    
    # Generate comprehensive summary report
    generate_summary_report(results_df, config.output_dir)
    
    return comparison_df