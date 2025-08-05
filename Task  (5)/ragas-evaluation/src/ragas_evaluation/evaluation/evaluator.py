"""Main evaluation module for running RAGAS evaluation on all retrievers."""
import time
from typing import List, Dict, Any

import pandas as pd
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from ragas import evaluate, RunConfig
from ragas.metrics import (
    LLMContextRecall,
    Faithfulness,
    FactualCorrectness,
    SemanticSimilarity,
    AnswerRelevancy,
    ContextPrecision,
    AnswerCorrectness,
)
import numpy as np
from rich.console import Console
from rich.progress import track
from rich.table import Table
from pathlib import Path

from ..config import config
from ..retrievers.base import RetrieverInfo

console = Console()


class EvaluationResult:
    """Container for evaluation results."""
    
    def __init__(self, retriever_name: str, metrics: Dict[str, float], 
                 latency: float, estimated_cost: float, doc_count: float):
        self.retriever_name = retriever_name
        self.metrics = metrics
        self.latency = latency
        self.estimated_cost = estimated_cost
        self.doc_count = doc_count
        self.ragas_score = metrics.get('ragas_score', 0.0)


def create_rag_chain(retriever: Any) -> RetrievalQA:
    """Create a RAG chain with the given retriever."""
    llm = ChatOpenAI(
        model=config.openai_model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        api_key=config.openai_api_key
    )
    
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    return chain


def evaluate_single_retriever(
    retriever_info: RetrieverInfo,
    test_dataset: pd.DataFrame
) -> EvaluationResult:
    """Evaluate a single retriever using RAGAS metrics."""
    console.print(f"\n[bold cyan]Evaluating {retriever_info.name}...[/bold cyan]")
    
    # Create RAG chain
    chain = create_rag_chain(retriever_info.retriever)
    
    # Prepare evaluation data
    # Handle different column names from RAGAS
    if 'question' in test_dataset.columns:
        questions = test_dataset['question'].tolist()
    elif 'user_input' in test_dataset.columns:
        questions = test_dataset['user_input'].tolist()
    else:
        raise ValueError("No question/user_input column found in test dataset")
    
    # Handle ground truth column
    if 'ground_truth' in test_dataset.columns:
        ground_truths = test_dataset['ground_truth'].tolist()
    elif 'reference' in test_dataset.columns:
        ground_truths = test_dataset['reference'].tolist()
    else:
        ground_truths = [''] * len(questions)  # Empty if no ground truth
    
    # Run predictions and measure performance
    predictions = []
    contexts = []
    total_latency = 0
    total_docs = 0
    
    for question in track(questions, description=f"Processing {retriever_info.name}"):
        start_time = time.time()
        
        try:
            result = chain.invoke({"query": question})
            predictions.append(result['result'])
            
            # Extract contexts
            source_docs = result.get('source_documents', [])
            contexts.append([doc.page_content for doc in source_docs])
            total_docs += len(source_docs)
            
        except Exception as e:
            console.print(f"[red]Error processing question: {e}[/red]")
            predictions.append("")
            contexts.append([])
        
        total_latency += time.time() - start_time
    
    # Calculate average metrics
    avg_latency = total_latency / len(questions)
    avg_docs = total_docs / len(questions)
    
    # Estimate cost (rough approximation)
    estimated_cost = estimate_cost(questions, predictions, contexts)
    
    # Prepare data for RAGAS evaluation
    from ragas import EvaluationDataset, SingleTurnSample
    from ragas.llms import LangchainLLMWrapper
    
    # Create evaluation samples
    samples = []
    for i in range(len(questions)):
        sample = SingleTurnSample(
            user_input=questions[i],
            response=predictions[i],
            retrieved_contexts=contexts[i],
            reference=ground_truths[i] if i < len(ground_truths) else None
        )
        samples.append(sample)
    
    # Create evaluation dataset
    eval_dataset = EvaluationDataset(samples=samples)
    
    # Initialize evaluator LLM and embeddings
    evaluator_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=config.openai_model,
            temperature=0,
            api_key=config.openai_api_key
        )
    )
    
    # Initialize embeddings for metrics that need them
    from langchain_openai import OpenAIEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            model=config.openai_embedding_model,
            api_key=config.openai_api_key
        )
    )
    
    # Initialize metrics with proper parameters
    # Keep existing metrics
    faithfulness_metric = Faithfulness()
    context_recall_metric = LLMContextRecall(llm=evaluator_llm)
    factual_correctness_metric = FactualCorrectness(llm=evaluator_llm)
    semantic_similarity_metric = SemanticSimilarity()
    
    # Add new metrics with REQUIRED parameters
    answer_relevancy_metric = AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)
    context_precision_metric = ContextPrecision(llm=evaluator_llm)
    answer_correctness_metric = AnswerCorrectness(llm=evaluator_llm, weights=[0.5, 0.5])  # Balance between similarity and correctness
    
    # Configure run settings for better parsing
    # Optimized settings to reduce evaluation time
    run_config = RunConfig(
        timeout=60,      # Reduced from 180 - most calls complete in <30s
        max_retries=3,   # Reduced from 15 - avoid excessive retries
        max_wait=60,     # Reduced from 180 - faster retry cycles
        max_workers=4    # Increased from 1 - parallel processing for speed
    )
    
    # Run RAGAS evaluation with better error handling
    try:
        results = evaluate(
            dataset=eval_dataset,
            metrics=[
                faithfulness_metric,
                context_recall_metric,
                factual_correctness_metric,
                semantic_similarity_metric,
                answer_relevancy_metric,
                context_precision_metric,
                answer_correctness_metric
            ],
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,  # Added embeddings
            run_config=run_config,
            raise_exceptions=True  # Changed to see parsing errors
        )
    except Exception as e:
        console.print(f"[yellow]Warning: Evaluation error: {str(e)[:200]}[/yellow]")
        
        # Try without problematic metrics if parsing failed
        if "parse" in str(e).lower() or "output" in str(e).lower():
            console.print("[yellow]Retrying with fallback metrics only...[/yellow]")
            results = evaluate(
                dataset=eval_dataset,
                metrics=[
                    faithfulness_metric,
                    semantic_similarity_metric
                ],
                llm=evaluator_llm,
                embeddings=evaluator_embeddings,
                run_config=run_config,
                raise_exceptions=False
            )
    
    # Extract metrics (calculate averages)
    try:
        # Try to get scores from results
        if hasattr(results, 'scores') and callable(results.scores):
            scores = results.scores()
        elif hasattr(results, 'scores') and isinstance(results.scores, dict):
            scores = results.scores
        elif hasattr(results, 'to_pandas'):
            # Convert to pandas and calculate means
            df = results.to_pandas()
            
            # Debug: Print sample data to understand the issue
            if retriever_info.name == "BM25":  # Only debug once
                console.print("\n[yellow]Debug - Sample evaluation data:[/yellow]")
                if len(df) > 0:
                    sample_idx = 0
                    console.print(f"Question: {questions[sample_idx][:100]}...")
                    console.print(f"Reference: {ground_truths[sample_idx][:100]}...")
                    console.print(f"Response: {predictions[sample_idx][:100]}...")
                    console.print(f"Retrieved contexts: {len(contexts[sample_idx])} docs")
                    if contexts[sample_idx]:
                        console.print(f"First context: {contexts[sample_idx][0][:100]}...")
            
            # Calculate scores, handling NaN values
            scores = {}
            all_metrics = ['faithfulness', 'llm_context_recall', 'factual_correctness', 
                          'semantic_similarity', 'answer_relevancy', 'context_precision', 
                          'answer_correctness']
            
            for metric_name in all_metrics:
                if metric_name in df:
                    # Get non-NaN values
                    values = df[metric_name].dropna()
                    if len(values) > 0:
                        scores[metric_name] = values.mean()
                    else:
                        scores[metric_name] = 0.0
                        console.print(f"[yellow]All values for {metric_name} are NaN (parsing failures)[/yellow]")
                else:
                    scores[metric_name] = 0.0
        else:
            # Fallback - try to access as dict
            scores = dict(results)
            
        metrics = {
            'faithfulness': scores.get('faithfulness', 0.0),
            'llm_context_recall': scores.get('llm_context_recall', 0.0),
            'factual_correctness': scores.get('factual_correctness', 0.0),
            'semantic_similarity': scores.get('semantic_similarity', 0.0),
        }
        
        # Handle cases where scores might be lists/arrays
        for key, value in metrics.items():
            if hasattr(value, 'mean'):
                metrics[key] = value.mean()
            elif isinstance(value, list):
                metrics[key] = sum(value) / len(value) if value else 0.0
                
    except Exception as e:
        console.print(f"[yellow]Warning: Error extracting metrics: {e}[/yellow]")
        metrics = {
            'faithfulness': 0.0,
            'llm_context_recall': 0.0,
            'factual_correctness': 0.0,
            'semantic_similarity': 0.0,
        }
    
    # Calculate RAGAS score as average
    metrics['ragas_score'] = sum(metrics.values()) / len(metrics)
    
    return EvaluationResult(
        retriever_name=retriever_info.name,
        metrics=metrics,
        latency=avg_latency,
        estimated_cost=estimated_cost,
        doc_count=avg_docs
    )


def estimate_cost(questions: List[str], answers: List[str], contexts: List[List[str]]) -> float:
    """Estimate API cost for the evaluation."""
    # Rough token estimation
    total_tokens = 0
    
    for q, a, ctx_list in zip(questions, answers, contexts):
        # Question tokens
        total_tokens += len(q.split()) * 1.3
        # Answer tokens
        total_tokens += len(a.split()) * 1.3
        # Context tokens
        for ctx in ctx_list:
            total_tokens += len(ctx.split()) * 1.3
    
    # GPT-4.1-mini pricing: $0.40 per 1M input, $1.60 per 1M output
    # Assuming 80% input, 20% output
    input_cost = (total_tokens * 0.8 / 1_000_000) * 0.40
    output_cost = (total_tokens * 0.2 / 1_000_000) * 1.60
    
    return input_cost + output_cost


def evaluate_all_retrievers(
    retrievers: List[RetrieverInfo],
    test_dataset: pd.DataFrame
) -> List[EvaluationResult]:
    """Evaluate all retrievers and return results."""
    results = []
    
    for retriever_info in retrievers:
        try:
            result = evaluate_single_retriever(retriever_info, test_dataset)
            results.append(result)
        except Exception as e:
            console.print(f"[red]Failed to evaluate {retriever_info.name}: {e}[/red]")
    
    return results


def display_results_table(results: List[EvaluationResult]):
    """Display evaluation results in a rich table."""
    table = Table(title="Retriever Evaluation Results (All Metrics)")
    
    # Add columns - include new metrics
    table.add_column("Retriever", style="cyan", no_wrap=True)
    table.add_column("RAGAS Score", justify="right", style="bold")
    table.add_column("Faithfulness", justify="right")
    table.add_column("Context Recall*", justify="right", style="dim")
    table.add_column("Factual Correct*", justify="right", style="dim")
    table.add_column("Semantic Sim", justify="right")
    table.add_column("Answer Relevancy", justify="right")
    table.add_column("Context Precision", justify="right")
    table.add_column("Answer Correct", justify="right")
    table.add_column("Latency (s)", justify="right")
    table.add_column("Cost ($)", justify="right")
    
    # Sort by RAGAS score
    sorted_results = sorted(results, key=lambda x: x.ragas_score, reverse=True)
    
    # Add rows
    for result in sorted_results:
        # Format cells - dim out 0.0 values
        def format_metric(value, is_zero_expected=False):
            if value == 0.0 and is_zero_expected:
                return "[dim]0.000[/dim]"
            return f"{value:.3f}"
        
        table.add_row(
            result.retriever_name,
            f"{result.ragas_score:.3f}",
            format_metric(result.metrics.get('faithfulness', 0)),
            format_metric(result.metrics.get('llm_context_recall', 0), is_zero_expected=True),
            format_metric(result.metrics.get('factual_correctness', 0), is_zero_expected=True),
            format_metric(result.metrics.get('semantic_similarity', 0)),
            format_metric(result.metrics.get('answer_relevancy', 0)),
            format_metric(result.metrics.get('context_precision', 0)),
            format_metric(result.metrics.get('answer_correctness', 0)),
            f"{result.latency:.2f}",
            f"{result.estimated_cost:.4f}"
        )
    
    console.print(table)
    console.print("\n[dim]* These metrics often show 0.0 due to strict claim matching (see METRICS_EXPLANATION.md)[/dim]")
    console.print("[dim]  See UNDERSTANDING_ZERO_VALUES.md for detailed explanation of when and why metrics return 0.0[/dim]")


def save_results(results: List[EvaluationResult], output_path: Path):
    """Save evaluation results to CSV and JSON."""
    # Convert to DataFrame
    data = []
    for result in results:
        row = {
            'retriever': result.retriever_name,
            'ragas_score': result.ragas_score,
            'latency': result.latency,
            'estimated_cost': result.estimated_cost,
            'avg_docs': result.doc_count,
            **result.metrics
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Save CSV
    csv_path = output_path.with_suffix('.csv')
    df.to_csv(csv_path, index=False)
    console.print(f"[green]Saved results to {csv_path}[/green]")
    
    # Save JSON
    json_path = output_path.with_suffix('.json')
    df.to_json(json_path, orient='records', indent=2)
    console.print(f"[green]Saved results to {json_path}[/green]")
    
    return df