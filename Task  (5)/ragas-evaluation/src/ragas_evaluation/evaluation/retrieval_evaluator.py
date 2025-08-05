"""Retrieval-focused evaluator using appropriate metrics for comparing retrieval methods"""
import time
from typing import List, Dict, Any

import pandas as pd
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from ragas import evaluate, RunConfig
from ragas.metrics import (
    # Retrieval-specific metrics
    ContextPrecision,
    ContextRecall,
    ContextRelevance,
    ContextEntityRecall,
    ContextUtilization,
    # Keep one generation metric for reference
    SemanticSimilarity,
)
from rich.console import Console
from rich.progress import track
from rich.table import Table
from pathlib import Path

from ..config import config
from ..retrievers.base import RetrieverInfo

console = Console()


class RetrievalEvaluationResult:
    """Container for retrieval evaluation results."""
    
    def __init__(self, retriever_name: str, metrics: Dict[str, float], 
                 latency: float, estimated_cost: float, doc_count: float):
        self.retriever_name = retriever_name
        self.metrics = metrics
        self.latency = latency
        self.estimated_cost = estimated_cost
        self.doc_count = doc_count
        self.retrieval_score = self._calculate_retrieval_score(metrics)
    
    def _calculate_retrieval_score(self, metrics: Dict[str, float]) -> float:
        """Calculate weighted retrieval score focusing on retrieval quality"""
        # Weight retrieval metrics higher
        weights = {
            'context_precision': 0.25,
            'context_recall': 0.25,
            'context_relevance': 0.25,
            'context_entity_recall': 0.15,
            'context_utilization': 0.10
        }
        
        score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in metrics and metrics[metric] > 0:
                score += metrics[metric] * weight
                total_weight += weight
        
        return score / total_weight if total_weight > 0 else 0.0


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


def evaluate_retriever_performance(
    retriever_info: RetrieverInfo,
    test_dataset: pd.DataFrame
) -> RetrievalEvaluationResult:
    """Evaluate a retriever using retrieval-specific metrics."""
    console.print(f"\n[bold cyan]Evaluating {retriever_info.name} (Retrieval Metrics)...[/bold cyan]")
    
    # Create RAG chain
    chain = create_rag_chain(retriever_info.retriever)
    
    # Prepare evaluation data
    questions = test_dataset['user_input'].tolist() if 'user_input' in test_dataset else test_dataset['question'].tolist()
    references = test_dataset['reference'].tolist() if 'reference' in test_dataset else [''] * len(questions)
    
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
    
    # Estimate cost
    estimated_cost = estimate_cost(questions, predictions, contexts)
    
    # Prepare data for RAGAS evaluation
    from ragas import EvaluationDataset, SingleTurnSample
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import OpenAIEmbeddings
    
    # Create evaluation samples
    samples = []
    for i in range(len(questions)):
        sample = SingleTurnSample(
            user_input=questions[i],
            response=predictions[i],
            retrieved_contexts=contexts[i],
            reference=references[i] if i < len(references) else None
        )
        samples.append(sample)
    
    # Create evaluation dataset
    eval_dataset = EvaluationDataset(samples=samples)
    
    # Initialize evaluator components
    evaluator_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=config.openai_model,
            temperature=0,
            api_key=config.openai_api_key
        )
    )
    
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            model=config.openai_embedding_model,
            api_key=config.openai_api_key
        )
    )
    
    # Initialize RETRIEVAL-SPECIFIC metrics
    context_precision = ContextPrecision(llm=evaluator_llm)
    context_recall = ContextRecall(llm=evaluator_llm)
    context_relevance = ContextRelevance(llm=evaluator_llm)  # No embeddings parameter
    context_entity_recall = ContextEntityRecall(llm=evaluator_llm)
    context_utilization = ContextUtilization(llm=evaluator_llm)
    semantic_similarity = SemanticSimilarity(embeddings=evaluator_embeddings)  # Needs embeddings
    
    # Configure optimized run settings
    run_config = RunConfig(
        timeout=60,
        max_retries=3,
        max_wait=60,
        max_workers=4
    )
    
    # Run evaluation
    try:
        results = evaluate(
            dataset=eval_dataset,
            metrics=[
                context_precision,
                context_recall,
                context_relevance,
                context_entity_recall,
                context_utilization,
                semantic_similarity
            ],
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
            run_config=run_config,
            raise_exceptions=False
        )
        
        # Extract metrics
        if hasattr(results, 'to_pandas'):
            df = results.to_pandas()
            metrics = {}
            
            # Map expected names to actual column names
            metric_mappings = {
                'context_precision': ['context_precision', 'nv_context_precision'],
                'context_recall': ['context_recall', 'nv_context_recall', 'llm_context_recall'],
                'context_relevance': ['context_relevance', 'nv_context_relevance'],
                'context_entity_recall': ['context_entity_recall', 'nv_context_entity_recall'],
                'context_utilization': ['context_utilization', 'nv_context_utilization'],
                'semantic_similarity': ['semantic_similarity', 'nv_semantic_similarity']
            }
            
            # Debug: print actual columns
            console.print(f"[dim]Debug - DataFrame columns: {df.columns.tolist()}[/dim]")
            
            for metric_name, possible_names in metric_mappings.items():
                found = False
                for col_name in possible_names:
                    if col_name in df:
                        values = df[col_name].dropna()
                        if len(values) > 0:
                            metrics[metric_name] = values.mean()
                        else:
                            metrics[metric_name] = 0.0
                        found = True
                        break
                
                if not found:
                    metrics[metric_name] = 0.0
        else:
            metrics = dict(results)
            
    except Exception as e:
        console.print(f"[yellow]Warning: Evaluation error: {e}[/yellow]")
        metrics = {
            'context_precision': 0.0,
            'context_recall': 0.0,
            'context_relevance': 0.0,
            'context_entity_recall': 0.0,
            'context_utilization': 0.0,
            'semantic_similarity': 0.0
        }
    
    return RetrievalEvaluationResult(
        retriever_name=retriever_info.name,
        metrics=metrics,
        latency=avg_latency,
        estimated_cost=estimated_cost,
        doc_count=avg_docs
    )


def estimate_cost(questions: List[str], answers: List[str], contexts: List[List[str]]) -> float:
    """Estimate API cost for the evaluation."""
    total_tokens = 0
    
    for q, a, ctx_list in zip(questions, answers, contexts):
        total_tokens += len(q.split()) * 1.3
        total_tokens += len(a.split()) * 1.3
        for ctx in ctx_list:
            total_tokens += len(ctx.split()) * 1.3
    
    # GPT-4o pricing
    input_cost = (total_tokens * 0.8 / 1_000_000) * 2.50
    output_cost = (total_tokens * 0.2 / 1_000_000) * 10.00
    
    return input_cost + output_cost


def display_retrieval_results(results: List[RetrievalEvaluationResult]):
    """Display retrieval evaluation results in a rich table."""
    table = Table(title="Retrieval Method Evaluation Results")
    
    # Add columns
    table.add_column("Retriever", style="cyan", no_wrap=True)
    table.add_column("Retrieval Score", justify="right", style="bold")
    table.add_column("Context Precision", justify="right")
    table.add_column("Context Recall", justify="right")
    table.add_column("Context Relevance", justify="right")
    table.add_column("Entity Recall", justify="right")
    table.add_column("Utilization", justify="right")
    table.add_column("Latency (s)", justify="right")
    table.add_column("Docs/Query", justify="right")
    
    # Sort by retrieval score
    sorted_results = sorted(results, key=lambda x: x.retrieval_score, reverse=True)
    
    # Add rows
    for result in sorted_results:
        table.add_row(
            result.retriever_name,
            f"{result.retrieval_score:.3f}",
            f"{result.metrics.get('context_precision', 0):.3f}",
            f"{result.metrics.get('context_recall', 0):.3f}",
            f"{result.metrics.get('context_relevance', 0):.3f}",
            f"{result.metrics.get('context_entity_recall', 0):.3f}",
            f"{result.metrics.get('context_utilization', 0):.3f}",
            f"{result.latency:.2f}",
            f"{result.doc_count:.1f}"
        )
    
    console.print(table)
    
    # Winner analysis
    winner = sorted_results[0]
    console.print(f"\n[bold green]Best Retrieval Method: {winner.retriever_name}[/bold green]")
    console.print(f"Retrieval Score: {winner.retrieval_score:.3f}")
    console.print("\nThis evaluation focuses on retrieval quality metrics, not generation quality.")
    console.print("See RAGAS documentation for metric details: https://docs.ragas.io/")