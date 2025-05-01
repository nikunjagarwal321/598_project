#!/usr/bin/env python
"""
Retrieval benchmarking suite for TriviaQA dataset.

Dependencies:
  pip install rank-bm25 sentence-transformers faiss-cpu tqdm matplotlib seaborn pandas nltk scikit-learn memory_profiler
"""

import json
import os
import gzip
import time
import resource
import re
import psutil
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize

# Retrieval libs
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss

# Chunker implementations
from chunkers import FixedChunker, OverlappingChunker, SemanticChunker

# Retriever implementations
from retrievers.bm25_retriever import BM25Retriever
from retrievers.dense_retriever import DPRRetriever
from retrievers.colbert_retriever import ColBERTRetriever
from retrievers.hybrid_retriever import HybridRetriever

# Download NLTK resources (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Advanced evaluation utilities
def exact_match(gold: str, passages: List[str]) -> int:
    """Check if the gold answer appears in any retrieved passage."""
    g = gold.lower()
    return int(any(g in p.lower() for p in passages))

def calculate_precision(gold: str, passages: List[str]) -> float:
    """Calculate precision of retrieved passages."""
    g = gold.lower()
    correct = sum(1 for p in passages if g in p.lower())
    return correct / len(passages) if passages else 0

def calculate_recall(gold: str, passages: List[str], top_k: int) -> float:
    """Calculate recall for retrieved passages."""
    # Simplified recall for this scenario
    g = gold.lower()
    return 1.0 if any(g in p.lower() for p in passages) else 0.0

def calculate_f1(precision: float, recall: float) -> float:
    """Calculate F1 score from precision and recall."""
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def reciprocal_rank(gold: str, passages: List[str]) -> float:
    """Calculate Mean Reciprocal Rank (where first correct answer appears)."""
    g = gold.lower()
    for i, passage in enumerate(passages):
        if g in passage.lower():
            return 1.0 / (i + 1)
    return 0.0

def hit_rate_at_k(gold: str, passages: List[str], k: int) -> int:
    """Check if the answer is found within top k results."""
    if k > len(passages):
        k = len(passages)
    g = gold.lower()
    return int(any(g in p.lower() for p in passages[:k]))

def semantic_similarity(gold: str, passages: List[str], model) -> float:
    """Calculate semantic similarity between gold answer and passages."""
    if not passages:
        return 0.0
        
    # Encode gold answer and passages
    gold_embedding = model.encode([gold], convert_to_numpy=True)
    passage_embeddings = model.encode(passages, convert_to_numpy=True)
    
    # Calculate similarities
    similarities = cosine_similarity(gold_embedding, passage_embeddings)[0]
    
    # Return max similarity score
    return float(np.max(similarities))

def analyze_chunks(chunks: List[str]) -> Dict[str, float]:
    """Analyze quality metrics of chunks."""
    if not chunks:
        return {
            "avg_chunk_length": 0,
            "sentence_completeness": 0,
            "chunk_length_variance": 0
        }
    
    # Calculate average chunk length in words
    chunk_lengths = [len(chunk.split()) for chunk in chunks]
    avg_length = np.mean(chunk_lengths)
    length_variance = np.var(chunk_lengths)
    
    # Check sentence completeness (how many chunks end with sentence-ending punctuation)
    sentence_endings = sum(1 for chunk in chunks if re.search(r'[.!?]$', chunk.strip()))
    sentence_completeness = sentence_endings / len(chunks) if chunks else 0
    
    return {
        "avg_chunk_length": avg_length,
        "sentence_completeness": sentence_completeness,
        "chunk_length_variance": length_variance
    }

def measure_memory_usage():
    """Measure current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # Convert to MB

class Timer:
    """Simple timer context manager."""
    def __enter__(self):
        self.start = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.ms = (self.end - self.start) * 1000

def load_triviaqa(
    dataset_path: str,
    max_docs: int = 300,
    max_qa_pairs: int = 100
) -> Tuple[Dict[str, str], List[Dict]]:
    """
    Load TriviaQA dataset and return documents and QA pairs.
    
    Args:
        dataset_path: Path to the TriviaQA JSON file
        max_docs: Maximum number of documents to load
        max_qa_pairs: Maximum number of QA pairs to load
        
    Returns:
        Tuple of (documents dictionary, list of QA pairs)
    """
    print(f"Loading TriviaQA from {dataset_path}")
    
    root = Path(dataset_path).parent
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)["Data"]

    docs = {}
    qa_pairs = []
    
    def read_evidence(file_path: Path) -> str:
        if not file_path.exists():
            return ""
        try:
            if file_path.suffix == ".gz":
                with gzip.open(file_path, "rt", encoding="utf-8", errors="ignore") as g:
                    return g.read()
            return file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return ""

    def add_doc(doc_info, prefix: str, j: int):
        if len(docs) >= max_docs:
            return
        doc_id = f"{prefix}_{j}"
        txt = ""
        if doc_info.get("Filename"):
            txt = read_evidence(root / doc_info["Filename"])
        # Fallbacks
        txt = txt or doc_info.get("Snippet", "") or doc_info.get("Title", "") or doc_info.get("Url", "")
        docs[doc_id] = txt

    for i, item in enumerate(data):
        if i >= max_qa_pairs:
            break

        # QA
        question = item["Question"]
        aliases = item["Answer"].get("NormalizedAliases") or []
        gold = aliases[0] if aliases else item["Answer"]["NormalizedValue"]
        
        # Add question complexity estimation (word count as proxy)
        complexity = len(question.split())
        
        qa_pairs.append({
            "question": question, 
            "answer": gold,
            "complexity": complexity
        })

        # Wiki evidence
        for j, d in enumerate(item.get("EntityPages", [])):
            add_doc(d, f"wiki_{d.get('Title','wiki')}", j)
        # Web search evidence
        for j, d in enumerate(item.get("SearchResults", [])):
            add_doc(d, "web", j)

    # Drop empty documents
    docs = {k: v for k, v in docs.items() if v.strip()}
    print(f"Loaded {len(docs)} non-empty documents and {len(qa_pairs)} QA pairs")
    return docs, qa_pairs

def benchmark_retrieval(retriever, query: str, gold_answer: str, model, top_k: int) -> Dict[str, Any]:
    """Benchmark a single retrieval operation with comprehensive metrics."""
    # Measure memory before retrieval
    mem_before = measure_memory_usage()
    
    # Measure retrieval time
    start = time.time()
    results = retriever.retrieve(query, top_k=top_k)
    end = time.time()
    latency = end - start
    
    # Measure memory after retrieval
    mem_after = measure_memory_usage()
    mem_usage = mem_after - mem_before
    
    # Calculate basic metrics
    em = exact_match(gold_answer, results)
    precision = calculate_precision(gold_answer, results)
    recall = calculate_recall(gold_answer, results, top_k)
    f1 = calculate_f1(precision, recall)
    
    # Calculate advanced metrics
    mrr = reciprocal_rank(gold_answer, results)
    hit_rate_1 = hit_rate_at_k(gold_answer, results, 1)
    hit_rate_3 = hit_rate_at_k(gold_answer, results, 3)
    semantic_sim = semantic_similarity(gold_answer, results, model)
    
    return {
        "results": results,
        "latency": latency,
        "em": em,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mrr": mrr,
        "hit_rate_1": hit_rate_1, 
        "hit_rate_3": hit_rate_3,
        "semantic_similarity": semantic_sim,
        "memory_usage_mb": mem_usage
    }

def run_all_strategies(docs: Dict[str, str], qa_pairs: List[Dict], top_k: int = 5) -> pd.DataFrame:
    """
    Run a comprehensive evaluation of all chunking and retrieval strategies.
    
    Returns:
        DataFrame with all evaluation results
    """
    # Define all strategies
    chunkers = {
        "fixed_small": FixedChunker(chunk_size=64, drop_last=False),
        "fixed_medium": FixedChunker(chunk_size=128, drop_last=False),
        "fixed_large": FixedChunker(chunk_size=256, drop_last=False),
        "overlap_small": OverlappingChunker(chunk_size=96, overlap=32, drop_last=False),
        "overlap_medium": OverlappingChunker(chunk_size=192, overlap=64, drop_last=False),
        "overlap_large": OverlappingChunker(chunk_size=320, overlap=64, drop_last=False),
        "semantic_small": SemanticChunker(chunk_char_limit=500),
        "semantic_medium": SemanticChunker(chunk_char_limit=1000),
        "semantic_large": SemanticChunker(chunk_char_limit=2000),
    }
    
    retrievers = {
        "bm25": BM25Retriever,
        "dpr": DPRRetriever,
        "colbert": ColBERTRetriever,
        "hybrid": HybridRetriever
    }
    
    # Initialize embedding model for semantic similarity calculations
    semantic_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    
    doc_values = list(docs.values())
    results = []
    chunking_stats = []
    
    # For progress tracking
    total_combinations = len(chunkers) * len(retrievers)
    combination_count = 0
    
    # Sample a smaller set of QA pairs for evaluation to keep runtime manageable
    eval_qa_pairs = qa_pairs[:20]  # Adjust based on runtime constraints
    
    for chunker_name, chunker in chunkers.items():
        # Create chunks for this chunker
        print(f"\nProcessing chunker: {chunker_name}")
        chunks = []
        chunk_time_start = time.time()
        
        for doc in tqdm(doc_values, desc="Chunking documents"):
            doc_chunks = chunker.chunk(doc)
            chunks.extend(doc_chunks)
            
        chunk_time = time.time() - chunk_time_start
        
        # Analyze chunk quality
        chunk_metrics = analyze_chunks(chunks)
        chunking_stats.append({
            "chunker": chunker_name,
            "chunks_count": len(chunks),
            "chunking_time": chunk_time,
            "avg_chunk_length": chunk_metrics["avg_chunk_length"],
            "sentence_completeness": chunk_metrics["sentence_completeness"],
            "chunk_length_variance": chunk_metrics["chunk_length_variance"]
        })
            
        print(f"Created {len(chunks)} chunks with {chunker_name} in {chunk_time:.2f}s")
        print(f"Average chunk length: {chunk_metrics['avg_chunk_length']:.1f} words")
        
        for retriever_name, retriever_class in retrievers.items():
            combination_count += 1
            print(f"Running combination {combination_count}/{total_combinations}: {chunker_name} + {retriever_name}")
            
            # Initialize retriever with chunks
            try:
                # Measure index building time and memory
                index_mem_before = measure_memory_usage()
                index_time_start = time.time()
                retriever = retriever_class(chunks)
                index_time = time.time() - index_time_start
                index_mem_usage = measure_memory_usage() - index_mem_before
                
                print(f"Index built in {index_time:.2f}s, used {index_mem_usage:.1f}MB memory")
                
                # Evaluate on QA pairs
                qa_results = []
                query_times = []
                
                for qa in tqdm(eval_qa_pairs, desc=f"Evaluating {retriever_name}"):
                    query, gold = qa["question"], qa["answer"]
                    complexity = qa.get("complexity", len(query.split()))  # Word count as fallback
                    
                    result = benchmark_retrieval(retriever, query, gold, semantic_model, top_k)
                    result["query_complexity"] = complexity
                    qa_results.append(result)
                    query_times.append(result["latency"])
                
                # Calculate aggregate metrics
                avg_latency = sum(r["latency"] for r in qa_results) / len(qa_results)
                latency_variance = np.var([r["latency"] for r in qa_results])
                avg_em = sum(r["em"] for r in qa_results) / len(qa_results)
                avg_precision = sum(r["precision"] for r in qa_results) / len(qa_results)
                avg_recall = sum(r["recall"] for r in qa_results) / len(qa_results)
                avg_f1 = sum(r["f1"] for r in qa_results) / len(qa_results)
                avg_mrr = sum(r["mrr"] for r in qa_results) / len(qa_results)
                avg_hit_1 = sum(r["hit_rate_1"] for r in qa_results) / len(qa_results)
                avg_hit_3 = sum(r["hit_rate_3"] for r in qa_results) / len(qa_results)
                avg_semantic_sim = sum(r["semantic_similarity"] for r in qa_results) / len(qa_results)
                avg_memory = sum(r["memory_usage_mb"] for r in qa_results) / len(qa_results)
                
                # Calculate correlation between query complexity and performance
                complexities = [r["query_complexity"] for r in qa_results]
                em_scores = [r["em"] for r in qa_results]
                complexity_em_corr = np.corrcoef(complexities, em_scores)[0, 1] if len(qa_results) > 1 else 0
                
                results.append({
                    "chunker": chunker_name,
                    "retriever": retriever_name,
                    "chunks_count": len(chunks),
                    "avg_chunk_length": chunk_metrics["avg_chunk_length"],
                    "sentence_completeness": chunk_metrics["sentence_completeness"],
                    "chunking_time": chunk_time,
                    "indexing_time": index_time,
                    "indexing_memory_mb": index_mem_usage,
                    "avg_query_time": avg_latency,
                    "query_time_variance": latency_variance,
                    "em_score": avg_em,
                    "precision": avg_precision,
                    "recall": avg_recall,
                    "f1_score": avg_f1,
                    "mrr": avg_mrr,
                    "hit_rate_1": avg_hit_1,
                    "hit_rate_3": avg_hit_3,
                    "semantic_similarity": avg_semantic_sim,
                    "query_memory_mb": avg_memory,
                    "complexity_correlation": complexity_em_corr
                })
                
                # Print summary statistics
                print(f"Results: EM={avg_em:.3f}, MRR={avg_mrr:.3f}, HR@1={avg_hit_1:.3f}, Latency={avg_latency:.3f}s")
                
            except Exception as e:
                print(f"Error with {chunker_name} + {retriever_name}: {str(e)}")
                # Continue with next combination
    
    # Convert to DataFrame for easier analysis
    results_df = pd.DataFrame(results)
    chunking_df = pd.DataFrame(chunking_stats)
    
    return results_df, chunking_df

def generate_comparison_plots(results_df: pd.DataFrame, chunking_df: pd.DataFrame, output_dir: str = "results"):
    """Generate comparison visualizations from evaluation results."""
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(exist_ok=True)
    
    # Set plot style
    plt.style.use('ggplot')
    sns.set_theme(style="whitegrid")
    
    # 1. EM Score by Chunker and Retriever
    plt.figure(figsize=(12, 8))
    chart = sns.catplot(
        data=results_df, 
        kind="bar",
        x="chunker", y="em_score", 
        hue="retriever",
        height=6, aspect=1.5
    )
    chart.set_xticklabels(rotation=45, ha="right")
    plt.title("Exact Match Score by Chunker and Retriever")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/em_score_comparison.png", dpi=300)
    
    # 2. MRR Score by Chunker and Retriever
    plt.figure(figsize=(12, 8))
    chart = sns.catplot(
        data=results_df, 
        kind="bar",
        x="chunker", y="mrr", 
        hue="retriever",
        height=6, aspect=1.5
    )
    chart.set_xticklabels(rotation=45, ha="right")
    plt.title("Mean Reciprocal Rank by Chunker and Retriever")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/mrr_comparison.png", dpi=300)
    
    # 3. Hit Rate @ 1 comparison
    plt.figure(figsize=(12, 8))
    chart = sns.catplot(
        data=results_df, 
        kind="bar",
        x="chunker", y="hit_rate_1", 
        hue="retriever",
        height=6, aspect=1.5
    )
    chart.set_xticklabels(rotation=45, ha="right")
    plt.title("Hit Rate @ 1 by Chunker and Retriever")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/hit_rate_1_comparison.png", dpi=300)
    
    # 4. Query Time comparison
    plt.figure(figsize=(12, 8))
    chart = sns.catplot(
        data=results_df, 
        kind="bar",
        x="chunker", y="avg_query_time", 
        hue="retriever",
        height=6, aspect=1.5
    )
    chart.set_xticklabels(rotation=45, ha="right")
    plt.title("Average Query Time by Chunker and Retriever")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/query_time_comparison.png", dpi=300)
    
    # 5. Memory Usage comparison
    plt.figure(figsize=(12, 8))
    chart = sns.catplot(
        data=results_df, 
        kind="bar",
        x="chunker", y="query_memory_mb", 
        hue="retriever",
        height=6, aspect=1.5
    )
    chart.set_xticklabels(rotation=45, ha="right")
    plt.title("Memory Usage by Chunker and Retriever")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/memory_usage_comparison.png", dpi=300)
    
    # 6. Indexing Time comparison
    plt.figure(figsize=(12, 8))
    chart = sns.catplot(
        data=results_df, 
        kind="bar",
        x="chunker", y="indexing_time", 
        hue="retriever",
        height=6, aspect=1.5
    )
    chart.set_xticklabels(rotation=45, ha="right")
    plt.title("Indexing Time by Chunker and Retriever")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/indexing_time_comparison.png", dpi=300)
    
    # 7. Performance vs Indexing Time (scatter plot)
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=results_df,
        x="indexing_time", y="em_score",
        hue="retriever", style="chunker",
        s=100, alpha=0.7
    )
    plt.title("Performance vs Indexing Time")
    plt.xlabel("Indexing Time (s)")
    plt.ylabel("Exact Match Score")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/performance_vs_indexing_time.png", dpi=300)
    
    # 8. Performance vs Query Time (scatter plot)
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=results_df,
        x="avg_query_time", y="em_score",
        hue="retriever", style="chunker",
        s=100, alpha=0.7
    )
    plt.title("Performance vs Query Time")
    plt.xlabel("Average Query Time (s)")
    plt.ylabel("Exact Match Score")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/performance_vs_query_time.png", dpi=300)
    
    # 9. Performance vs Memory Usage (scatter plot)
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=results_df,
        x="query_memory_mb", y="em_score",
        hue="retriever", style="chunker",
        s=100, alpha=0.7
    )
    plt.title("Performance vs Memory Usage")
    plt.xlabel("Memory Usage (MB)")
    plt.ylabel("Exact Match Score")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/performance_vs_memory.png", dpi=300)
    
    # 10. Chunking Quality Metrics
    plt.figure(figsize=(12, 8))
    chunking_metrics = chunking_df.melt(
        id_vars=["chunker"], 
        value_vars=["avg_chunk_length", "sentence_completeness", "chunk_length_variance"],
        var_name="metric", value_name="value"
    )
    sns.catplot(
        data=chunking_metrics,
        kind="bar",
        x="chunker", y="value",
        hue="metric",
        height=6, aspect=1.5
    )
    plt.title("Chunking Quality Metrics")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/chunking_quality_metrics.png", dpi=300)
    
    # 11. Radar Chart for Top Performers
    # Get top performer for each retriever
    top_performers = []
    for retriever in results_df['retriever'].unique():
        retriever_df = results_df[results_df['retriever'] == retriever]
        top_row = retriever_df.loc[retriever_df['em_score'].idxmax()]
        top_performers.append(top_row)
    
    top_df = pd.DataFrame(top_performers)
    
    # Prepare radar chart
    metrics = ['em_score', 'precision', 'recall', 'mrr', 'hit_rate_1']
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Close the plot
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    for _, row in top_df.iterrows():
        values = [row[m] for m in metrics]
        values += values[:1]  # Close the polygon
        
        label = f"{row['retriever']} + {row['chunker']}"
        ax.plot(angles, values, linewidth=2, label=label)
        ax.fill(angles, values, alpha=0.1)
    
    # Add labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_title("Top Performers by Retriever")
    plt.legend(loc='upper right')
    plt.savefig(f"{output_dir}/radar_chart_top_performers.png", dpi=300)
    
    # 12. Query Complexity vs Performance
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=results_df,
        x="complexity_correlation", y="em_score",
        hue="retriever", style="chunker",
        s=100, alpha=0.7
    )
    plt.title("Query Complexity Correlation vs Performance")
    plt.xlabel("Correlation between Query Complexity and EM Score")
    plt.ylabel("Average EM Score")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/complexity_vs_performance.png", dpi=300)
    
    # 13. Performance Profile - Multiple Metrics in One View
    # Normalize metrics for better visualization
    metrics_to_plot = ['em_score', 'mrr', 'f1_score', 'hit_rate_1', 'hit_rate_3', 'semantic_similarity']
    profile_data = results_df.copy()
    
    for metric in metrics_to_plot:
        max_val = profile_data[metric].max()
        if max_val > 0:
            profile_data[f"{metric}_norm"] = profile_data[metric] / max_val
    
    # Melt the dataframe for plotting
    profile_melted = profile_data.melt(
        id_vars=["chunker", "retriever"], 
        value_vars=[f"{m}_norm" for m in metrics_to_plot],
        var_name="metric", value_name="normalized_value"
    )
    
    # Clean up metric names for display
    profile_melted["metric"] = profile_melted["metric"].str.replace("_norm", "")
    
    plt.figure(figsize=(14, 8))
    g = sns.catplot(
        data=profile_melted,
        kind="bar",
        x="metric", y="normalized_value",
        hue="retriever", col="chunker",
        height=6, aspect=0.8,
        col_wrap=3,
        sharey=True
    )
    g.set_titles("{col_name}")
    plt.suptitle("Performance Profile Across Metrics (Normalized)", y=1.02)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/performance_profile.png", dpi=300)
    
    # Save results to CSV
    results_df.to_csv(f"{output_dir}/benchmark_results.csv", index=False)
    chunking_df.to_csv(f"{output_dir}/chunking_metrics.csv", index=False)
    
    print(f"Saved all plots and results to {output_dir}/")

def triviaqa_benchmark(dataset_path, max_docs=300, max_qa=100, top_k=5):
    """Run comprehensive benchmarking on TriviaQA dataset."""
    # Load data
    docs, qa_pairs = load_triviaqa(dataset_path, max_docs, max_qa)
    if not docs:
        raise RuntimeError("No non-empty docs – check dataset path / permissions")

    # Run all strategies
    results_df, chunking_df = run_all_strategies(docs, qa_pairs, top_k)
    
    # Generate visualizations
    generate_comparison_plots(results_df, chunking_df)
    
    return results_df, chunking_df

def main():
    # Parameters
    TRIVIAQA_PATH = "../triviaqa-unfiltered/unfiltered-web-dev.json"
    MAX_DOCS = 100  # Reduced for faster evaluation
    MAX_QA = 50     # Reduced for faster evaluation
    TOP_K = 5
    
    # Run the benchmark
    results_df, chunking_df = triviaqa_benchmark(TRIVIAQA_PATH, MAX_DOCS, MAX_QA, TOP_K)
    
    # Display summary table
    print("\n===== BENCHMARK SUMMARY =====")
    summary = results_df.groupby(['chunker', 'retriever']).agg({
        'em_score': 'mean',
        'mrr': 'mean',
        'hit_rate_1': 'mean',
        'f1_score': 'mean',
        'avg_query_time': 'mean',
        'indexing_time': 'mean',
        'query_memory_mb': 'mean'
    }).reset_index()
    
    # Sort by EM score (best performers first)
    summary_sorted = summary.sort_values('em_score', ascending=False)
    print(summary_sorted.to_string(index=False, float_format="%.3f"))
    
    # Print top performers for different criteria
    best_em = summary_sorted.iloc[0]
    print(f"\nBest for accuracy: {best_em['chunker']} chunking with {best_em['retriever']} retrieval")
    print(f"  → EM Score: {best_em['em_score']:.3f}, MRR: {best_em['mrr']:.3f}, HR@1: {best_em['hit_rate_1']:.3f}")
    print(f"  → Query time: {best_em['avg_query_time']:.3f}s, Memory: {best_em['query_memory_mb']:.1f}MB")
    
    # Fastest with decent performance (at least 60% of best)
    decent_threshold = 0.6 * best_em['em_score']
    fastest = summary_sorted[summary_sorted['em_score'] >= decent_threshold].sort_values('avg_query_time').iloc[0]
    print(f"\nBest for speed: {fastest['chunker']} chunking with {fastest['retriever']} retrieval")
    print(f"  → EM Score: {fastest['em_score']:.3f}, Query time: {fastest['avg_query_time']:.3f}s")
    
    # Most memory efficient with decent performance
    memory_efficient = summary_sorted[summary_sorted['em_score'] >= decent_threshold].sort_values('query_memory_mb').iloc[0]
    print(f"\nBest for memory efficiency: {memory_efficient['chunker']} chunking with {memory_efficient['retriever']} retrieval")
    print(f"  → EM Score: {memory_efficient['em_score']:.3f}, Memory: {memory_efficient['query_memory_mb']:.1f}MB")
    
    # Best compromise (product of normalized metrics)
    # Normalize metrics
    max_em = summary['em_score'].max()
    min_time = summary['avg_query_time'].min() 
    min_memory = summary['query_memory_mb'].min()
    
    # Create balanced score (higher is better)
    summary['balanced_score'] = (
        (summary['em_score'] / max_em) * 
        (min_time / (summary['avg_query_time'] + 0.001)) * 
        (min_memory / (summary['query_memory_mb'] + 0.001))
    )
    
    balanced = summary.sort_values('balanced_score', ascending=False).iloc[0]
    print(f"\nBest overall balance: {balanced['chunker']} chunking with {balanced['retriever']} retrieval")
    print(f"  → EM Score: {balanced['em_score']:.3f}, Query time: {balanced['avg_query_time']:.3f}s, Memory: {balanced['query_memory_mb']:.1f}MB")

if __name__ == "__main__":
    main()




