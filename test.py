#!/usr/bin/env python
"""
Enhanced Retrieval Evaluation for TriviaQA-unfiltered.

Dependencies:
  pip install rank-bm25 sentence-transformers faiss-cpu tqdm pandas matplotlib numpy
"""

from __future__ import annotations
import os, json, gzip, time, random
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
from datetime import datetime

# Retrieval libs
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss

# Chunkers
from chunkers import FixedChunker, OverlappingChunker, SemanticChunker

# Add reranker imports
from rerankers.embedding_reranker import EmbeddingReranker
from rerankers.cross_encoder_reranker import CrossEncoderReranker

# Add retriever imports 
from retrievers.bm25_retriever import BM25Retriever
from retrievers.dense_retriever import DPRRetriever
from retrievers.colbert_retriever import ColBERTRetriever
from retrievers.hybrid_retriever import HybridRetriever

# ────────────────────────────────────────────────────────────────────────────
# 1. TriviaQA loader
# ────────────────────────────────────────────────────────────────────────────
def load_triviaqa(
    dataset_path: str,
    max_docs: int = 300,
    max_qa_pairs: int = 100,
) -> Tuple[Dict[str, str], List[Dict]]:
    """
    Load TriviaQA dataset and return documents and QA pairs.
    
    Returns:
        docs   : {doc_id: raw_text}
        qa_pairs: [{"question": str, "answer": str}, …]
    """
    print(f"[load] TriviaQA from {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)["Data"]

    root = Path(dataset_path).parent
    docs: Dict[str, str] = {}
    qa_pairs: List[Dict] = []

    def read_evidence(file_path: Path) -> str:
        if not file_path.exists():
            return ""
        # Many evidence files are .gz
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
        qa_pairs.append({"question": question, "answer": gold})

        # Wiki evidence
        for j, d in enumerate(item.get("EntityPages", [])):
            add_doc(d, f"wiki_{d.get('Title','wiki')}", j)
        # Web search evidence
        for j, d in enumerate(item.get("SearchResults", [])):
            add_doc(d, "web", j)

    # Drop empties
    docs = {k: v for k, v in docs.items() if v.strip()}
    print(f"[load] kept {len(docs)} non-empty docs, {len(qa_pairs)} QA pairs")
    return docs, qa_pairs


# ────────────────────────────────────────────────────────────────────────────
# 2. Evaluation Metrics
# ────────────────────────────────────────────────────────────────────────────
def exact_match(gold: str, passages: List[str]) -> int:
    """Check if gold answer is contained in any passage."""
    g = gold.lower()
    return int(any(g in p.lower() for p in passages))

def precision_at_k(gold: str, passages: List[str]) -> float:
    """Calculate precision@k - fraction of retrieved items that are relevant."""
    if not passages:
        return 0.0
    relevant = sum(1 for p in passages if gold.lower() in p.lower())
    return relevant / len(passages)

def recall_at_k(gold: str, passages: List[str]) -> float:
    """Simplified recall - did we find the answer at all."""
    return exact_match(gold, passages)

def mrr(gold: str, passages: List[str]) -> float:
    """Mean Reciprocal Rank - 1/position of first relevant item."""
    for i, passage in enumerate(passages):
        if gold.lower() in passage.lower():
            return 1.0 / (i + 1)
    return 0.0


# ────────────────────────────────────────────────────────────────────────────
# 3. Main
# ────────────────────────────────────────────────────────────────────────────
def run_evaluation(
    docs: Dict[str, str], 
    qa_pairs: List[Dict], 
    chunker_name: str,
    chunker_params: Dict[str, Any],
    retrievers: List[str] = ["bm25", "dpr"],
    rerankers: List[str] = ["none"],
    top_k_values: List[int] = [5],
    output_dir: str = "results",
    seed: int = 42
) -> Dict[str, Dict[str, float]]:
    """Run evaluation with given chunker and save results."""
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # DEBUG: Log document statistics
    doc_lengths = [len(txt) for txt in docs.values()]
    doc_token_lengths = [len(txt.split()) for txt in docs.values()]
    print(f"[debug] Document stats: {len(docs)} docs")
    print(f"[debug] Chars - min: {min(doc_lengths)}, max: {max(doc_lengths)}, avg: {sum(doc_lengths)/len(doc_lengths):.1f}")
    print(f"[debug] Tokens - min: {min(doc_token_lengths)}, max: {max(doc_token_lengths)}, avg: {sum(doc_token_lengths)/len(doc_token_lengths):.1f}")
    
    # Print a sample document for inspection
    sample_doc_id = list(docs.keys())[0]
    print(f"[debug] Sample document '{sample_doc_id}': {docs[sample_doc_id][:200]}...")
    
    # Initialize chunker based on name and params
    if chunker_name == "fixed":
        chunker = FixedChunker(**chunker_params)
    elif chunker_name == "overlap":
        chunker = OverlappingChunker(**chunker_params)
    elif chunker_name == "semantic":
        chunker = SemanticChunker(**chunker_params)
    else:
        raise ValueError(f"Unknown chunker: {chunker_name}")
    
    # DEBUG: Track per-document chunking statistics
    doc_chunk_counts = {}
    
    # Chunk documents
    chunk_texts, chunk_to_doc = [], []
    for doc_id, txt in docs.items():
        doc_chunks = chunker.chunk(txt)
        doc_chunk_counts[doc_id] = len(doc_chunks)
        
        for ch in doc_chunks:
            chunk_texts.append(ch)
            chunk_to_doc.append(doc_id)
    
    # DEBUG: Analyze chunking results
    chunk_counts = list(doc_chunk_counts.values())
    if chunk_counts:
        print(f"[debug] Chunks per doc - min: {min(chunk_counts)}, max: {max(chunk_counts)}, avg: {sum(chunk_counts)/len(chunk_counts):.1f}")
        print(f"[debug] Docs with multiple chunks: {sum(1 for c in chunk_counts if c > 1)} out of {len(chunk_counts)}")
        
        # Print chunk counts distribution
        chunk_dist = {}
        for count in chunk_counts:
            chunk_dist[count] = chunk_dist.get(count, 0) + 1
        print(f"[debug] Chunk count distribution: {chunk_dist}")
        
        # Debug sample chunks for a document that should have been chunked
        if sample_doc_id in doc_chunk_counts:
            sample_chunks = chunker.chunk(docs[sample_doc_id])
            print(f"[debug] Sample doc chunks ({len(sample_chunks)}):")
            for i, chunk in enumerate(sample_chunks[:3]):
                print(f"[debug] Chunk {i}: {chunk[:100]}...")
            if len(sample_chunks) > 3:
                print(f"[debug] ... and {len(sample_chunks) - 3} more chunks")

    print(f"[chunk] {chunker.__class__.__name__}: {len(chunk_texts)} chunks "
          f"(≈{len(chunk_texts)/len(docs):.1f} per doc)")
    
    # Initialize all retrievers
    retriever_map = {}
    
    # Only initialize the retrievers that are requested
    if "bm25" in retrievers:
        # Create BM25 retriever with raw chunks
        print("[retriever] Initializing BM25...")
        bm25_index = BM25Okapi([c.split() for c in chunk_texts])
        retriever_map["bm25"] = lambda q, k: [chunk_texts[i] for i in bm25_index.get_top_n(q.split(), list(range(len(chunk_texts))), n=k)]
    
    if "dpr" in retrievers:
        # Create DPR with embeddings
        print("[retriever] Initializing DPR...")
        dpr_model = SentenceTransformer("facebook-dpr-ctx_encoder-multiset-base")
        print("[dpr] encoding chunks...")
        ctx_emb = dpr_model.encode(
            chunk_texts,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        faiss_index = faiss.IndexFlatIP(ctx_emb.shape[1])
        faiss_index.add(ctx_emb)
        print(f"[dpr] Faiss index: {faiss_index.ntotal} vectors")
        
        retriever_map["dpr"] = lambda q, k: [
            chunk_texts[i] for i in faiss_index.search(
                dpr_model.encode([q], normalize_embeddings=True), 
                k
            )[1][0]
        ]
    
    if "colbert" in retrievers:
        print("[retriever] Initializing ColBERT...")
        colbert = ColBERTRetriever(chunk_texts)
        retriever_map["colbert"] = lambda q, k: colbert.retrieve(q, top_k=k)
    
    if "hybrid" in retrievers:
        print("[retriever] Initializing Hybrid retriever...")
        hybrid = HybridRetriever(chunk_texts)
        retriever_map["hybrid"] = lambda q, k: hybrid.retrieve(q, top_k=k)
    
    # Initialize rerankers
    reranker_map = {"none": None}
    
    if "embed" in rerankers:
        print("[reranker] Initializing embedding reranker...")
        reranker_map["embed"] = EmbeddingReranker()
    
    if "cross" in rerankers:
        print("[reranker] Initializing cross-encoder reranker...")
        reranker_map["cross"] = CrossEncoderReranker()
    
    # Create result trackers for all combinations
    results = {}
    for retriever_name in retrievers:
        for reranker_name in rerankers:
            for top_k in top_k_values:
                system_id = f"{retriever_name}+{reranker_name}@{top_k}"
                results[system_id] = {
                    "latency": [], "em": [], "precision": [], "recall": [], "mrr": []
                }
    
    # Store detailed results for each query
    query_results = []
    
    # Evaluation loop
    for qa in tqdm(qa_pairs, desc="evaluating"):
        q, gold = qa["question"], qa["answer"]
        query_data = {"question": q, "answer": gold}
        
        # Process each retriever-reranker-topk combination
        for retriever_name in retrievers:
            retriever = retriever_map[retriever_name]
            
            for reranker_name in rerankers:
                reranker = reranker_map[reranker_name]
                
                for top_k in top_k_values:
                    system_id = f"{retriever_name}+{reranker_name}@{top_k}"
                    
                    # Retrieval step
                    t0 = time.perf_counter()
                    retrieved_docs = retriever(q, top_k)
                    
                    # Reranking step (if applicable)
                    if reranker and reranker_name != "none":
                        retrieved_docs = reranker.rerank(q, retrieved_docs)[:top_k]
                        
                    latency = (time.perf_counter() - t0) * 1e3
                    
                    # Compute metrics
                    results[system_id]["latency"].append(latency)
                    results[system_id]["em"].append(exact_match(gold, retrieved_docs))
                    results[system_id]["precision"].append(precision_at_k(gold, retrieved_docs))
                    results[system_id]["recall"].append(recall_at_k(gold, retrieved_docs))
                    results[system_id]["mrr"].append(mrr(gold, retrieved_docs))
                    
                    # Store passages for this configuration
                    query_data[f"{system_id}_passages"] = retrieved_docs
                    query_data[f"{system_id}_latency"] = latency
        
        query_results.append(query_data)
    
    # Calculate aggregates
    summary = {}
    for system_id, metrics in results.items():
        summary[system_id] = {
            metric: np.mean(values) for metric, values in metrics.items()
        }
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{chunker_name}_{timestamp}"
    
    # Save summary results
    summary_df = pd.DataFrame.from_dict({
        (system_id, metric): [value] 
        for system_id, metrics in summary.items() 
        for metric, value in metrics.items()
    }, orient='columns')
    
    # Add experiment metadata
    summary_df["chunker"] = chunker_name
    summary_df["chunker_params"] = str(chunker_params)
    summary_df["retrievers"] = str(retrievers)
    summary_df["rerankers"] = str(rerankers)
    summary_df["top_k_values"] = str(top_k_values)
    summary_df["seed"] = seed
    summary_df["num_chunks"] = len(chunk_texts)
    summary_df["chunks_per_doc"] = len(chunk_texts) / len(docs)
    
    # Save to CSV
    summary_path = output_path / f"{experiment_name}_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    
    # Save detailed query results
    details_path = output_path / f"{experiment_name}_details.json"
    with open(details_path, 'w') as f:
        json.dump(query_results, f, indent=2)
    
    # Generate plots
    plot_path = output_path / f"{experiment_name}_metrics.png"
    plot_metrics(summary, chunker_name, chunker_params, plot_path)
    
    print(f"\n=== RESULTS for {chunker_name} ({chunker_params}) ===")
    # Print a table of results, grouped by retriever and showing all metrics
    for retriever_name in retrievers:
        print(f"\n{retriever_name.upper()} Retriever:")
        for reranker_name in rerankers:
            for top_k in top_k_values:
                system_id = f"{retriever_name}+{reranker_name}@{top_k}"
                metrics = summary[system_id]
                print(f"  {reranker_name:5s} @ k={top_k:<2d} | EM={metrics['em']:.3f} | "
                     f"P@{top_k}={metrics['precision']:.3f} | "
                     f"MRR={metrics['mrr']:.3f} | Latency≈{metrics['latency']:.1f} ms")
    
    print(f"Results saved to {output_path}")
    
    return summary


def plot_metrics(summary, chunker_name, chunker_params, output_path):
    """Generate a bar chart of retrieval metrics."""
    metrics = ["em", "precision", "recall", "mrr"]
    labels = ["Exact Match", "Precision@k", "Recall@k", "MRR"]
    
    # Group systems by retriever type
    systems_by_retriever = {}
    for system_id in summary.keys():
        retriever, config = system_id.split('+', 1)
        if retriever not in systems_by_retriever:
            systems_by_retriever[retriever] = []
        systems_by_retriever[retriever].append(system_id)
    
    # Create a plot for each retriever
    for retriever, systems in systems_by_retriever.items():
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(metrics))
        width = 0.8 / len(systems)  # Scale bar width based on number of systems
        
        for i, system_id in enumerate(systems):
            values = [summary[system_id][metric] for metric in metrics]
            offset = i * width - (len(systems) - 1) * width / 2
            ax.bar(x + offset, values, width, label=system_id.split('+')[1])
        
        ax.set_ylabel('Score')
        ax.set_title(f'Retrieval Metrics: {retriever.upper()} Retriever\n{chunker_name.capitalize()} Chunker ({chunker_params})')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend(title="Reranker@k", loc="upper right")
        
        fig.tight_layout()
        plt.savefig(output_path.with_stem(f"{output_path.stem}_{retriever}"))
        plt.close(fig)
    
    # Create a combined plot showing the best configuration of each retriever
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Find best configuration for each retriever based on EM
    best_systems = []
    for retriever, systems in systems_by_retriever.items():
        best_system = max(systems, key=lambda s: summary[s]['em'])
        best_systems.append(best_system)
    
    x = np.arange(len(metrics))
    width = 0.8 / len(best_systems)
    
    for i, system_id in enumerate(best_systems):
        values = [summary[system_id][metric] for metric in metrics]
        offset = i * width - (len(best_systems) - 1) * width / 2
        ax.bar(x + offset, values, width, label=system_id)
    
    ax.set_ylabel('Score')
    ax.set_title(f'Best Retrieval Configurations\n{chunker_name.capitalize()} Chunker ({chunker_params})')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(title="System", loc="upper right")
    
    fig.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Run retrieval evaluation experiments')
    parser.add_argument('--dataset', default="../triviaqa-unfiltered/unfiltered-web-dev.json",
                      help='Path to TriviaQA dataset')
    parser.add_argument('--max_docs', type=int, default=300,
                      help='Maximum number of documents to load')
    parser.add_argument('--max_qa', type=int, default=100,
                      help='Maximum number of QA pairs to evaluate')
    parser.add_argument('--output_dir', default='results',
                      help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    parser.add_argument('--chunker', choices=['fixed', 'overlap', 'semantic', 'all'],
                      default='all', help='Chunker to use')
    parser.add_argument('--retrievers', default='bm25,dpr',
                      help='Comma-separated list of retrievers to evaluate (bm25,dpr,colbert,hybrid)')
    parser.add_argument('--rerankers',  default='none',
                      help='Comma-separated list of rerankers to evaluate (none,embed,cross)')
    parser.add_argument('--top_k_list', default='5,10',
                      help='Comma-separated list of k values for retrieval depth')
    
    args = parser.parse_args()
    
    # Parse comma-separated argument lists
    retrievers = [r.strip() for r in args.retrievers.split(',')]
    rerankers = [r.strip() for r in args.rerankers.split(',')]
    top_k_values = [int(k) for k in args.top_k_list.split(',')]
    
    print(f"Will evaluate with:")
    print(f"  Retrievers: {retrievers}")
    print(f"  Rerankers: {rerankers}")
    print(f"  Top-k values: {top_k_values}")
    
    # Load dataset
    docs, qa_pairs = load_triviaqa(args.dataset, args.max_docs, args.max_qa)
    if not docs:
        raise RuntimeError("No non-empty docs – check dataset path / permissions")
    
    # Define chunker configurations to test
    chunker_configs = {
        "fixed": [
            {"chunk_size": 64, "drop_last": False},
            {"chunk_size": 128, "drop_last": False},
            {"chunk_size": 256, "drop_last": False},
        ],
        "overlap": [
            {"chunk_size": 128, "overlap": 32, "drop_last": False},
            {"chunk_size": 256, "overlap": 64, "drop_last": False},
        ],
        "semantic": [
            {"chunk_char_limit": 500},
            {"chunk_char_limit": 1000},
            {"chunk_char_limit": 1500},
        ],
    }
    
    # Run evaluations
    all_results = {}
    
    if args.chunker == 'all':
        chunkers_to_evaluate = list(chunker_configs.keys())
    else:
        chunkers_to_evaluate = [args.chunker]
    
    for chunker_name in chunkers_to_evaluate:
        for params in chunker_configs[chunker_name]:
            print(f"\n=== Evaluating {chunker_name} chunker with {params} ===")
            results = run_evaluation(
                docs, 
                qa_pairs,
                chunker_name,
                params,
                retrievers=retrievers,
                rerankers=rerankers,
                top_k_values=top_k_values,
                output_dir=args.output_dir,
                seed=args.seed
            )
            all_results[f"{chunker_name}_{str(params)}"] = results
    
    # Generate comparative visualization
    if len(all_results) > 1:
        generate_comparison_plots(all_results, args.output_dir)


def generate_comparison_plots(all_results, output_dir):
    """Generate plots comparing different chunking strategies and retrievers."""
    output_path = Path(output_dir)
    
    # Collect all system configurations and metrics
    all_systems = set()
    for exp_results in all_results.values():
        all_systems.update(exp_results.keys())
    
    metrics = ["em", "precision", "recall", "mrr", "latency"]
    labels = ["Exact Match", "Precision@k", "Recall@k", "MRR", "Latency (ms)"]
    
    # Create a combined dataframe for easier analysis
    rows = []
    for exp_name, exp_results in all_results.items():
        chunker_type = exp_name.split('_')[0]
        for system_id, metrics_values in exp_results.items():
            retriever_name, rest = system_id.split('+', 1)
            reranker_name, top_k = rest.split('@')
            
            row = {
                'experiment': exp_name,
                'chunker': chunker_type,
                'retriever': retriever_name,
                'reranker': reranker_name,
                'top_k': int(top_k),
                **metrics_values
            }
            rows.append(row)
    
    combined_df = pd.DataFrame(rows)
    
    # Save the combined results CSV
    combined_df.to_csv(output_path / "combined_results.csv", index=False)
    
    # Generate comparison plots for each metric
    for metric, label in zip(metrics, labels):
        # 1. Compare chunkers (best config per chunker)
        plt.figure(figsize=(14, 8))
        
        # Group by chunker and find best configuration for each
        chunker_comparison = combined_df.groupby(['chunker', 'retriever', 'reranker', 'top_k'])[metric].mean().reset_index()
        best_per_chunker = chunker_comparison.loc[chunker_comparison.groupby('chunker')[metric].idxmax()]
        
        # Plot bar chart
        ax = plt.subplot(111)
        chunkers = best_per_chunker['chunker'].unique()
        x = np.arange(len(chunkers))
        
        # Create bars
        ax.bar(x, best_per_chunker[metric], width=0.6)
        
        # Add configuration details as annotations
        for i, (_, row) in enumerate(best_per_chunker.iterrows()):
            ax.annotate(f"{row['retriever']}+{row['reranker']}@{row['top_k']}",
                       (i, row[metric]),
                       textcoords="offset points",
                       xytext=(0, 10),
                       ha='center')
        
        ax.set_ylabel(label)
        ax.set_title(f'Best {label} by Chunking Strategy')
        ax.set_xticks(x)
        ax.set_xticklabels(chunkers)
        
        plt.tight_layout()
        plt.savefig(output_path / f"best_chunker_{metric}.png")
        plt.close()
        
        # 2. Compare retrievers (best config per retriever)
        plt.figure(figsize=(14, 8))
        
        # Group by retriever and find best configuration
        retriever_comparison = combined_df.groupby(['retriever', 'chunker', 'reranker', 'top_k'])[metric].mean().reset_index()
        best_per_retriever = retriever_comparison.loc[retriever_comparison.groupby('retriever')[metric].idxmax()]
        
        # Plot bar chart
        ax = plt.subplot(111)
        retrievers = best_per_retriever['retriever'].unique()
        x = np.arange(len(retrievers))
        
        # Create bars
        ax.bar(x, best_per_retriever[metric], width=0.6)
        
        # Add configuration details as annotations
        for i, (_, row) in enumerate(best_per_retriever.iterrows()):
            ax.annotate(f"{row['chunker']}+{row['reranker']}@{row['top_k']}",
                       (i, row[metric]),
                       textcoords="offset points",
                       xytext=(0, 10),
                       ha='center')
        
        ax.set_ylabel(label)
        ax.set_title(f'Best {label} by Retriever Type')
        ax.set_xticks(x)
        ax.set_xticklabels(retrievers)
        
        plt.tight_layout()
        plt.savefig(output_path / f"best_retriever_{metric}.png")
        plt.close()
        
        # 3. Compare rerankers (best config per reranker)
        plt.figure(figsize=(14, 8))
        
        # Group by reranker and find best configuration
        reranker_comparison = combined_df.groupby(['reranker', 'chunker', 'retriever', 'top_k'])[metric].mean().reset_index()
        best_per_reranker = reranker_comparison.loc[reranker_comparison.groupby('reranker')[metric].idxmax()]
        
        # Plot bar chart
        ax = plt.subplot(111)
        rerankers = best_per_reranker['reranker'].unique()
        x = np.arange(len(rerankers))
        
        # Create bars
        ax.bar(x, best_per_reranker[metric], width=0.6)
        
        # Add configuration details as annotations
        for i, (_, row) in enumerate(best_per_reranker.iterrows()):
            ax.annotate(f"{row['chunker']}+{row['retriever']}@{row['top_k']}",
                       (i, row[metric]),
                       textcoords="offset points",
                       xytext=(0, 10),
                       ha='center')
        
        ax.set_ylabel(label)
        ax.set_title(f'Best {label} by Reranker Type')
        ax.set_xticks(x)
        ax.set_xticklabels(rerankers)
        
        plt.tight_layout()
        plt.savefig(output_path / f"best_reranker_{metric}.png")
        plt.close()
        
        # 4. Compare top-k values (best config per k)
        plt.figure(figsize=(14, 8))
        
        # Group by top_k and find best configuration
        topk_comparison = combined_df.groupby(['top_k', 'chunker', 'retriever', 'reranker'])[metric].mean().reset_index()
        best_per_topk = topk_comparison.loc[topk_comparison.groupby('top_k')[metric].idxmax()]
        
        # Plot bar chart
        ax = plt.subplot(111)
        topks = best_per_topk['top_k'].unique()
        x = np.arange(len(topks))
        
        # Create bars
        ax.bar(x, best_per_topk[metric], width=0.6)
        
        # Add configuration details as annotations
        for i, (_, row) in enumerate(best_per_topk.iterrows()):
            ax.annotate(f"{row['chunker']}+{row['retriever']}+{row['reranker']}",
                       (i, row[metric]),
                       textcoords="offset points",
                       xytext=(0, 10),
                       ha='center')
        
        ax.set_ylabel(label)
        ax.set_title(f'Best {label} by Retrieval Depth (k)')
        ax.set_xticks(x)
        ax.set_xticklabels([f'k={k}' for k in topks])
        
        plt.tight_layout()
        plt.savefig(output_path / f"best_topk_{metric}.png")
        plt.close()
    
    # Create top-level summary of the absolute best configuration
    best_configs = {}
    for metric in metrics:
        best_idx = combined_df[metric].idxmax()
        best_config = combined_df.loc[best_idx]
        best_configs[metric] = {
            'chunker': best_config['chunker'],
            'retriever': best_config['retriever'],
            'reranker': best_config['reranker'],
            'top_k': best_config['top_k'],
            'value': best_config[metric]
        }
    
    # Save best configurations as JSON
    with open(output_path / "best_configs.json", 'w') as f:
        json.dump(best_configs, f, indent=2)
    
    # Print best configurations
    print("\n=== BEST CONFIGURATIONS BY METRIC ===")
    for metric, config in best_configs.items():
        print(f"{metric.upper()}: {config['chunker']}+{config['retriever']}+{config['reranker']}@{config['top_k']} = {config['value']:.3f}")


if __name__ == "__main__":
    main()
