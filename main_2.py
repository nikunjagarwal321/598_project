import json
import os
from typing import Dict, List, Tuple


def load_triviaqa(dataset_path: str) -> Tuple[Dict[str, str], List[Dict]]:
    """
    Load TriviaQA dataset and return documents and QA pairs.

    Args:
        dataset_path: Path to the TriviaQA JSON file (e.g., 'unfiltered-dev.json')
        max_docs: Maximum number of documents to load
        max_qa_pairs: Maximum number of QA pairs to load

    Returns:
        Tuple of (documents dictionary, list of QA pairs)
    """
    print(f"Loading TriviaQA dataset from: {dataset_path}")

    # Load the TriviaQA dataset
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)['Data']

    # Extract documents and QA pairs
    docs = {}
    qa_pairs = []

    for i, item in enumerate(data):
        if len(docs) > 200:
            break
        question = item['Question']
        answers = item['Answer']['NormalizedAliases']

        # Use the first answer as the gold answer
        gold_answer = answers if answers else list(item['Answer']['NormalizedValue'])

        qa_pairs.append({
            "question": question,
            "answer": gold_answer
        })

        # Extract documents from Wikipedia sources
        if 'EntityPages' in item:
            for j, doc_info in enumerate(item['EntityPages']):

                doc_id = f"wiki_{doc_info['Title']}_{j}"

                # Check if file exists
                if 'Filename' in doc_info:
                    doc_path = os.path.join(os.path.dirname("data/wikipedia"), doc_info['Filename'])
                    if os.path.exists(doc_path):
                        try:
                            with open(doc_path, 'r', encoding='utf-8') as doc_file:
                                doc_content = doc_file.read()
                                docs[doc_id] = doc_content

                        except:
                            # If file can't be read, use the snippet
                            docs[doc_id] = doc_info.get('Snippet', '')
                    else:
                        # If file doesn't exist, use the snippet
                        docs[doc_id] = doc_info.get('Snippet', '')
                else:
                    # If no filename, use the snippet
                    docs[doc_id] = doc_info.get('Snippet', '')

        # Add web documents if available
        if 'SearchResults' in item:
            for j, doc_info in enumerate(item['SearchResults']):

                doc_id = f"web_{j}_{doc_info.get('Title', '')}"

                if 'Filename' in doc_info:
                    doc_path = os.path.join("data/web", doc_info['Filename'])
                    if os.path.exists(doc_path):
                        try:
                            with open(doc_path, 'r', encoding='utf-8') as doc_file:
                                doc_content = doc_file.read()
                                docs[doc_id] = doc_content
                        except:
                            docs[doc_id] = doc_info.get('Snippet', '')
                    else:
                        docs[doc_id] = doc_info.get('Snippet', '')
                else:
                    docs[doc_id] = doc_info.get('Snippet', '')

    print(f"Loaded {len(docs)} documents and {len(qa_pairs)} QA pairs from TriviaQA")
    return docs, qa_pairs


documents_dict, qa_pairs = load_triviaqa("triviaqa-unfiltered/unfiltered-web-dev.json")
# %%
import time


def benchmark_retrieval(retriever, query, top_k):
    start = time.time()
    results = retriever.retrieve(query, top_k=top_k)
    end = time.time()
    latency = end - start
    return results, latency


def match_exists(retrieved_docs, answers):
    for doc in retrieved_docs:
        if any(ans.lower() in doc.lower() for ans in answers):
            return 1
    return 0


def benchmark_all(strategies, documents, qas, top_k=5):
    results = []
    overall_retriever_results = []

    for chunker_name, chunker in strategies["chunkers"].items():
        chunks = []

        # Chunking Latency
        chunk_start_time = time.time()
        for doc in documents:
            chunks.extend(chunker.chunk(doc))
        chunk_end_time = time.time()
        chunking_latency = chunk_end_time - chunk_start_time

        # Retriever Latency
        for retriever_name, retriever_class in strategies["retrievers"].items():
            retriever_initialize_start_time = time.time()
            retriever = retriever_class(chunks)
            retriever_initialize_end_time = time.time()
            retriever_initialize_latency = retriever_initialize_end_time - retriever_initialize_start_time
            total_retrieve_latency = 0
            correct_docs = 0

            for qa in qas:
                query, answers = qa["question"], qa["answer"]
                retrieved_docs, latency = benchmark_retrieval(retriever, query, top_k)
                does_match = match_exists(retrieved_docs, answers)
                results.append({
                    "query": query,
                    "chunker": chunker_name,
                    "retriever": retriever_name,
                    "latency": latency,
                    "results": retrieved_docs,
                    "answers": answers,
                    "does_match": does_match
                })
                total_retrieve_latency += latency
                correct_docs += does_match
            overall_result = {
                "retriever": retriever_name,
                "chunker": chunker_name,
                "initialize_latency": retriever_initialize_latency,
                "retrieve_latency": total_retrieve_latency / len(qas),
                "chunking_latency": chunking_latency,
                "accuracy": correct_docs / len(qas) * 100,
            }
            print(overall_result)
            overall_retriever_results.append(overall_result)

    return results, overall_retriever_results


# %%
from chunkers import FixedChunker, OverlappingChunker, SemanticChunker
from retrievers.bm25_retriever import BM25Retriever
from retrievers.dense_retriever import DPRRetriever
from retrievers.colbert_retriever import ColBERTRetriever
from retrievers.hybrid_retriever import HybridRetriever

documents = list(documents_dict.values())
top_k = 5

strategies = {
    "chunkers": {
        "fixed": FixedChunker(chunk_size=20, drop_last=False),
        "overlapping": OverlappingChunker(chunk_size=24, overlap=8, drop_last=False),
        "semantic": SemanticChunker(chunk_char_limit=120)
    },
    "retrievers": {
        "bm25": BM25Retriever,
        # "dpr": DPRRetriever,
        "colbert": ColBERTRetriever,
        "hybrid": HybridRetriever
    }
}

results, overall_retriever_results = benchmark_all(strategies, documents, qa_pairs, top_k)

# Print results
for res in overall_retriever_results:
    print("Result", res)

import matplotlib.pyplot as plt
import numpy as np

# Extract values
retrievers = [entry['retriever'] for entry in overall_retriever_results]
init_latencies = [entry['initialize_latency'] for entry in overall_retriever_results]
retrieve_latencies = [entry['retrieve_latency'] for entry in overall_retriever_results]
chunk_latencies = [entry['chunking_latency'] for entry in overall_retriever_results]
accuracies = [entry['accuracy'] for entry in overall_retriever_results]

x = np.arange(len(retrievers))
width = 0.2

# Create subplot for latency
fig, ax1 = plt.subplots(figsize=(10, 6))

# Latency bars
ax1.bar(x - width, init_latencies, width, label='Init Latency')
ax1.bar(x, retrieve_latencies, width, label='Retrieve Latency')
ax1.bar(x + width, chunk_latencies, width, label='Chunking Latency')

ax1.set_xlabel('Retriever')
ax1.set_ylabel('Latency (seconds)')
ax1.set_title('Retriever Latency Comparison')
ax1.set_xticks(x)
ax1.set_xticklabels(retrievers)
ax1.legend(loc='upper left')

# Create secondary y-axis for accuracy
ax2 = ax1.twinx()
ax2.plot(x, accuracies, color='green', marker='o', linewidth=2, label='Accuracy (%)')
ax2.set_ylabel('Accuracy (%)')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.show()
