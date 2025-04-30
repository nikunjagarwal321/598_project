import json
import os
from typing import Dict, List, Tuple

def load_triviaqa(dataset_path: str, max_docs: int = 100, max_qa_pairs: int = 50) -> Tuple[Dict[str, str], List[Dict]]:
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
        if i >= max_qa_pairs:
            break
            
        question = item['Question']
        answers = item['Answer']['NormalizedAliases']
        
        # Use the first answer as the gold answer
        gold_answer = answers[0] if answers else item['Answer']['NormalizedValue']
        
        qa_pairs.append({
            "question": question,
            "answer": gold_answer
        })
        
        # Extract documents from Wikipedia sources
        if 'EntityPages' in item:
            for j, doc_info in enumerate(item['EntityPages']):
                if len(docs) >= max_docs:
                    break
                    
                doc_id = f"wiki_{doc_info['Title']}_{j}"
                
                # Check if file exists
                if 'Filename' in doc_info:
                    doc_path = os.path.join(os.path.dirname(dataset_path), doc_info['Filename'])
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
                if len(docs) >= max_docs:
                    break
                    
                doc_id = f"web_{j}_{doc_info.get('Title', '')}"
                
                if 'Filename' in doc_info:
                    doc_path = os.path.join(os.path.dirname(dataset_path), doc_info['Filename'])
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