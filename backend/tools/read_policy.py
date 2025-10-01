import pandas as pd
import os
import re
import glob
import faiss
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import json
from collections import defaultdict
import gc
import fitz

def read_policy_coding(file_path: str) -> pd.DataFrame:
    """
    Reads a CSV file containing policy coding data and returns it as a pandas DataFrame.
    
    Args:
        file_path (str): The path to the CSV file.
        
    Returns:
        pd.DataFrame: A DataFrame containing the policy coding data.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error reading the file {file_path}: {e}")
        return pd.DataFrame()

def build_knowledge_tree(codebook):
    knowledge_tree_small = {}
    codebook = pd.read_csv("/home/vicente/Github/BDLab-Agent/backend/DISES/Codebook.csv", dtype=str)

    for idx, row in codebook.iterrows():
        if row['Code Name'] is None or str(row['Code Name']).lower() == 'nan' or len(str(row['Code Name']).strip()) == 0 or str(row["Definition"]).lower() == 'nan':
            continue
        else:
            if row["Code Name"] == 'Amount Allocated':
                continue
            raw_definition = str(row['Definition'])
            definition = raw_definition.split("if",1)[1].strip()
            knowledge_tree_small[row['Code Name']] = definition
        
    print(knowledge_tree_small)
    len(knowledge_tree_small)

def flatten_knowledge_tree(node, path=""):
    """
    Recursively traverses a nested dictionary and creates a list of
    augmented strings with the full hierarchical path.
    """
    items = []
    for key, value in node.items():
        current_path = f"{path} -> {key}" if path else key
        if "Definition" in value and isinstance(value["Definition"], str):
            definition = value["Definition"]
            # Store both the augmented string for embedding and the clean path for lookup
            items.append({
                "path": current_path,
                "augmented_string": f"Category: {current_path}. Definition: {definition}"
            })
        else:
            items.extend(flatten_knowledge_tree(value, path=current_path))
    return items

def build_faiss_index(flattened_items, embedder):
    
    augmented_definitions_for_embedding = []
    for item in flattened_items:
        augmented_definitions_for_embedding.append(item['augmented_string'])

    clean_code_paths = []
    for item in flattened_items:
        clean_code_paths.append(item['path'])

    print(f"Embedding {len(augmented_definitions_for_embedding)} augmented code definitions...")
    definition_embeddings = embedder.encode(
        augmented_definitions_for_embedding,
        show_progress_bar=True,
        convert_to_numpy=True
    ).astype('float32')

    embedding_dimension = definition_embeddings.shape[1]
    index = faiss.IndexFlatIP(embedding_dimension)
    faiss.normalize_L2(definition_embeddings)
    index.add(definition_embeddings)

    print("Faiss index for code definitions created successfully!")
    return index, clean_code_paths, embedder


def load_pdfs_from_directory(directory_path):
    """Loads text from all PDFs in a directory."""
    if not Path(directory_path).exists():
        print(f"Directory '{directory_path}' not found. Please create it and add your PDFs.")
        return []
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            filepath = os.path.join(directory_path, filename)
            try:
                doc = fitz.open(filepath)
                full_text = "".join(page.get_text() for page in doc)
                doc.close()
                if "References" in full_text:
                    full_text = full_text[:full_text.index("References")]
                documents.append({
                    "page_content": full_text,
                    "metadata": {"source": filename}
                })
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    print(f"Loaded {len(documents)} documents from '{directory_path}'")
    return documents

def split_into_chunks(text, max_chars=1000):
    """
    Splits text first by paragraph, then splits any long paragraphs by character count.
    """
    paragraphs = []
    for p in text.split('\n\n'):
        if len(p.strip()) > 200:
            paragraphs.append(p.strip())

    #paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 100]
    final_chunks = []
    for p in paragraphs:
        if len(p) > max_chars:
            # if a paragraph is too long split it further
            for i in range(0, len(p), max_chars):
                final_chunks.append(p[i:i + max_chars])
        else:
            final_chunks.append(p)
    return final_chunks

def parse_codes_from_text(raw_text, all_code_names):
    """Scans raw LLM output and extracts valid code names."""
    found_codes = []
    raw_text_lower = raw_text.lower()
    for code in all_code_names:
        if re.search(r'\b' + re.escape(code.lower()) + r'\b', raw_text_lower):
            found_codes.append(code)
    return list(dict.fromkeys(found_codes))

def classify_paragraph_with_rag(paragraph, k=5, index=None, clean_code_paths=None, augmented_definitions_for_embedding=None, model=None, tokenizer=None, embedder=None):
    """Classifies a single paragraph using the RAG pipeline."""
    # retrieve relevant context
    query_text = f"Represent this sentence for searching relevant passages: {paragraph}"
    paragraph_embedding = embedder.encode([query_text]).astype('float32')
    faiss.normalize_L2(paragraph_embedding)
    distances, indices = index.search(paragraph_embedding, k)
    
    retrieved_paths = []
    context_list = []
    for i in indices[0]:
        retrieved_paths.append(clean_code_paths[i])
        context_list.append(augmented_definitions_for_embedding[i])
    
    context = "\n".join(context_list)
    #context = "\n".join(augmented_definitions_for_embedding[i] for i in indices[0])
    
    # ]build prompt
    system_prompt = "You are a policy analyst. First, reason about which of the candidate codes apply to the paragraph. Then, conclude with a comma-separated list of the full, applicable code paths."
    user_prompt = f"Candidate Codes:\n---\n{context}---\n\nParagraph:\n---\n\"{paragraph}\"\n---"
    
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # generate response
    with torch.inference_mode():
        outputs = model.generate(
            **inputs, max_new_tokens=250, do_sample=False, pad_token_id=tokenizer.eos_token_id,
            eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|return|>")]
        )
    
    # parse codes out
    raw_output = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return parse_codes_from_text(raw_output, retrieved_paths)


def classify(PDF_DIRECTORY, RESULTS_FILE):
    classification_results = defaultdict(list)
    documents = load_pdfs_from_directory(PDF_DIRECTORY)
    
    for doc in documents:
        filename = doc['metadata']['source']
        print(f"\nProcessing {filename}...")
        
        # use the new safer chunking function
        chunks = split_into_chunks(doc['page_content'])
        print(f"Found {len(chunks)} chunks to classify.")
        
        for i, chunk in enumerate(chunks):
            predicted_codes = classify_paragraph_with_rag(chunk)
            
            paragraph_data = {
                "chunk_number": i + 1,
                "text_snippet": chunk,
                "predicted_codes": predicted_codes
            }
            classification_results[filename].append(paragraph_data)
            print(f"Chunk {i+1}: {predicted_codes}")

            # clearing GPU cache after each step
            torch.cuda.empty_cache()
            gc.collect()

    print(f"\nSaving results to {RESULTS_FILE}")
    with open(RESULTS_FILE, 'w') as f:
        json.dump(classification_results, f, indent=4)
    print("Done.")
