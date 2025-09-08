import os
import re
import glob
import numpy as np
import faiss
import torch
import sys
from pathlib import Path
from transformers import SentenceTransformer

def text_splitter(text, chunk_size=500, chunk_overlap=50):
    """
    Splits a text into overlapping chunks manually.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    chunks = []
    start_index = 0

    # loop to capture chunks with overlap
    while start_index < len(text):
        end_index = start_index + chunk_size

        chunk = text[start_index:end_index]
        chunks.append(chunk)
        
        start_index += chunk_size - chunk_overlap

    return chunks

def read_files(files_path):
    files_path = Path("/home/vicente/Github/BDLab-Agent/backend/data/kaggle/us-senate-bill")
    files_list = list(files_path.glob("*.txt"))
    total_files = len(files_list)

    # Read the .txt files and split them into chunks: [{file_id, title, chunk_id}]
    raw_chunks = []
    chunk_metadata = []
    file_titles = {}

    for path in files_list:
        text = path.read_text(encoding="utf-8")
        file_id = path.stem
        title = path.name
        file_titles[file_id] = title
    
        # split the entire document text with overlap
        document_chunks = text_splitter(text, chunk_size=500, chunk_overlap=50)

        for chunk_id, chunk in enumerate(document_chunks):

            # this adds enrichment to each chunk so the embedding captures a more complete context representation
            chunk_with_title = f"From the bill titled '{title}': {chunk}"
            raw_chunks.append(chunk_with_title)
            chunk_metadata.append({
                "file_id": file_id,
                "title": title,
                "chunk_id": chunk_id
            })
            
    print(f"Loaded {total_files} files, created {len(raw_chunks)} chunks.")
    embed_model = "data/embeddings/gte-large"
    embedder = SentenceTransformer(embed_model)

    # embed the chunks
    chunk_embs = embedder.encode(
        raw_chunks,
        convert_to_numpy=True,
        show_progress_bar=True
    ).astype(np.float32)

    # normalize for cosine similarity search
    faiss.normalize_L2(chunk_embs)

    dim = int(chunk_embs.shape[1])
    index = faiss.IndexFlatIP(dim)
    index.add(chunk_embs)

    return index, raw_chunks, chunk_metadata, file_titles, embedder