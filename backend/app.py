import os
import re
import glob
import faiss
import torch
import sys
from pathlib import Path
from config import DevelopmentConfig, ProductionConfig, ModelConfig
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import json
import gc
from collections import defaultdict
from backend.tools.embed import read_files
from backend.models.chat_agents import load_llm, chat_oss
from backend.tools.retrieval import retrieve_context, ask_rag
from backend.tools.vector_db import IVFCosineFAISS
from backend.tools.read_policy import load_pdfs_from_directory, build_faiss_index, flatten_knowledge_tree, split_into_chunks,classify_paragraph_with_rag

app = Flask(__name__)

# If FLASK_ENV or FLASK_DEBUG tells us we’re in dev mode, use DevelopmentConfig
if os.environ.get("FLASK_ENV") == "development" or os.environ.get("FLASK_DEBUG") == "1":
    app.config.from_object(DevelopmentConfig)
    print("Loaded DevelopmentConfig")
else:
    app.config.from_object(ProductionConfig)
    print("Loaded ProductionConfig")

# Now safe to reference CORS_ORIGINS, with a fallback to empty list
CORS(app, resources={r"/api/*": {"origins": app.config.get("CORS_ORIGINS", [])}})

PDF_DIRECTORY = "/home/vicente/Github/BDLab-Agent/backend/utils/uncoded"
RESULTS_FILE = "classification_results.json"

emd_model = "/home/vicente/Github/BDLab-Agent/backend/data/embeddings/bge-large"
embedder = SentenceTransformer(emd_model, device='cuda')

base = Path("/home/vicente/Github/BDLab-Agent/backend/data/LLMs/gpt-oss-20b")
model = AutoModelForCausalLM.from_pretrained(str(base),dtype=torch.bfloat16,device_map="auto",local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(str(base),local_files_only=True)

with open('knowlesge_tree.json','r') as f:
    knowledge_tree = json.load(f)

print("Loaded knowledge tree")
flattened_items = flatten_knowledge_tree(knowledge_tree)
index, clean_code_paths, embedder = build_faiss_index(flattened_items, embedder)

# uncoded pdfs
classification_results = defaultdict(list)
documents = load_pdfs_from_directory(PDF_DIRECTORY)

for doc in documents:
    filename = doc['metadata']['source']
    print(f"\nProcessing {filename}...")
    
    # use the new safer chunking function
    chunks = split_into_chunks(doc['page_content'])
    print(f"Found {len(chunks)} chunks to classify.")
    
    for i, chunk in enumerate(chunks):
        predicted_codes = classify_paragraph_with_rag(chunk, index, clean_code_paths, embedder, model, tokenizer)
        
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

print("Model and tokenizer loaded successfully.")
store = IVFCosineFAISS(index_key="IVF4096,PQ64", nprobe=16)


# Serving after restarting the app:
serve = IVFCosineFAISS(index_key="IVF4096,PQ64", nprobe=16)
serve.load_mmap("index.faiss")
serve.load_corpus("chunks.jsonl")
ctx, metas, D, I = serve.search("What is FAISS IVF?", k=3)

# Appending later:
w = IVFCosineFAISS(index_key="IVF4096,PQ64", nprobe=16)
w.load_writable("index.faiss")
w.load_corpus("chunks.jsonl")
w.add(["New chunk about cosine"], [{"doc_id": len(w.raw_chunks)}])
w.save("index.faiss")
# keep sidecar in sync
w.save_corpus("chunks.jsonl")

   
@app.route("/api/qb", methods=["POST"])
def qb():
    payload = request.get_json() or {}
    question = payload.get("question", "").strip()
    if not question:
        return jsonify({"error": "Please provide a question."}), 400

    try:
        answer_text, sources_list = ask_rag(question)

        response_payload = {
            "answer": answer_text,
            "sources": sources_list
        }
        
        return jsonify(response_payload)
        
    except Exception as e:
        print(f"An error occurred in the RAG pipeline: {e}", file=sys.stderr)
        return jsonify({"error": "Sorry, something went wrong on our end."}), 500



