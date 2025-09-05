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
from collections import defaultdict
#from tools.embed import chunk_files, embed_chunks
#from tools.retrieval import chat_oss, ask_rag, retrieve_context

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

# Load your documents using model configuration 
# Local model configuration 
files_path = Path("/home/vicente/Github/BDLab-Agent/backend/data/kaggle/us-senate-bill")
files_list = list(files_path.glob("*.txt"))
total_files = len(files_list)


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

base = Path("/home/vicente/Github/BDLab-Agent/backend/data/GPTModels/gpt-oss-20b")
model = AutoModelForCausalLM.from_pretrained(str(base),dtype=torch.bfloat16,device_map="auto",local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(str(base),local_files_only=True)
model.eval()
def chat_oss(user_prompt, system_prompt=None, max_new_tokens=512, do_sample=True, temperature=0.1):
    """
    Core function to generate a response from the LLM without external context.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature

    with torch.inference_mode():
        outputs = model.generate(**inputs, **gen_kwargs)

    prompt_length = inputs["input_ids"].shape[1]
    raw_output = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
    
    clean_response = raw_output.split("assistantfinal")[-1].strip()
    if clean_response.startswith("analysis"):
        clean_response = clean_response[len("analysis"):].strip()
        
    return clean_response

# # A simple system prompt to keep the model helpful.
# base_system_prompt = "You are a helpful assistant."

# question = "What were the key findings of Project Minerva according to the final report?"

# # Calling the model without any external context.
# response = chat_oss(
#     user_prompt=question,
#     system_prompt=base_system_prompt,
#     do_sample=False
# )

# print(f"Question: {question}\n")
# print(f"Model's Answer (Without RAG): \n{response}")

def retrieve_context(query, k=3):
    """
    Retrieves the top-k most relevant chunks from the FAISS index for a given query.
    """
    print(f"Retrieving context for query: '{query}'")

    # embed the query
    query_emb = embedder.encode([query], convert_to_numpy=True).astype(np.float32)

    # normalize the query embedding (for cosine similarity)
    faiss.normalize_L2(query_emb)

    # search the FAISS index
    distances, indices = index.search(query_emb, k)

    # fetch the actual text chunks using the indices
    retrieve_chunks_text = []
    for i in indices[0]:
        retrieve_chunks_text.append(raw_chunks[i])

    retrieved_chunks_meta = []
    for i in indices[0]:
        retrieved_chunks_meta.append(chunk_metadata[i])

    # we combine single context string
    context = "\n\n---\n\n".join(retrieve_chunks_text)

    print("Context retrieved successfully.")

    # Return both the context string AND the list of metadata dictionaries
    return context, retrieved_chunks_meta


def ask_rag(query):
    """
    The complete RAG pipeline.
    Retrieves context, builds a prompt, and generates an answer with sources.
    """
    # First retrieve context
    retrieved_context, sources = retrieve_context(query, k=3)

    # Now we create the RAG prompt:
    # This Combine the context and query into a single prompt for the LLM, (instructing it on how to behave)
    augmented_prompt  = """
        You are a helpful assistant for answering questions about US Senate bills and Acts.
        Use the following context to answer the user's question.
        If the answer is not found in the context, state that you cannot find the answer in the provided documents.
        Do not use any external knowledge or make up information.

        (START CONTEXT): {context} (END CONTEXT).

        USER QUESTION: {question} """.strip()

    # based on the prompt template, we create the final prompt text passing in the retrieved context and user question
    final_prompt_text = augmented_prompt.format(context=retrieved_context, question=query)

    print("\nGENERATING RESPONSE:\n")

    # passing the fully formatted RAG prompt as the "user_prompt"
    response = chat_oss(final_prompt_text, max_new_tokens=512, do_sample=False)

    # print("SOURCES USED:\n")
    # for i, meta in enumerate(sources):
    #     print(f"Source {i+1}: {meta['title']} (Chunk ID: {meta['chunk_id']})")
    
    return response, sources


# # Example 1:
# response, sources = ask_rag("Who is Joanne Chesimard and what did she do?")
# print("\n\nFINAL ANSWER:\n")
# print(response)
   
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

if __name__ == "__main__":
    app.run(host="
# TO RUN FRONT END:
# one terminal
# `firebase emulators:start`
# second tgerminal 
# `npm run dev`