from config import DevelopmentConfig, ProductionConfig
from flask import Flask, request, jsonify
from flask_cors import CORS
import os, glob, faiss, torch
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

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

# Load your documents & chunk them
DATA_FOLDER = Path(__file__).parent / "data" / "kaggle" / "us-senate-bill"
FILE_PATHS = list(DATA_FOLDER.glob("*.txt"))
NUM_FILES = len(FILE_PATHS)

raw_chunks = []
for path in FILE_PATHS:
    text = path.read_text(encoding="utf-8")
    for para in text.split("\n\n"):
        s = para.strip()
        if len(s) > 200:
            raw_chunks.append(s)

# Here we use a sentence tranformer to build the embeddings (text to numeric vectors)
# FAISS is used as a vector database to index those embeddings
# This gives us the ability to search for similar chunks of text
# Not quite a database, this vector index lives in memory
embedder = SentenceTransformer("all-MiniLM-L6-v2")
chunk_embs = embedder.encode(raw_chunks, convert_to_numpy=True, show_progress_bar=True)

dim = chunk_embs.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(chunk_embs)

#BASE = "/home/vicente/Github/BDLab-Agent/backend/data/models/llama3-8b-instruct/models--meta-llama--Meta-Llama-3-8B-Instruct"
#SNAP = "8afb486c1db24fe5011ec46dfbe5b5dccdb575c2"
BASE = "/home/vicente/Github/BDLab-Agent/backend/data/models/gpt2-large/models--gpt2-large"
SNAP = "32b71b12589c2f8d625668d2335a01cac3249519"
model = AutoModelForCausalLM.from_pretrained(f"{BASE}/snapshots/{SNAP}",torch_dtype=torch.bfloat16,device_map="auto",local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(f"{BASE}/snapshots/{SNAP}",local_files_only=True)

# if using GPT2, set pad_token_id to eos_token_id, otherwise coment it out and uncomment the lines above for Llama3
model.config.pad_token_id = model.config.eos_token_id

# chat wrapper
def chat(prompt: str, max_new_tokens: int = 200) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(inputs.input_ids,attention_mask=inputs.attention_mask,max_new_tokens=max_new_tokens,do_sample=True,temperature=0.7)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    cleaned_text = text[len(prompt):].strip().split("\n\n")[0]

    return cleaned_text

@app.route("/api/qb", methods=["POST"])
def qb():
    payload = request.get_json() or {}
    question = payload.get("question", "").strip()
    if not question:
        return jsonify({"error": "Please provide a question."}), 400

    # we first embed the question from the user
    q_emb = embedder.encode([question], convert_to_numpy=True)

    # then we retrieve top 5 chunks
    distances, indices = index.search(q_emb, 5)
    passages = []
    for idx in indices[0]:
        passages.append(raw_chunks[idx])

    # metadata lines to add context
    metadata_lines = []
    metadata_lines.append(f"You have ingested {NUM_FILES} text files:")
    for fp in FILE_PATHS:
        metadata_lines.append(f"- {Path(fp).name}")

    metadata_block = "\n".join(metadata_lines)
    passages_block = "\n\n".join(passages)

    # We combine the components above to assemble the full promp
    prompt = f"""
    {metadata_block}You are a helpful assistant. Use ONLY the following passages:\n
    Retrieved Passages: {passages_block}.\n
    Question: {question}.\n
    Answer:"""

    # Sedingfing it to the model
    try:
        answer = chat(prompt)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
