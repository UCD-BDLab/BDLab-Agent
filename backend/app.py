import os
import re
import glob
import faiss
import torch
from pathlib import Path
from config import DevelopmentConfig, ProductionConfig, ModelConfig
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

app = Flask(__name__)

# load the model configuration
app.config.from_object(ModelConfig)

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
DATA_FOLDER = ModelConfig.PROJECT_ROOT / "data" / "kaggle" / "us-senate-bill"
FILE_PATHS = list(DATA_FOLDER.glob("*.txt"))
NUM_FILES = len(FILE_PATHS)

# Read the .txt files and split them into chunks
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
EMBEDDING_MODEL_PATH = "data/embeddings/gte-large"
embedder = SentenceTransformer(EMBEDDING_MODEL_PATH)
chunk_embs = embedder.encode(raw_chunks, convert_to_numpy=True, show_progress_bar=True).astype(np.float32)

dim = chunk_embs.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(chunk_embs)

base = Path(app.config["MODEL_BASE_PATH"])
model = AutoModelForCausalLM.from_pretrained(str(base),torch_dtype=torch.bfloat16,device_map="auto",local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(str(base),local_files_only=True)
model.eval()


# if using GPT2, set pad_token_id to eos_token_id, otherwise coment it out and uncomment the lines above for Llama3
#model.config.pad_token_id = model.config.eos_token_id
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
if model.config.pad_token_id is None:
    model.config.pad_token_id = tokenizer.pad_token_id

tokenizer.padding_side = "left"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

BULLET_PREFIX = r'(?:[-*]\s+|\(?\d{1,3}[.)]\)?\s+)'

def extract_bullets(txt: str):
    txt = txt.replace('\r\n', '\n').strip()
    pattern = rf'(^\s*{BULLET_PREFIX}.+?)(?=\n\s*(?:[-*]|\(?\d{{1,3}}[.)]\)?)\s+|$)'

    matches = re.findall(pattern, txt, flags=re.M | re.S)

    items = []
    for m in matches:
        cleaned = re.sub(rf'^\s*{BULLET_PREFIX}', '', m).strip()
        if cleaned:
            items.append(cleaned)

    return items

# # chat wrapper
# def chat(prompt: str, max_new_tokens: int = 200) -> str:
#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
#     outputs = model.generate(inputs.input_ids,attention_mask=inputs.attention_mask,max_new_tokens=max_new_tokens,do_sample=True,temperature=0.7)
#     text = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     cleaned_text = text[len(prompt):].strip().split("\n\n")[0]

#     return cleaned_text

# def chat(prompt: str, max_new_tokens: int = 300):
#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
#     outputs = model.generate(
#         inputs.input_ids,
#         attention_mask=inputs.attention_mask,
#         max_new_tokens=max_new_tokens,
#         do_sample=True,
#         temperature=0.7,
#         top_p=0.95,
#         eos_token_id=getattr(tokenizer, "eos_token_id", None),
#         pad_token_id=getattr(tokenizer, "pad_token_id", getattr(tokenizer, "eos_token_id", None)),
#     )
#     text = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     completion = text[len(prompt):].strip()
#     return completion

def chat(messages, max_new_tokens: int = 300) -> str:
    """
    Renders via chat_template.jinja and returns only the completion (no prompt echo or leftovers)
    messages: List of dicts with "role" and "content" keys
    """
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(rendered, return_tensors="pt")
    for k in list(inputs.keys()):
        v = inputs[k]
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
            pad_token_id=tokenizer.pad_token_id,
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    completion = text[len(rendered):].strip()
    return completion


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
    # metadata_lines = []
    # metadata_lines.append(f"You have ingested {NUM_FILES} text files:")
    # for fp in FILE_PATHS:
    #     metadata_lines.append(f"- {Path(fp).name}")

    #metadata_block = "\n".join(metadata_lines)
    #passages_block = "\n\n".join(passages)

    SYSTEM = (
    "You are a helpful assistant. "
    "Answer the user's question directly in natural language. "
    "Use the provided context only if it helps, but do NOT mention or quote the context, "
    "files, retrieval steps, or anything about how you got the information."
    )

    context = "\n\n".join(passages)
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "system", "content": f"Context:\n{context}"},
        {"role": "user", "content": question},
    ]

    #prompt = (
    #     f"{SYSTEM}\n\n"
    #     f"Context:\n{context}\n\n"
    #     f"User question: {question}\n"
    #     f"Assistant answer:"
    # )
    #We combine the components above to assemble the full promp
    # prompt = f"""
    # {metadata_block}You are a helpful assistant. Use ONLY the following passages:\n
    # Retrieved Passages: {passages_block}.\n
    # Question: {question}.\n
    # Answer:"""

    # prompt = f"""
    #     Please answer in concise bullet points. Start each point with "- ".
    #     {metadata_block}
    #     Use ONLY the following passages:
    #     {passages_block}
    #     Question: {question}
    #     Answer (bullet points):
    # """

    # Sedingfing it to the model
    try:
        answer = chat(messages, max_new_tokens=300)
        return jsonify({"answer": answer})
        #answer = chat(prompt)
        #bullets = extract_bullets(answer)
        #return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
