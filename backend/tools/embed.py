import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


def chunk_files(file_paths,codebook,coded_bills chunk_size=500, overlap=50):
    """
    Chunk all files into pieces with metadata.
    Returns:
        raw_chunks: List of text chunks.
        chunk_metadata: List of metadata dicts for each chunk.
        file_titles: Dict mapping file_id to title.
    """
    raw_chunks = []
    chunk_metadata = []
    file_titles = {}

    for index,row in codebook.iterrows():
        sub_category = row['Code Subcategory']
        code_name = row['Code Name']
        code_definition = row['Definition']


    for path in file_paths:
        text = path.read_text(encoding="utf-8")
        file_id = path.stem
        title = path.name
        file_titles[file_id] = title

        chunk_id = 0
        parts = text.split("\n\n")
        for para in parts:
            s = para.strip()
            if len(s) > chunk_size:
                
                # Prepend the title to each chunk's text
                #chunk_with_title = f"From the bill titled '{title}':\n{s}"
                chunk_with
                raw_chunks.append(chunk_with_title)
                # ---------------------
                
                chunk_metadata.append({
                    "file_id": file_id,
                    "title": title,
                    "chunk_id": chunk_id
                })
                chunk_id += 1

    assert len(raw_chunks) == len(chunk_metadata), "chunk_metadata misaligned with raw_chunks"

    #print(f"Loaded {NUM_FILES} files, {len(raw_chunks)} chunks")
    return raw_chunks, chunk_metadata, file_titles

def manual_text_splitter(text, chunk_size=1000, chunk_overlap=100):
    """
    Splits a text into overlapping chunks manually.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    chunks = []
    start_index = 0

    while start_index < len(text):
        end_index = start_index + chunk_size
        
        chunk = text[start_index:end_index]
        chunks.append(chunk)
        
        start_index += chunk_size - chunk_overlap

    return chunks



# --- Example Usage ---

# # Let's create a sample text that is longer than 1000 characters
# sample_text = "A" * 500 + "B" * 500 + "C" * 500 + "D" * 200
# print(f"Total length of text: {len(sample_text)}\n")

# # Use our manual splitter
# my_chunks = manual_text_splitter(
#     text=sample_text,
#     chunk_size=1000,
#     chunk_overlap=100
# )


def embed_chunks(model, raw_chunks, batch_size=16):
    model_path = "data/embeddings/gte-large"
    embedder = SentenceTransformer(model_path)

    # NumPy, not tensor
    chunk_embs = embedder.encode(
        raw_chunks,
        convert_to_numpy=True,
        show_progress_bar=True
    ).astype(np.float32)

    faiss.normalize_L2(chunk_embs)  # cosine w/ IP index

    dim = int(chunk_embs.shape[1])
    index = faiss.IndexFlatIP(dim)
    index.add(chunk_embs)

    return embedder, index