import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class IVFCosineFAISS:
    """FAISS IVF+PQ index wrapper using cosine similarity via L2-normalized embeddings.

    This class builds and queries a FAISS index (e.g., `IVF4096,PQ64`) over
    SentenceTransformers embeddings. Cosine similarity is implemented by
    L2-normalizing vectors and using inner product search.

    Attributes:
        embedder: SentenceTransformer model used to embed texts.
        index_key: FAISS index factory string (e.g., "IVF4096,PQ64").
        nprobe: Number of inverted lists to probe during search (recall/speed knob).
        index: The underlying FAISS index (wrapped in IndexIDMap2).
        raw_chunks: List of raw text chunks in the corpus.
        chunk_metadata: List of per-chunk metadata dicts aligned with `raw_chunks`.
        dim: Embedding dimensionality (set at build time).

    Example:
        ivf = IVFCosineFAISS(model_name="data/embeddings/gte-large", index_key="IVF4096,PQ64", nprobe=16)
        ivf.build(raw_chunks=["a", "b", "c"])
        context, metas, scores, idxs = ivf.search("query", k=2)
    """

    def __init__(self, model_name="data/embeddings/gte-large",
                 index_key="IVF4096,PQ64", nprobe=16):
        """Initialize the index wrapper.

        Args:
            model_name: Path or Hugging Face ID for the SentenceTransformer model.
            index_key: FAISS index factory spec (e.g., "IVF4096,PQ64").
            nprobe: FAISS `nprobe` used at search time.

        Notes:
            - Cosine similarity is achieved by L2-normalizing embeddings and using FAISS with `METRIC_INNER_PRODUCT`.
            - The index is not created until `build()` or `load_*()` is called.
        """
        self.embedder = SentenceTransformer(model_name)
        self.index_key = index_key
        self.nprobe = int(nprobe)
        self.index = None
        self.raw_chunks = []
        self.chunk_metadata = []
        self.dim = None

    # ---------- embeddings ----------
    def _embed_norm(self, texts):
        """Embed texts and L2-normalize vectors for cosine-sim via inner product.

        Args:
            texts: Iterable of strings to embed.

        Returns:
            np.ndarray: Float32 array of shape (n, dim), L2-normalized.

        Raises:
            ValueError: If `texts` is empty.
        """
        embs = self.embedder.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False
        ).astype(np.float32)
        faiss.normalize_L2(embs)  # cosine via IP
        return embs

    def build(self, raw_chunks, chunk_metadata=None, train_sample_size=100_000):
        """Build and populate the FAISS index from raw text chunks.

        Trains an IVF/PQ index on a sample of the corpus, then adds all vectors
        with stable integer IDs matching their position in `raw_chunks`.

        Args:
            raw_chunks: Iterable of text chunks to index.
            chunk_metadata: Iterable of metadata dicts aligned with `raw_chunks`. If None, fills with empty dicts.
            train_sample_size: Max number of chunks used to train the coarse quantizer.

        Side Effects:
            - Sets `self.index`, `self.raw_chunks`, `self.chunk_metadata`, and `self.dim`.

        Raises:
            ValueError: If `raw_chunks` is empty.
            RuntimeError: If FAISS training fails.
        """
        self.raw_chunks = list(raw_chunks)
        if chunk_metadata is not None:
            self.chunk_metadata = list(chunk_metadata)
        else:
            for _ in raw_chunks:
                self.chunk_metadata.append({})

        train_texts = self.raw_chunks[:min(train_sample_size, len(self.raw_chunks))]
        train_vecs = self._embed_norm(train_texts)

        self.dim = int(train_vecs.shape[1])
        base = faiss.index_factory(self.dim, self.index_key, faiss.METRIC_INNER_PRODUCT)
        base.train(train_vecs)

        base = faiss.IndexIDMap2(base)
        all_vecs = self._embed_norm(self.raw_chunks)
        ids = np.arange(len(self.raw_chunks), dtype=np.int64)
        base.add_with_ids(all_vecs, ids)

        if hasattr(base, "nprobe"):
            base.nprobe = self.nprobe
        self.index = base


    def save_corpus(self, path="chunks.jsonl"):
        """Save texts and metadata to a JSONL sidecar file.

        Each line is a JSON object with keys:
            - "text": str
            - "meta": dict

        Args:
            path: Output path for the JSONL file.
        """
        with open(path, "w", encoding="utf-8") as f:
            for t, m in zip(self.raw_chunks, self.chunk_metadata):
                f.write(json.dumps({"text": t, "meta": m}, ensure_ascii=False) + "\n")

    def load_corpus(self, path="chunks.jsonl"):
        """Load texts and metadata from a JSONL sidecar file.

        The file must contain one JSON object per line with "text" and optional "meta".

        Args:
            path: Input path to the JSONL file.

        Side Effects:
            - Sets `self.raw_chunks` and `self.chunk_metadata`.
        """
        texts, metas = [], []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                texts.append(obj["text"])
                metas.append(obj.get("meta", {}))
        self.raw_chunks = texts
        self.chunk_metadata = metas

    def save(self, path: str):
        """Persist the FAISS index to disk.

        Args:
            path: Output path for the FAISS index file.

        Raises:
            AssertionError: If `self.index` is None.
        """
        assert self.index is not None, "No index to save. Build or load one first."
        faiss.write_index(self.index, path)

    def load_mmap(self, path: str, nprobe: int = None):
        """Memory-map a read-only FAISS index from disk.

        Args:
            path: Path to an index file saved with `save()`.
            nprobe: Optional override for search-time `nprobe`.

        Side Effects:
            - Sets `self.index` to a read-only, memory-mapped index.
            - Updates `self.index.nprobe` if supported.

        Notes:
            - Read-only mmap indexes cannot be modified via `add()`.
        """
        flags = faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY
        self.index = faiss.read_index(path, flags)
        if hasattr(self.index, "nprobe"):
            self.index.nprobe = self.nprobe if nprobe is None else int(nprobe)

    def load_writable(self, path: str):
        """Load a writable FAISS index fully into RAM.

        Args:
            path: Path to an index file saved with `save()`.

        Side Effects:
            - Sets `self.index` to a writable index in memory.
            - Updates `self.index.nprobe` if supported.
        """
        # loads it fully into RAM so we can modify it
        self.index = faiss.read_index(path)
        if hasattr(self.index, "nprobe"):
            self.index.nprobe = self.nprobe

    def add(self, new_chunks, new_metadata=None):
        """Append new chunks (and metadata) to the existing index.

        Args:
            new_chunks: Iterable of new text chunks to embed and add.
            new_metadata: Iterable of metadata dicts aligned with `new_chunks`.
                If None, fills with empty dicts.

        Returns:
            list[int]: The integer IDs assigned to the added chunks.

        Raises:
            AssertionError: If `self.index` is None (build or load first).
            RuntimeError: If the loaded index is read-only (e.g., mmapped).
        """
        assert self.index is not None, "Load or build the index first."
        # quick check: memory-mapped read-only isdexes cannot be modified
        try:
            # touch C++ object to surface RO issues early
            _ = self.index.is_trained
        except Exception as e:
            raise RuntimeError("Index appears read-only. Load with load_writable() before add().") from e

        start = len(self.raw_chunks)
        new_chunks = list(new_chunks)
        self.raw_chunks.extend(new_chunks)

        if new_metadata is None:
            new_metadata = []
            for _ in new_chunks:
                new_metadata.append({})
        else:
            new_metadata = list(new_metadata)
        self.chunk_metadata.extend(new_metadata)

        vecs = self._embed_norm(new_chunks)
        ids = np.arange(start, start + len(new_chunks), dtype=np.int64)
        self.index.add_with_ids(vecs, ids)
        return list(ids)

    def search(self, query: str, k: int = 3):
        """Search the index with a text query.

        Args:
            query: Natural language query string.
            k: Number of nearest neighbors to return.

        Returns:
            tuple:
                - context (str): Top-k chunk texts joined by separators.
                - metas (list[dict]): Metadata dicts for the hits.
                - scores (list[float]): FAISS inner-product scores (cosine, due to normalization).
                - indices (list[int]): Corpus indices of the hits.

        Notes:
            - If the index returns fewer than `k` valid results, FAISS may include `-1` indices. 
            - Consider filtering those before dereferencing if you expect sparse matches.
        """
        q = self._embed_norm([query])
        D, I = self.index.search(q, k)
        hit_idxs = I[0].tolist()
        texts= []
        for i in hit_idxs:
            texts.append(self.raw_chunks[i])

        metas = []
        for i in hit_idxs:
            metas.append(self.chunk_metadata[i])

        context = "\n\n---\n\n".join(texts)
        return context, metas, D[0].tolist(), hit_idxs
