"""
Embedding and retrieval utilities.

Retrieval design
-----------------
We implement a **dense retrieval** pipeline using sentence-transformers
embeddings with cosine similarity scoring.

Model choice justification:
  ``all-MiniLM-L6-v2`` maps sentences to a 384-dimensional dense vector
  space.  It is small (~80 MB), fast, and achieves competitive results on
  semantic textual similarity benchmarks.  For a domain like Shakespeare
  where queries are often modern English and passages are Early Modern
  English, a general-purpose embedding model trained on diverse corpora
  provides a reasonable balance without requiring domain-specific fine-tuning.

We also provide an optional **TF-IDF sparse retrieval** component.  This
helps with keyword-heavy queries (e.g. character names, locations) where
lexical matching outperforms semantic similarity.

The final retriever can run in three modes:
  - ``dense``   : embedding cosine similarity only  (default)
  - ``sparse``  : TF-IDF cosine similarity only
  - ``hybrid``  : weighted combination of dense + sparse scores
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from config import INDEX_DIR


Chunk = Dict[str, Any]


class EmbeddingRetriever:
    """
    Embedding-based retriever with optional TF-IDF hybrid scoring.

    Parameters
    ----------
    embedding_model_name : str
        A sentence-transformers model identifier.
    mode : str
        One of 'dense', 'sparse', 'hybrid'.
    dense_weight : float
        Weight for dense scores in hybrid mode (sparse = 1 - dense_weight).
    """

    def __init__(
        self,
        embedding_model_name: str,
        mode: str = "dense",
        dense_weight: float = 0.7,
    ):
        self.mode = mode
        self.dense_weight = dense_weight
        self.embedding_model_name = embedding_model_name

        self.chunks: List[Chunk] = []
        self.embeddings: Optional[np.ndarray] = None
        self.tfidf_matrix = None
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self._model = None  # lazy-loaded

    # ── lazy model loading ────────────────────────────────────────────

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required. "
                    "Install with: pip install sentence-transformers"
                ) from exc
            print(f"Loading embedding model: {self.embedding_model_name} ...")
            self._model = SentenceTransformer(self.embedding_model_name)
            print("Embedding model loaded.")
        return self._model

    # ── index building ────────────────────────────────────────────────

    def build_index(self, chunks: List[Chunk]) -> None:
        """Create embeddings (and optionally TF-IDF vectors) for all chunks."""
        if not chunks:
            raise ValueError("No chunks supplied to build_index().")

        self.chunks = chunks
        texts = [chunk["text"] for chunk in chunks]

        # Dense embeddings
        if self.mode in ("dense", "hybrid"):
            print(f"Encoding {len(texts)} chunks with {self.embedding_model_name} ...")
            self.embeddings = np.asarray(
                self.model.encode(texts, show_progress_bar=True, batch_size=32)
            )
            print(f"Embeddings shape: {self.embeddings.shape}")

        # Sparse TF-IDF
        if self.mode in ("sparse", "hybrid"):
            print("Building TF-IDF index ...")
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=10000,
                stop_words="english",
                ngram_range=(1, 2),
            )
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            print(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")

    # ── retrieval ─────────────────────────────────────────────────────

    def retrieve(
        self, query: str, top_k: int = 5
    ) -> List[Tuple[Chunk, float]]:
        """
        Retrieve the top-k most relevant chunks for a query.

        Returns a list of (chunk, score) tuples sorted by descending score.
        """
        scores = np.zeros(len(self.chunks))

        # Dense scoring
        if self.mode in ("dense", "hybrid") and self.embeddings is not None:
            query_emb = np.asarray(self.model.encode([query]))
            dense_scores = cosine_similarity(query_emb, self.embeddings)[0]
            weight = self.dense_weight if self.mode == "hybrid" else 1.0
            scores += weight * dense_scores

        # Sparse scoring
        if self.mode in ("sparse", "hybrid") and self.tfidf_vectorizer is not None:
            query_tfidf = self.tfidf_vectorizer.transform([query])
            sparse_scores = cosine_similarity(query_tfidf, self.tfidf_matrix)[0]
            weight = (1 - self.dense_weight) if self.mode == "hybrid" else 1.0
            scores += weight * sparse_scores

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in top_indices]

    # ── persistence ───────────────────────────────────────────────────

    def save_index(self, path: Optional[Path] = None) -> Path:
        """Save the built index to disk for fast reload."""
        save_dir = path or INDEX_DIR
        save_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "chunks": self.chunks,
            "embeddings": self.embeddings,
            "mode": self.mode,
            "dense_weight": self.dense_weight,
            "embedding_model_name": self.embedding_model_name,
        }
        index_path = save_dir / "retrieval_index.pkl"
        with open(index_path, "wb") as fh:
            pickle.dump(data, fh)

        # Save TF-IDF separately if present
        if self.tfidf_vectorizer is not None:
            tfidf_path = save_dir / "tfidf_index.pkl"
            with open(tfidf_path, "wb") as fh:
                pickle.dump(
                    {"vectorizer": self.tfidf_vectorizer, "matrix": self.tfidf_matrix},
                    fh,
                )

        print(f"Index saved to {save_dir}")
        return save_dir

    def load_index(self, path: Optional[Path] = None) -> None:
        """Load a previously saved index."""
        load_dir = path or INDEX_DIR
        index_path = load_dir / "retrieval_index.pkl"

        if not index_path.exists():
            raise FileNotFoundError(f"No saved index at {index_path}")

        with open(index_path, "rb") as fh:
            data = pickle.load(fh)

        self.chunks = data["chunks"]
        self.embeddings = data.get("embeddings")
        self.mode = data.get("mode", self.mode)
        self.dense_weight = data.get("dense_weight", self.dense_weight)

        tfidf_path = load_dir / "tfidf_index.pkl"
        if tfidf_path.exists():
            with open(tfidf_path, "rb") as fh:
                tfidf_data = pickle.load(fh)
            self.tfidf_vectorizer = tfidf_data["vectorizer"]
            self.tfidf_matrix = tfidf_data["matrix"]

        print(f"Index loaded from {load_dir} ({len(self.chunks)} chunks)")


# ── CLI quick test ────────────────────────────────────────────────────

if __name__ == "__main__":
    from config import DEFAULT_TOP_K, EMBEDDING_MODEL_NAME
    from data_loader import load_all_scene_chunks
    from chunking import create_chunks, format_chunk_for_display

    records = load_all_scene_chunks()
    chunks = create_chunks(records)

    retriever = EmbeddingRetriever(EMBEDDING_MODEL_NAME, mode="dense")
    retriever.build_index(chunks)

    query = "Why does Macbeth kill Duncan?"
    results = retriever.retrieve(query, top_k=DEFAULT_TOP_K)

    print(f"\nQuery: {query}\n")
    for rank, (chunk, score) in enumerate(results, 1):
        print("=" * 80)
        print(f"Rank {rank} | Score: {score:.4f}")
        print(format_chunk_for_display(chunk, include_text=False))
