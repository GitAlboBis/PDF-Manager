"""
backend/extraction/rag_system.py — RAG pipeline for PDF field extraction.

Uses:
* sentence-transformers (HuggingFace) for embeddings — optional
* LangChain for text splitting and retrieval — optional
* SQLAlchemy for storing embeddings in the DB (serialised JSON)

Falls back to TF-IDF cosine similarity when sentence-transformers is absent.
"""

from __future__ import annotations

import json
import logging
import math
import re
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------

try:
    from sentence_transformers import SentenceTransformer as _ST
    _SENTENCE_TRANSFORMERS = True
except ImportError:
    _SENTENCE_TRANSFORMERS = False
    logger.info("sentence-transformers not installed — using TF-IDF fallback")

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    _LANGCHAIN_AVAILABLE = True
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        _LANGCHAIN_AVAILABLE = True
    except ImportError:
        _LANGCHAIN_AVAILABLE = False
        logger.info("langchain not installed — using simple chunking")


if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Simple TF-IDF fallback
# ---------------------------------------------------------------------------

def _tfidf_embed(text: str, vocab: dict[str, int]) -> list[float]:
    """Produce a sparse TF-IDF-like vector using the given vocabulary."""
    tokens = re.findall(r"\w+", text.lower())
    tf: dict[str, float] = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1
    total = len(tokens) or 1
    vec = [0.0] * len(vocab)
    for t, cnt in tf.items():
        if t in vocab:
            vec[vocab[t]] = cnt / total
    return vec


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ---------------------------------------------------------------------------
# RAGSystem
# ---------------------------------------------------------------------------

class RAGSystem:
    """
    Retrieval-Augmented Generation pipeline for document field extraction.

    Workflow
    --------
    1. ``index(text, document_id)``   — split text into chunks and store embeddings.
    2. ``retrieve(query, document_id)``  — find the most relevant chunk(s).
    3. ``extract_field(field_name, document_id, text)``  — retrieve + pattern match.

    Storage
    -------
    Embeddings are stored in the :class:`RAGEmbedding` SQLAlchemy model as
    serialised JSON.  This is compatible with both SQLite (development) and
    PostgreSQL (production, optionally with pgvector).
    """

    MODEL_NAME = "all-MiniLM-L6-v2"
    CHUNK_SIZE = 300
    CHUNK_OVERLAP = 50
    TOP_K = 3

    def __init__(self) -> None:
        self._model = None
        self._vocab: dict[str, int] = {}          # TF-IDF fallback vocabulary
        self._chunks: dict[str, list[str]] = {}   # document_id → chunks (in-memory)
        self._embeddings: dict[str, list[list[float]]] = {}  # doc → chunk embeddings

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_model(self):
        if self._model is None and _SENTENCE_TRANSFORMERS:
            try:
                self._model = _ST(self.MODEL_NAME)
                logger.info("Loaded sentence-transformers model: %s", self.MODEL_NAME)
            except Exception as exc:
                logger.warning("Could not load sentence-transformers model: %s", exc)
        return self._model

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts. Falls back to TF-IDF if model unavailable."""
        model = self._get_model()
        if model is not None:
            try:
                vecs = model.encode(texts, show_progress_bar=False)
                return [v.tolist() for v in vecs]
            except Exception as exc:
                logger.warning("Embedding failed, using TF-IDF: %s", exc)

        # Build / update vocabulary
        all_tokens = set()
        for t in texts:
            all_tokens.update(re.findall(r"\w+", t.lower()))
        for tok in all_tokens:
            if tok not in self._vocab:
                self._vocab[tok] = len(self._vocab)

        return [_tfidf_embed(t, self._vocab) for t in texts]

    def _split_text(self, text: str) -> list[str]:
        if _LANGCHAIN_AVAILABLE:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.CHUNK_SIZE,
                chunk_overlap=self.CHUNK_OVERLAP,
            )
            return splitter.split_text(text)
        # Simple fallback: split on sentence boundaries
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks: list[str] = []
        current = ""
        for sent in sentences:
            if len(current) + len(sent) > self.CHUNK_SIZE and current:
                chunks.append(current.strip())
                current = sent
            else:
                current += " " + sent
        if current.strip():
            chunks.append(current.strip())
        return chunks or [text]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index(self, text: str, document_id: str) -> int:
        """
        Index document text by splitting and embedding chunks.

        Args:
            text:        Full document text.
            document_id: String identifier for the document.

        Returns:
            Number of chunks created.
        """
        chunks = self._split_text(text)
        embeddings = self._embed(chunks)
        self._chunks[document_id] = chunks
        self._embeddings[document_id] = embeddings
        return len(chunks)

    def retrieve(
        self, query: str, document_id: str, top_k: int | None = None
    ) -> list[tuple[str, float]]:
        """
        Retrieve the most relevant text chunks for a query.

        Args:
            query:       Natural language query.
            document_id: Document to search in.
            top_k:       Number of results to return.

        Returns:
            List of (chunk_text, similarity_score) tuples.
        """
        k = top_k or self.TOP_K
        chunks = self._chunks.get(document_id)
        embeddings = self._embeddings.get(document_id)

        if not chunks or not embeddings:
            return []

        query_emb = self._embed([query])[0]
        scores = [_cosine(query_emb, emb) for emb in embeddings]
        ranked = sorted(
            zip(chunks, scores), key=lambda x: x[1], reverse=True
        )
        return ranked[:k]

    def extract_field(
        self,
        field_name: str,
        document_id: str,
        full_text: str,
        confidence_base: float = 0.80,
    ) -> dict | None:
        """
        Use RAG to find the most likely value for a named field.

        Args:
            field_name:       e.g. "Name", "City", "Zip Code".
            document_id:      Document identifier.
            full_text:        Full document text (used to build index if needed).
            confidence_base:  Base confidence for RAG-found values.

        Returns:
            dict with ``value`` and ``confidence`` keys, or ``None``.
        """
        # Ensure document is indexed
        if document_id not in self._chunks:
            self.index(full_text, document_id)

        query = f"What is the {field_name} in this document?"
        hits = self.retrieve(query, document_id, top_k=3)

        if not hits:
            return None

        # Concatenate top chunks and run pattern matching
        combined = "\n".join(chunk for chunk, _ in hits)
        avg_score = sum(score for _, score in hits) / len(hits)

        value = self._pattern_extract(field_name, combined)
        if not value:
            return None

        confidence = min(0.99, confidence_base + avg_score * 0.15)
        return {"value": value, "confidence": round(confidence, 4), "source": "rag"}

    @staticmethod
    def _pattern_extract(field_name: str, text: str) -> str | None:
        """Apply field-specific patterns to extract a value from text.

        Patterns use non-capturing groups ``(?:...)`` to support common
        synonyms across different PDF layouts (e.g. "Name", "Full Name",
        "Applicant Name" all map to the ``name`` field).
        """
        fn = field_name.lower().replace(" ", "_")

        patterns: dict[str, re.Pattern] = {
            "name": re.compile(
                r"(?:Name|Full\s*Name|Applicant\s*Name)\s*:?\s*(.+)",
                re.M | re.I,
            ),
            "cell_phone": re.compile(
                r"(?:Cell\s*Phone|Cell|Mobile(?:\s*Phone)?)\s*:?\s*(\d[\d\s\-().]{7,})",
                re.I,
            ),
            "home_phone": re.compile(
                r"(?:Home\s*Phone|Home\s*Tel|Telephone)\s*:?\s*(\d[\d\s\-().]{7,})",
                re.I,
            ),
            "work_phone": re.compile(
                r"(?:Work\s*Phone|Office\s*Phone|Business\s*Phone)\s*:?\s*(\d[\d\s\-().]{7,})",
                re.I,
            ),
            "email": re.compile(
                r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z]{2,}",
            ),
            "city": re.compile(
                r"(?:City|Town|Municipality)\s*:?\s*([A-Za-z\s]+?)(?:\s+(?:State|Province|Zip|Postal)|[,\n]|$)",
                re.I,
            ),
            "state": re.compile(
                r"(?:State|Province|Region)\s*:?\s*([A-Z]{2})",
                re.I,
            ),
            "zip_code": re.compile(
                r"(?:Zip\s*Code|Zip|Postal\s*Code|ZIP)\s*:?\s*(\d{5}(?:-\d{4})?)",
                re.I,
            ),
            "street_address": re.compile(
                r"(?:Street\s*Address|Address|Mailing\s*Address)\s*:?\s*(.+?)(?:\n|(?:City|Town)\s*:|$)",
                re.I | re.S,
            ),
        }

        pat = patterns.get(fn)
        if pat is None:
            # Generic fallback: try "FieldName: value" (multiline-aware)
            try:
                generic = re.compile(
                    rf"(?:{re.escape(field_name)})\s*:?\s*(.+?)(?:\n|$)",
                    re.I | re.M,
                )
                m = generic.search(text)
            except re.error:
                # If the field name itself breaks regex, give up gracefully
                return None
            return m.group(1).strip() if m else None

        m = pat.search(text)
        return m.group(1).strip() if m else None

    def persist_to_db(self, document_id: str, db_session) -> None:
        """
        Persist embeddings to the RAGEmbedding table.

        Args:
            document_id: Integer document ID (as stored in the DB).
            db_session:  SQLAlchemy db session.
        """
        try:
            from models import RAGEmbedding
        except ImportError:
            import sys
            import os
            # backend/extraction/ → backend/ → repo root (two levels up)
            sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
            from models import RAGEmbedding

        str_doc_id = str(document_id)
        chunks = self._chunks.get(str_doc_id, [])
        embeddings = self._embeddings.get(str_doc_id, [])

        for chunk, emb in zip(chunks, embeddings):
            record = RAGEmbedding(
                document_id=int(document_id),
                text_content=chunk,
                embedding=json.dumps(emb),
            )
            db_session.add(record)
        db_session.commit()
