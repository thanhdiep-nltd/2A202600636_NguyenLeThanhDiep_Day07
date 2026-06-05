from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        # Split on sentence boundaries: . ! ? followed by space or end of string
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Clean up whitespace for each sentence
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return []
        
        # Group sentences into chunks
        chunks: list[str] = []
        current_chunk_sentences: list[str] = []
        
        for sentence in sentences:
            current_chunk_sentences.append(sentence)
            # Check if we've reached the desired number of sentences per chunk
            if len(current_chunk_sentences) >= self.max_sentences_per_chunk:
                # Join the sentences back into a single chunk string
                chunks.append(" ".join(current_chunk_sentences))
                current_chunk_sentences = []
        
        # Add any remaining sentences as the last chunk
        if current_chunk_sentences:
            chunks.append(" ".join(current_chunk_sentences))
        
        return chunks

class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        return self._split(text, self.separators)
    
    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        if len(current_text) <= self.chunk_size:
            return [current_text]
        
        if not remaining_separators:
            return [current_text]
        
        separator = remaining_separators[0]
        parts = current_text.split(separator)
        
        if len(parts) == 1:
            return self._split(current_text, remaining_separators[1:])
        
        chunks: list[str] = []
        for part in parts:
            chunks.extend(self._split(part, remaining_separators[1:]))
        
        return chunks


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    dot_ab = _dot(vec_a, vec_b)
    mag_a = math.sqrt(_dot(vec_a, vec_a))
    mag_b = math.sqrt(_dot(vec_b, vec_b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot_ab / (mag_a * mag_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        # Initialize chunkers
        chunkers = {
            "fixed_size": FixedSizeChunker(chunk_size=chunk_size),
            "by_sentences": SentenceChunker(),
            "recursive": RecursiveChunker(chunk_size=chunk_size)
        }

        results = {}
        for name, chunker in chunkers.items():
            chunks = chunker.chunk(text)
            avg_len = sum(len(c) for c in chunks) / len(chunks) if chunks else 0.0
            results[name] = {
                "count": len(chunks),
                "avg_length": avg_len,
                "chunks": chunks
            }

        return results