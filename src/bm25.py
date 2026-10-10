import re
import json
from .models import MinimalSource
from typing import Any
from math import log
from pathlib import Path


class BM25:
    def __init__(self, index_path: str = 'data/processed/index.json'):
        self.k1: float = 1.5
        self.b: float = 0.75
        self.load(index_path)

    @staticmethod
    def tokenize(text: str) -> list[str]:
        res: list[str] = re.findall(r"[A-Za-z0-9_]+", text)
        extra: list[str] = []
        for s in res:
            parts: list[
                str
                ] = re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|[0-9]+", s)
            if len(parts) > 1:
                extra.extend(parts)
        res.extend(extra)
        return [s.lower() for s in res]

    def load(self, index_path: str) -> None:
        try:
            with open(index_path) as f:
                data: dict[str, Any] = json.load(f)
        except (OSError, json.JSONDecodeError) as err:
            print(f"{err.__class__.__name__}: {err}")
            exit(1)
        self.sources = [MinimalSource(
            file_path=source['file_path'],
            first_character_index=source['first_character_index'],
            last_character_index=source['last_character_index']
        ) for source in data['sources']]
        self.term_freqs = data['term_freqs']
        self.doc_freqs = data['doc_freqs']
        self.doc_lengths = data['doc_lengths']
        self.n_docs = data['n_docs']
        self.avg_length = data['avg_length']

    def idf(self, term: str) -> float:
        N: int = self.n_docs
        df: int = self.doc_freqs.get(term, 0)
        return log(1 + (N - df + 0.5) / (df + 0.5))

    def score_term(self, term: str, doc_id: int) -> float:
        tf: int = self.term_freqs[doc_id].get(term, 0)
        if tf == 0:
            return 0.0
        idf: float = self.idf(term)
        doc_len: int = self.doc_lengths[doc_id]
        avg_len: float = self.avg_length
        k1: float = self.k1
        b: float = self.b
        numerator: float = tf * (k1 + 1)
        denominator: float = tf + k1 * (
            1 - b + b * doc_len / avg_len
            ) if avg_len else 0.0
        return idf * (numerator / denominator) if denominator else 0.0

    def score(self, query_tokens: list[str], doc_id: int) -> float:
        if not query_tokens:
            return 0.0
        scores: list[float] = [
            self.score_term(token, doc_id) for token in query_tokens
            ]
        return sum(scores)

    def get_scores(self, query: str) -> dict[int, float]:
        query_tokens: list[str] = self.tokenize(query)
        return {i: self.score(query_tokens, i) for i in range(self.n_docs)}

    def get_top_k(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        scores: dict[int, float] = self.get_scores(query)
        top: list[tuple[int, float]] = sorted(
            scores.items(), key=lambda x: x[1], reverse=True
            )
        return top[:k]

    @staticmethod
    def read_chunk_text(source: MinimalSource) -> str:
        path: Path = Path(source.file_path)
        try:
            with open(path) as f:
                i: int = source.first_character_index
                j: int = source.last_character_index + 1
                return f.read()[i:j]
        except OSError as err:
            print(f"{err.__class__.__name__}: {err}")
            return ''
