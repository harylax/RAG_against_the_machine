import re
from collections import Counter
from chunk import process_all_chunks, Chunk
from pathlib import Path
from math import log


class Token(Chunk):
    def __init__(
            self,
            id: int | None,
            file_path: Path, first_character_index: int,
            last_character_index: int,
            text: str
            ) -> None:
        super().__init__(
            id, file_path, first_character_index, last_character_index, text
            )
        self.tokens: list[str] = BM25.tokenize(text)
        self.term_frequency: Counter[str] = Counter(self.tokens)


class Indexer:
    def __init__(
            self,
            path: str = 'data/raw/vllm-0.10.1',
            max_chunk_size: int = 2000
            ) -> None:
        self.chunks: list[Chunk] = process_all_chunks(path, max_chunk_size)
        self.tokens: list[Token] = [Token(
            token.id,
            token.file_path,
            token.first_character_index,
            token.last_character_index,
            token.text) for token in self.chunks]
        self.term_freqs: list[Counter[str]] = [
            token.term_frequency
            for token in self.tokens
            ]
        self.doc_freqs: Counter[str] = self.count_doc_frequency()
        self.doc_lengths: list[int] = [
            len(token.tokens) for token in self.tokens
            ]
        self.n_docs: int = len(self.chunks)
        self.avg_length: float = (
            sum(self.doc_lengths) / self.n_docs
            ) if self.n_docs else 0.0

    def count_doc_frequency(self) -> Counter[str]:
        res: Counter[str] = Counter()
        for tf in self.term_freqs:
            res.update(tf.keys())
        return res


class BM25:
    def __init__(self, indexer: Indexer):
        self.indexer: Indexer = indexer
        self.k1: float = 1.5
        self.b: float = 0.75

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

    def idf(self, term: str) -> float:
        N: int = self.indexer.n_docs
        df: int = self.indexer.doc_freqs.get(term, 0)
        return log(1 + (N - df + 0.5) / (df + 0.5))

    def score_term(self, term: str, doc_id: int) -> float:
        tf: int = self.indexer.term_freqs[doc_id].get(term, 0)
        if tf == 0:
            return 0.0
        idf: float = self.idf(term)
        doc_len: int = self.indexer.doc_lengths[doc_id]
        avg_len: float = self.indexer.avg_length

        numerator: float = tf * (self.k1 + 1)
        denominator: float = tf + self.k1 * (
            1 - self.b + self.b * doc_len / avg_len
            )
        return idf * (numerator / denominator)

    def score(self, query: str, doc_id: int) -> float:
        query_tokens: list[str] = self.tokenize(query)
        if not query_tokens:
            return 0.0
        scores: list[float] = [
            self.score_term(token, doc_id) for token in query_tokens
            ]
        return sum(scores)

    def get_scores(self, query: str) -> dict[int, float]:
        return {i: self.score(query, i) for i in range(self.indexer.n_docs)}

    def get_top_k(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        scores: dict[int, float] = self.get_scores(query)
        top: list[tuple[int, float]] = sorted(
            scores.items(), key=lambda x: x[1], reverse=True
            )
        return top[:k]


if __name__ == "__main__":
    idx: Indexer = Indexer('data/raw/vllm-0.10.1', 2000)
    bm25: BM25 = BM25(idx)

    # print(len(idx.chunks))
    # print(len(idx.doc_freqs))
    # print(idx.doc_freqs.most_common(10))

    # def count_doc_frequency(term_freqs: list[Counter[str]]) -> Counter[str]:
    #     res: Counter[str] = Counter()
    #     for tf in term_freqs:
    #         res.update(tf.keys())
    #     return res
    # term_freqs = [
    #     Counter({'abc': 1, 'yxz': 1}),
    #     Counter({'abc': 1, 'nop': 1})
    #     ]
    # print(count_doc_frequency(term_freqs))

    # print(idx.n_docs)
    # print(idx.avg_length)
    # print(len(idx.doc_lengths))

    # for term in ['the', 'version', 'rocm', 'get_rocm_version', 'zzzzzz']:
    #     print(
    #         f"term: {term}, doc freqs: {idx.doc_freqs.get(term, 0)}, "
    #         f"IDF: {bm25.idf(term)}"
    #         )

    # id: int = 12345
    # for term in ['the', 'model', '1', 'kv_caches', 'zzzzzz']:
    #     score: float = bm25.score_term(term, id)
    #     print(f"doc: {idx.tokens[id].text}")
    #     print(f"term: {repr(term)} -> score (doc id {id}): score {score}")
    import time
    start: float = time.perf_counter()
    top_k: list[tuple[int, float]] = bm25.get_top_k(
        "How do I configure the OpenAI compatible server in vLLM?",
        5)
    end: float = time.perf_counter()
    for doc_id, score in top_k:
        print(f"doc_id: {doc_id} | score: {score}")
        print(idx.tokens[doc_id].text)
        print('-------------------------------------------')
    # min: int = int((end - start) // 60)
    # sec: int = int((end - start) % 60)
    # print(f"Search completed in {min} minutes and {sec} seconds")
    print(f"Search took {end - start} seconds")
