from .bm25 import BM25
from .models import MinimalSearchResults
from uuid import uuid4


class Retriever:
    def __init__(self, bm25: BM25) -> None:
        self.bm25: BM25 = bm25

    def search(self, question: str, k: int = 5) -> MinimalSearchResults:
        return MinimalSearchResults(
            question_id=str(uuid4()),
            question=question,
            retrieved_sources=self.bm25.get_top_k_sources(question, k)
        )
