from ollama import chat  # type: ignore
from .bm25 import BM25
from .retrieve import Retriever
from .models import MinimalSearchResults, MinimalSource
from .utils import read_chunk_text
from typing import Any


class Answer:
    def __init__(self, bm25: BM25, model: str = 'qwen3:0.6b'):
        self.model: str = model
        self.bm25: BM25 = bm25

    def build_context(self, retrieved_sources: list[MinimalSource]) -> str:
        res: list[str] = []
        for i, source in enumerate(retrieved_sources, start=1):
            res.append(f"[{i}] {read_chunk_text(source).strip()}")
        return '\n'.join(res)

    def build_prompt(self, question: str, k: int = 5) -> str:
        search_results: MinimalSearchResults = \
            Retriever(self.bm25).search(question, k=k)
        res: str = 'Read the following chunks of documents:\n'
        res += self.build_context(search_results.retrieved_sources)
        res += (
            '\nOnly answer the following question based on the '
            'previous provided documents. If you cannot find the '
            'answer in the documents, just say "I do not know"\n'
            )
        res += f'The question: "{question}"\n'
        return res

    def answer(self, question: str, k: int = 5) -> str:
        response: Any = chat(
            model=self.model,
            messages=[
                {
                    'role': 'system',
                    'content': (
                        'You are a RAG assistant. '
                        'Only answer in 1 to 3 sentences.'
                    )
                },
                {
                    'role': 'user',
                    'content': self.build_prompt(question, k)
                }
            ],
            think=False,
            options={'temperature': 0, 'num_ctx': 4096, 'num_predict': 200}
        )
        return response['message']['content']
