from ollama import chat  # type: ignore
from .bm25 import BM25
from .retrieve import Retriever
from .models import (
    MinimalSearchResults, MinimalSource, StudentSearchResultsAndAnswer,
    MinimalAnswer, StudentSearchResults
)
from pathlib import Path
from .utils import read_chunk_text
from typing import Any
from tqdm import tqdm  # type: ignore


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

    def answer(
            self,
            question: str,
            k: int = 5,
            stream: bool = True
            ) -> str:
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
            stream=stream,
            options={'temperature': 0, 'num_ctx': 4096, 'num_predict': 200}
        )
        if not stream:
            return response['message']['content']
        print(f"\033[32m>>>Query: {question!r}\033[0m")
        print("\033[33mAnswer: \033[0m", end='')
        generated: str = ''
        for token in response:
            content: str = token['message']['content']
            generated += content
            print(f"\033[33m{content}\033[0m", end='', flush=True)
        print()
        return generated

    def answer_dataset(
            self,
            student_search_results_path: str,
            save_directory: str
    ) -> StudentSearchResultsAndAnswer:
        res: list[MinimalAnswer] = []
        search_path: Path = Path(student_search_results_path)
        if not search_path.exists():
            dataset_path: Path = Path(
                'data/output/search_results/UnansweredQuestions'
                ) / search_path.name
            print(
                "\033[31m"
                f"student_search_result_path: '{student_search_results_path}' "
                "not found.\nUsage: uv run python -m src search_dataset "
                f"-dataset_path {dataset_path} -k <int> "
                f"-save_directory {student_search_results_path}"
                )
            exit(1)
        student_search_result: StudentSearchResults = \
            StudentSearchResults.model_validate_json(search_path.read_text())
        for search_res in tqdm(
            student_search_result.search_results,
            desc='Answering',
            unit=' query'
                ):
            res.append(MinimalAnswer(
                question_id=search_res.question_id,
                question=search_res.question,
                retrieved_sources=search_res.retrieved_sources,
                answer=self.answer(search_res.question, 3, False)
                ))
        save_path: Path = Path(save_directory) / Path(
            student_search_results_path
            ).name
        student_search_results_and_answer: StudentSearchResultsAndAnswer = \
            StudentSearchResultsAndAnswer(
                search_results=res,
                k=student_search_result.k
                )
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(
            student_search_results_and_answer.model_dump_json(indent=2)
            )
        return student_search_results_and_answer
