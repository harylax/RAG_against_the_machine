from .bm25 import BM25
from .models import (
    MinimalSearchResults, StudentSearchResults, RagDataset
    )
from uuid import uuid4
from pathlib import Path
from tqdm import tqdm  # type: ignore
from pydantic import ValidationError  # type: ignore


class Retriever:
    def __init__(self, bm25: BM25) -> None:
        self.bm25: BM25 = bm25

    def search(
            self,
            question: str,
            question_id: str | None = None,
            k: int = 5
            ) -> MinimalSearchResults:
        return MinimalSearchResults(
            question_id=question_id if question_id else str(uuid4()),
            question=question,
            retrieved_sources=self.bm25.get_top_k_sources(question, k)
        )

    def search_dataset(
            self,
            dataset_path: str = ('data/datasets/UnansweredQuestions'
                                 '/dataset_docs_public.json'),
            k: int = 5,
            save_directory: str = ('data/output/search_results'
                                   '/UnansweredQuestions')
            ) -> None:
        try:
            rag_dataset: RagDataset = RagDataset.model_validate_json(
                Path(dataset_path).read_text()
                )
        except ValidationError as err:
            for error in err.errors():
                print(f"\033[31mValidationError: {error['msg']}\033[0m")
            exit(1)
        except OSError as err:
            print(f"\033[31m{err.__class__.__name__}: {err}\033[0m")
            exit(1)
        search_results: list[MinimalSearchResults] = []
        for question in tqdm(
            rag_dataset.rag_questions, desc='Searching', unit=' question '
                ):
            search_results.append(MinimalSearchResults(
                question_id=question.question_id,
                question=question.question,
                retrieved_sources=self.bm25.get_top_k_sources(
                    question.question, k
                    )
            ))
        save_path: Path = Path(save_directory) / Path(dataset_path).name
        save_path.parent.mkdir(parents=True, exist_ok=True)
        student_search_results: StudentSearchResults = StudentSearchResults(
            search_results=search_results,
            k=k
        )
        save_path.write_text(student_search_results.model_dump_json(indent=2))
        print(f"Saved student_search_results to {save_path}")
