from .index import Indexer
from .bm25 import BM25
from .retrieve import Retriever
from fire import Fire  # type: ignore
from .models import MinimalSearchResults
from .answer import Answer
from pathlib import Path


class CLI:
    def index(
            self,
            max_chunk_size: int = 2000,
            corpus_path: str = 'data/raw/vllm-0.10.1',
            index_path: str = 'data/processed/index.json'
            ) -> None:
        Indexer(max_chunk_size, corpus_path, index_path)

    def search(
            self,
            query: str,
            k: int = 5,
            question_id: str | None = None,
            index_path: str = 'data/processed/index.json'
            ) -> None:
        bm25: BM25 = BM25(index_path)
        res: MinimalSearchResults = Retriever(bm25).search(
            query, question_id, k
            )
        for source in res.retrieved_sources:
            print(
                f"{source.file_path} "
                f"[{source.first_character_index}:"
                f"{source.last_character_index + 1}]"
                )

    def search_dataset(
            self,
            dataset_path: str = ('data/datasets/UnansweredQuestions'
                                 '/dataset_docs_public.json'),
            k: int = 5,
            save_directory: str = ('data/output/search_results'
                                   '/UnansweredQuestions'),
            index_path: str = 'data/processed/index.json'
            ) -> None:
        if 'docs' in Path(dataset_path).name:
            bm25: BM25 = BM25(index_path, True)
        else:
            bm25 = BM25(index_path, False)
        Retriever(bm25).search_dataset(dataset_path, k, save_directory)

    def answer(
            self,
            query: str,
            k: int = 5,
            index_path: str = 'data/processed/index.json'
    ) -> None:
        Answer(BM25(index_path)).answer(query, k)

    def answer_dataset(
            self,
            student_search_results_path: str = (
                'data/output/search_results/'
                'UnansweredQuestions/dataset_docs_public.json'
                ),
            save_directory: str = (
                'data/output/search_results_and_answer/'
                'UnsansweredQuestions/dataset_docs_public.json'
                ),
            index_path: str = 'data/processed/index.json'
    ) -> None:
        Answer(BM25(index_path)).answer_dataset(
            student_search_results_path, save_directory
            )


if __name__ == "__main__":
    Fire(CLI)
