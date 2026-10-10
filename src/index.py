from collections import Counter
from .chunk import process_all_chunks, Chunk
from pathlib import Path
from typing import Any
import json
from .bm25 import BM25
from .models import MinimalSource


class Indexer:
    def __init__(
            self,
            corpus_path: str = 'data/raw/vllm-0.10.1',
            max_chunk_size: int = 2000,
            index_path: str = 'data/processed/index.json'
            ) -> None:
        self.max_chunk_size: int = max_chunk_size
        self.load(max_chunk_size, corpus_path, index_path)

    def load(
            self,
            max_chunk_size: int,
            corpus_path: str,
            index_path: str = 'data/processed/index.json'
            ) -> None:
        if self.reindexation_needed(corpus_path):
            self.chunks: list[Chunk] = process_all_chunks(
                corpus_path, max_chunk_size
                )
            self.sources: list[MinimalSource] = [
                MinimalSource(
                    file_path=str(chunk.file_path),
                    first_character_index=chunk.first_character_index,
                    last_character_index=chunk.last_character_index
                ) for chunk in self.chunks
            ]
            self.tokens: list[list[str]] = [
                BM25.tokenize(chunk.text) for chunk in self.chunks
                ]
            self.term_freqs: list[dict[str, int]] = [
                dict(Counter(token)) for token in self.tokens
                ]
            self.doc_freqs: dict[str, int] = self.count_doc_frequency()
            self.doc_lengths: list[int] = [
                len(token) for token in self.tokens
                ]
            self.n_docs: int = len(self.chunks)
            self.avg_length: float = (
                sum(self.doc_lengths) / self.n_docs
                ) if self.n_docs else 0.0
            self.save(index_path)
            self.write_checker(corpus_path)

    def count_doc_frequency(self) -> dict[str, int]:
        res: dict[str, int] = {}
        for terms in self.term_freqs:
            for term in terms.keys():
                res[term] = res.get(term, 0) + 1
        return res

    def save(self, save_path: str = 'data/processed/index.json') -> None:
        data: dict[str, Any] = {
            'sources': [
                {
                    'file_path': str(source.file_path),
                    'first_character_index': source.first_character_index,
                    'last_character_index': source.last_character_index
                } for source in self.sources
            ],
            'term_freqs': self.term_freqs,
            'doc_freqs': self.doc_freqs,
            'doc_lengths': self.doc_lengths,
            'n_docs': self.n_docs,
            'avg_length': self.avg_length
        }
        path: Path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        except OSError as err:
            print(f"{err.__class__.__name__}: {err}")

    def check_corpus(
            self,
            corpus_path: str = 'data/raw/vllm-0.10.1'
            ) -> dict[str, Any]:
        files: dict[str, Any] = {}
        path: Path = Path(corpus_path)
        for file in path.rglob('*'):
            if file.suffix in ['.md', '.py']:
                stat: Any = file.stat()
                files[str(file)] = [stat.st_mtime, stat.st_size]
        files['max_chunk_size'] = self.max_chunk_size
        return files

    def write_checker(
            self,
            corpus_path: str = 'data/raw/vllm-0.10.1'
            ) -> None:
        checker_path: Path = Path('data/processed/checker.json')
        checker_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(checker_path, 'w') as f:
                json.dump(self.check_corpus(corpus_path), f, indent=2)
        except OSError as err:
            print(f"{err.__class__.__name__}: {err}")

    def reindexation_needed(
            self,
            corpus_path: str = 'data/raw/vllm-0.10.1'
            ) -> bool:
        checker_path = Path("data/processed/checker.json")
        if not checker_path.exists():
            return True
        try:
            with open(checker_path) as f:
                old: dict[str, Any] = json.load(f)
        except (OSError, json.JSONDecodeError):
            return True
        if old.get('max_chunk_size') != self.max_chunk_size:
            return True
        if self.check_corpus(corpus_path) != old:
            return True
        return False


if __name__ == "__main__":
    idx: Indexer = Indexer('data/raw/vllm-0.10.1', 2000)
