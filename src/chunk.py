from pathlib import Path
from abc import ABC, abstractmethod
MIN_CHUNK_SIZE = 500


class Chunk:
    def __init__(
        self,
        id: int,
        file_path: Path,
        first_character_index: int,
        last_character_index: int,
        text: str
    ) -> None:
        self.id: int = id
        self.file_path: Path = file_path
        self.first_character_index: int = first_character_index
        self.last_character_index: int = last_character_index
        self.text: str = text


class Chunks(ABC):
    def __init__(self, max_chunk_size: int = 2000) -> None:
        # self.processed_md_chunks: list[Chunk] = []
        # self.processed_py_chunks: list[Chunk] = []
        self.processed_chunks: list[Chunk] = []
        self.max_chunk_size: int = max_chunk_size
        if (
            self.max_chunk_size < MIN_CHUNK_SIZE
            or self.max_chunk_size > 2000
        ):
            print(
                "\033[031mmax_chunk_size should be at least "
                f"{MIN_CHUNK_SIZE} and at most 2000\033[0m"
                )
            exit(1)

    def process_chunks(self, md_files: dict[Path, str]) -> None:
        previous_id: int | None = None
        for file_path, text in md_files.items():
            if self.processed_chunks:
                previous_id = self.processed_chunks[-1].id
            self.processed_chunks.extend(
                self.divide_file_into_chunks(
                    file_path, text, previous_id
                    )
                )

    @abstractmethod
    def divide_file_into_chunks(
            self,
            file_path: Path,
            text: str,
            previous_id: int | None = None
            ) -> list[Chunk]:
        ...


class MarkdownChunks(Chunks):
    def divide_file_into_chunks(
            self,
            file_path: Path,
            text: str,
            previous_id: int | None = None
            ) -> list[Chunk]:
        res: list[Chunk] = []
        id: int = 0 if previous_id is None else previous_id + 1
        n: int = len(text)
        i: int = 0
        start: int | None = None
        while i < n:
            if start is None and text[i] == '\n':
                i += 1
                continue

            if start is None:
                start = i

            if i - start < MIN_CHUNK_SIZE:
                i += 1
                continue

            if i - start >= self.max_chunk_size:
                j: int = text.rfind('\n', start + MIN_CHUNK_SIZE, i)
                if j < 0:
                    j = text.rfind('. ', start + MIN_CHUNK_SIZE, i)
                    if j < 0:
                        j = start + self.max_chunk_size
                    else:
                        j = j + 1
                res.append(
                    Chunk(id, file_path, start, j - 1, text[start:j])
                    )
                start = None
                id += 1
                i = j
                continue

            if text[i] == '\n' and text[i - 1] == '\n':
                end: int = i - 1
                res.append(
                    Chunk(id, file_path, start, end - 1, text[start:end])
                    )
                start = None
                id += 1
            i += 1

        if start is not None:
            res.append(Chunk(id, file_path, start, n - 1, text[start:n]))
        return res


class PythonFilesChunks(Chunks):
    def divide_file_into_chunks(
            self,
            file_path: Path,
            text: str,
            previous_id: int | None = None
            ) -> list[Chunk]:
        res: list[Chunk] = []
        ...
        return res
