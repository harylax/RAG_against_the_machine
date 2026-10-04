from pathlib import Path
from abc import ABC, abstractmethod
MIN_CHUNK_SIZE = 100
MIN_CHUNK_MARGIN = 300


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
        self.processed_chunks: list[Chunk] = []
        self.max_chunk_size: int = max_chunk_size
        if (
            self.max_chunk_size < MIN_CHUNK_SIZE + MIN_CHUNK_MARGIN
            or self.max_chunk_size > 2000
        ):
            print(
                "\033[031mmax_chunk_size should be at least "
                f"{MIN_CHUNK_SIZE + MIN_CHUNK_MARGIN} and at most 2000\033[0m"
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
    def __init__(self, max_chunk_size: int = 2000) -> None:
        super().__init__(max_chunk_size)
        self.syntax_prefixes: tuple[str, ...] = (
            'def ', 'class ', 'async def ', 'if ', 'while'
            'from ', 'import ', '@', 'try:', 'for '
        )
        self.second_prefixes: tuple[str, ...] = (
            'elif ', 'else:', 'except ', 'finally:', 'with '
        )

    def _split_block_at(
            self,
            text: str,
            indent: int,
            prefixes: tuple[str, ...]
            ) -> list[str]:
        _prefixes: tuple[str, ...] = tuple(
            ' ' * indent + prefix for prefix in prefixes
            )
        lines: list[str] = text.splitlines(keepends=True)
        res: list[str] = []
        start: int = 0
        for i, line in enumerate(lines):
            if line.startswith(_prefixes):
                res.append(''.join(lines[start:i]))
                start = i
        if start < len(lines):
            res.append(''.join(lines[start:]))
        return res

    def part_with_indented_blocks(self, text: str) -> list[str]:
        blocks: list[str] = [text]
        for level in range(6):
            res: list[str] = []
            for block in blocks:
                if len(block) > self.max_chunk_size:
                    res.extend(self._split_block_at(
                        block,
                        4 * level,
                        self.syntax_prefixes
                        ))
                else:
                    res.append(block)
            blocks = res
        for level in range(6):
            res = []
            for block in blocks:
                if len(block) > self.max_chunk_size:
                    res.extend(self._split_block_at(
                        block,
                        4 * level,
                        self.second_prefixes
                        ))
                else:
                    res.append(block)
            blocks = res
        return blocks

    def part_at_consecutive_newlines(self, text: str) -> list[str]:
        res: list[str] = []
        blocks: list[str] = self.part_with_indented_blocks(text)
        for block in blocks:
            if len(block) > self.max_chunk_size:
                lines: list[str] = block.splitlines(keepends=True)
                start: int = 0
                for i, line in enumerate(lines):
                    if line.startswith('\n'):
                        res.append(''.join(lines[start:i]))
                        start = i
                if start < len(lines):
                    res.append(''.join(lines[start:]))
            else:
                res.append(block)
        return res

    def part_at_newline(self, text: str) -> list[str]:
        res: list[str] = []
        blocks: list[str] = self.part_at_consecutive_newlines(text)
        for block in blocks:
            if len(block) > self.max_chunk_size:
                lines: list[str] = block.splitlines(keepends=True)
                res.extend(lines)
            else:
                res.append(block)
        return res

    def final_part(self, text: str) -> list[str]:
        res: list[str] = []
        blocks: list[str] = self.part_at_newline(text)
        for block in blocks:
            while len(block) > self.max_chunk_size:
                res.append(block[:self.max_chunk_size])
                block = block[self.max_chunk_size:]
            res.append(block)
        return res

    def concat_parts(self, text: str) -> list[str]:
        res: list[str] = []
        parts: list[str] = self.final_part(text)
        for part in parts:
            concatenated: bool = False
            if res:
                if len(res[-1]) < MIN_CHUNK_SIZE or len(part) < MIN_CHUNK_SIZE:
                    if len(res[-1]) + len(part) <= self.max_chunk_size:
                        res[-1] += part
                        concatenated = True
            if not concatenated:
                res.append(part)
        return res

    def divide_file_into_chunks(
            self,
            file_path: Path,
            text: str,
            previous_id: int | None = None
            ) -> list[Chunk]:
        res: list[Chunk] = []
        id: int = 0 if previous_id is None else previous_id + 1
        start: int = 0
        end: int = 0
        for part in self.concat_parts(text):
            end = start + len(part)
            res.append(Chunk(id, file_path, start, end - 1, part))
            start = end
            id += 1
        return res


if __name__ == "__main__":
    content: str | None = None
    # with open('data/raw/vllm-0.10.1/setup.py') as f:
    with open('src/chunk.py') as f:
        content = f.read()
    py_chunk = PythonFilesChunks(2000)
    for part in py_chunk.concat_parts(content):
        print(part)
        print("-------------------------------------------------------------")
