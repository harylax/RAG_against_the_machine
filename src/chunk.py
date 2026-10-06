from pathlib import Path
from abc import ABC, abstractmethod
import ast

MIN_CHUNK_SIZE = 0
MIN_CHUNK_MARGIN = 0


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
    def __init__(self, max_chunk_size: int = 2000):
        super().__init__(max_chunk_size)

    def compute_node_length(self, lines: list[str], node: ast.stmt) -> int:
        start: int = node.lineno - 1
        end: int | None = node.end_lineno
        sub: str = ''.join(lines[start:end])
        return len(sub)

    def build_tree(
            self, lines: list[str],
            node: ast.stmt | ast.Module
            ) -> list[ast.stmt]:
        if isinstance(node, ast.Module):
            res: list[ast.stmt] = []
            for child in node.body:
                res.extend(self.build_tree(lines, child))
            return res
        len_node: int = self.compute_node_length(lines, node)
        if not isinstance(node, (
            ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef,
            ast.If, ast.For, ast.While, ast.With, ast.Try,
            ast.AsyncFor, ast.AsyncWith
        )):
            return [node]
        if len_node <= self.max_chunk_size:
            return [node]
        res = []
        for child in node.body:
            res.extend(self.build_tree(lines, child))
        return res

    def combine_small_nodes(
            self,
            lines: list[str],
            tree: list[ast.stmt]
            ) -> list[list[ast.stmt]]:
        res: list[list[ast.stmt]] = []
        current: list[ast.stmt] = []
        current_size: int = 0
        for node in tree:
            len_node: int = self.compute_node_length(lines, node)
            if isinstance(
                node,
                (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)
                    ):
                if current:
                    res.append(current)
                    current = []
                    current_size = 0
                res.append([node])
                continue
            if current_size + len_node <= self.max_chunk_size:
                current.append(node)
                current_size += len_node
            else:
                res.append(current)
                current = [node]
                current_size = len_node
        if current:
            res.append(current)
        return res

    def build_blocks(self, text: str) -> list[str]:
        lines: list[str] = text.splitlines(keepends=True)
        tree: ast.Module = ast.parse(text)
        nodes: list[ast.stmt] = self.build_tree(lines, tree)
        list_nodes: list[
            list[ast.stmt]
            ] = self.combine_small_nodes(lines, nodes)
        res: list[str] = []
        start: int = 0
        for nodes in list_nodes:
            if not nodes:
                continue
            end: int | None = nodes[-1].end_lineno
            if not end:
                end = nodes[-1].lineno
            res.append(''.join(lines[start:end]))
            start = end
        tail: str = ''.join(lines[start:])
        if not res:
            return [tail]
        if len(res[-1]) + len(tail) <= self.max_chunk_size:
            res[-1] += tail
        else:
            res.append(tail)
        return res

    def part_at_consecutive_newlines(self, text: str) -> list[str]:
        res: list[str] = []
        blocks: list[str] = self.build_blocks(text)
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
                sub: str = ''
                for line in lines:
                    if len(sub) + len(line) <= self.max_chunk_size:
                        sub += line
                    else:
                        res.append(sub)
                        sub = line
                if sub:
                    res.append(sub)
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

    def divide_file_into_chunks(
            self,
            file_path: Path,
            text: str,
            previous_id: int | None = None
            ) -> list[Chunk]:
        blocks: list[str] = self.final_part(text)
        res: list[Chunk] = []
        id: int = 0 if previous_id is None else previous_id + 1
        start: int = 0
        end: int = 0
        for block in blocks:
            end = start + len(block)
            if block.strip():
                res.append(
                    Chunk(
                        id, file_path, start, end - 1, text[start:end]
                        )
                    )
                id += 1
            start = end
        return res


if __name__ == "__main__":
    content: str | None = None
    with open('data/raw/vllm-0.10.1/setup.py') as f:
        content = f.read()
    py_chunk = PythonFilesChunks(1)
    for part in py_chunk.final_part(content):
        if not part.strip():
            continue
        print(part)
        print("-------------------------------------------------------------")
