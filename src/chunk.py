from pathlib import Path
from abc import ABC, abstractmethod
import ast

MIN_CHUNK_SIZE = 50


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
            self.max_chunk_size < MIN_CHUNK_SIZE
            or self.max_chunk_size > 2000
        ):
            print(
                "\033[31mmax_chunk_size should be at least "
                f"{MIN_CHUNK_SIZE} and at most 2000\033[0m"
                )
            exit(1)

    def process_chunks(self, files: dict[Path, str]) -> None:
        previous_id: int | None = None
        for file_path, text in files.items():
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
        pass

    def split_at_chars(self, blocks: list[str], chars: str) -> list[str]:
        res: list[str] = []
        for block in blocks:
            if len(block) <= self.max_chunk_size:
                res.append(block)
                continue
            parts: list[str] = block.split(chars)
            end: int = len(parts) - 1
            current: str = ''
            for i, part in enumerate(parts):
                if i < end:
                    part += chars
                if current == '':
                    current += part
                    continue
                if len(current) + len(part) <= self.max_chunk_size:
                    current += part
                    continue
                res.append(current)
                current = part
            if current:
                res.append(current)
        return res

    def split_at_consecutive_newlines(self, blocks: list[str]) -> list[str]:
        return self.split_at_chars(blocks, '\n\n')

    def split_at_newline(self, blocks: list[str]) -> list[str]:
        return self.split_at_chars(blocks, '\n')

    def split_at_space(self, blocks: list[str]) -> list[str]:
        return self.split_at_chars(blocks, ' ')

    def merge_small_blocks(self, blocks: list[str]) -> list[str]:
        res: list[str] = []
        for block in blocks:
            if res and len(res[-1]) + len(block) <= self.max_chunk_size:
                res[-1] += block
                continue
            res.append(block)
        return res

    def split_at_max_chunk_size(self, blocks: list[str]) -> list[str]:
        res: list[str] = []
        for block in blocks:
            if len(block) <= self.max_chunk_size:
                res.append(block)
                continue
            while len(block) > self.max_chunk_size:
                res.append(block[:self.max_chunk_size])
                block = block[self.max_chunk_size:]
            res.append(block)
        return res

    def merge_blank_blocks(self, blocks: list[str]) -> list[str]:
        if not blocks:
            return []
        res: list[str] = []
        tmp: str = ''
        for first, second in zip(blocks[:-1], blocks[1:]):
            if tmp:
                first = tmp
            if (
                res and not first.strip()
                and len(res[-1]) + len(first) <= self.max_chunk_size
                    ):
                res[-1] += first
                tmp = ''
            elif (
                res and not first.strip()
                and len(first) + len(second) <= self.max_chunk_size
            ):
                tmp = first + second
            else:
                res.append(first)
                tmp = ''
        if tmp:
            res.append(tmp)
        else:
            res.append(blocks[-1])
        return res


class MarkdownChunks(Chunks):
    def __init__(self, max_chunk_size: int = 2000) -> None:
        super().__init__(max_chunk_size)

    def divide_by_title(self, text: str) -> list[str]:
        if len(text) <= self.max_chunk_size:
            return [text]
        blocks: list[list[str]] = []
        lines: list[str] = text.splitlines(keepends=True)
        start: int = 0
        end: int = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('#'):
                end = i
                blocks.append(lines[start:end])
                start = end
        if start < len(lines):
            blocks.append(lines[start:])
        res: list[list[str]] = []
        for block in blocks:
            if res and all(
                line.strip().startswith('#')
                or line.strip() == ''
                for line in res[-1]
            ):
                res[-1] += block
            else:
                res.append(block)
        return [''.join(lines) for lines in res]

    def split_in_sentences(self, text: str) -> list[str]:
        blocks: list[str] = self.divide_by_title(text)
        blocks = self.split_at_consecutive_newlines(blocks)
        blocks = self.split_at_newline(blocks)
        blocks = self.split_at_chars(blocks, '. ')
        blocks = self.split_at_chars(blocks, '! ')
        return self.split_at_chars(blocks, '? ')

    def divide_file_into_chunks(
        self,
        file_path: Path,
        text: str,
        previous_id: int | None = None
            ) -> list[Chunk]:
        blocks: list[str] = self.split_in_sentences(text)
        blocks = self.split_at_space(blocks)
        blocks = self.split_at_max_chunk_size(blocks)
        blocks = self.merge_small_blocks(blocks)
        blocks = self.merge_blank_blocks(blocks)
        res: list[Chunk] = []
        id: int = 0 if previous_id is None else previous_id + 1
        start: int = 0
        end: int = 0
        for block in blocks:
            if not block:
                continue
            end = start + len(block)
            res.append(
                Chunk(
                    id, file_path, start, end - 1, text[start:end]
                    )
                )
            id += 1
            start = end
        return res


class PythonFilesChunks(Chunks):
    def __init__(self, max_chunk_size: int = 2000) -> None:
        super().__init__(max_chunk_size)

    def compute_node_length(self, lines: list[str], node: ast.stmt) -> int:
        start: int = node.lineno - 1
        end: int | None = node.end_lineno
        sub: str = ''.join(lines[start:end])
        return len(sub)

    def get_smallest_nodes(
            self, lines: list[str],
            node: ast.stmt | ast.Module
            ) -> list[ast.stmt]:
        if isinstance(node, ast.Module):
            res: list[ast.stmt] = []
            for child in node.body:
                res.extend(self.get_smallest_nodes(lines, child))
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
            res.extend(self.get_smallest_nodes(lines, child))
        return res

    def merge_small_nodes(
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
        try:
            tree: ast.Module = ast.parse(text)
        except SyntaxError as err:
            print(f"\033[31mSyntaxError: {err}\033[0m")
            exit(1)
        nodes: list[ast.stmt] = self.get_smallest_nodes(lines, tree)
        list_nodes: list[
            list[ast.stmt]
            ] = self.merge_small_nodes(lines, nodes)
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
        if not tail:
            return res
        if res and len(res[-1]) + len(tail) <= self.max_chunk_size:
            res[-1] += tail
        else:
            res.append(tail)
        return res

    def divide_file_into_chunks(
            self,
            file_path: Path,
            text: str,
            previous_id: int | None = None
            ) -> list[Chunk]:
        blocks: list[str] = self.build_blocks(text)
        blocks = self.split_at_consecutive_newlines(blocks)
        blocks = self.split_at_newline(blocks)
        blocks = self.split_at_space(blocks)
        blocks = self.split_at_max_chunk_size(blocks)
        blocks = self.merge_small_blocks(blocks)
        blocks = self.merge_blank_blocks(blocks)
        res: list[Chunk] = []
        id: int = 0 if previous_id is None else previous_id + 1
        start: int = 0
        end: int = 0
        for block in blocks:
            if not block:
                continue
            end = start + len(block)
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
    py_chunk = PythonFilesChunks(400)
    for chunk in py_chunk.divide_file_into_chunks(
        Path('data/raw/vllm-0.10.1/setup.py'),
        content
            ):
        print(chunk.text)
        print("-------------------------------------------------------------")
    # content: str | None = None
    # with open('data/raw/vllm-0.10.1/README.md') as f:
    #     content = f.read()
    # md_chunk = MarkdownChunks(500)
    # for chunk in md_chunk.divide_file_into_chunks(
    #     Path('data/raw/vllm-0.10.1/README.md'),
    #     content
    #         ):
    #     print(chunk.text)
    #     print("-------------------------------------------------------------")
    # content: str | None = None
    # with open('data/raw/vllm-0.10.1/README.md') as f:
    #     content = f.read()
    # md_chunk = MarkdownChunks(100)
    # chunks = md_chunk.divide_file_into_chunks(
    #     Path('data/raw/vllm-0.10.1/README.md'),
    #     content
    # )
    # assert ''.join(chunk.text for chunk in chunks) == content
    # assert all(len(chunk.text) <= 100 for chunk in chunks)
    # content: str | None = None
    # with open('data/raw/vllm-0.10.1/setup.py') as f:
    #     content = f.read()
    # py_chunk = PythonFilesChunks(100)
    # chunks = py_chunk.divide_file_into_chunks(
    #     Path('data/raw/vllm-0.10.1/README.md'),
    #     content
    # )
    # assert ''.join(chunk.text for chunk in chunks) == content
    # assert all(len(chunk.text) <= 100 for chunk in chunks)
