from pathlib import Path
from abc import ABC, abstractmethod
import ast
from typing import Any
import json

MIN_CHUNK_SIZE = 200


class Chunk:
    def __init__(
        self,
        id: int | None,
        file_path: Path,
        first_character_index: int,
        last_character_index: int,
        text: str
    ) -> None:
        self.id: int | None = None
        self.file_path: Path = file_path
        self.first_character_index: int = first_character_index
        self.last_character_index: int = last_character_index
        self.text: str = text
        self.tokens: list[str] = []


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

    def process_chunks(
            self,
            files: dict[Path, str]
            ) -> None:
        previous_id: int | None = None
        for file_path, text in files.items():
            # if self.processed_chunks:
            #     previous_id = self.processed_chunks[-1].id
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
            if block:
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
                    None, file_path, start, end - 1, text[start:end]
                    )
                )
            id += 1
            start = end
        return res


class PythonFilesChunks(Chunks):
    def __init__(self, max_chunk_size: int = 2000) -> None:
        super().__init__(max_chunk_size)

    def node_length(self, lines: list[str], node: ast.stmt) -> int:
        start: int = node.lineno - 1
        end: int = node.end_lineno or node.lineno
        return len(''.join(lines[start:end]))

    def divide_in_nodes(self, text: str, tree: ast.Module) -> list[ast.stmt]:
        lines: list[str] = text.splitlines(keepends=True)
        res: list[ast.stmt] = []
        for node in tree.body:
            res.extend(self.get_smallest_nodes(lines, node))
        return res

    def get_smallest_nodes(
            self,
            lines: list[str],
            node: ast.stmt
            ) -> list[ast.stmt]:
        if self.node_length(lines, node) <= self.max_chunk_size:
            return [node]

        def get_children() -> list[ast.stmt]:
            children: list[ast.stmt] = list(getattr(node, 'body', []))
            children.extend(getattr(node, 'orelse', []))
            children.extend(getattr(node, 'finalbody', []))
            for handler in getattr(node, 'handlers', []):
                children.extend(handler.body)
            for case in getattr(node, 'cases', []):
                children.extend(case.body)
            return sorted(children, key=lambda child: child.lineno)

        res: list[ast.stmt] = []
        children: list[ast.stmt] = get_children()
        if not children:
            return [node]
        for child in children:
            res.extend(self.get_smallest_nodes(lines, child))
        return res

    def group_semantic_nodes(
            self,
            lines: list[str],
            nodes: list[ast.stmt]
            ) -> list[list[ast.stmt]]:
        res: list[list[ast.stmt]] = []
        group: list[ast.stmt] = []
        group_len: int = 0
        for node in nodes:
            node_len: int = self.node_length(lines, node)
            if not group:
                group = [node]
                group_len += node_len
                continue

            if node_len + group_len > self.max_chunk_size:
                res.append(group)
                group, group_len = [node], node_len
                continue

            if group[0].col_offset <= node.col_offset:
                group.append(node)
                group_len += node_len
            else:
                res.append(group)
                group, group_len = [node], node_len

        if group:
            res.append(group)
        return res

    def build_blocks(self, text: str) -> list[str]:
        try:
            tree: ast.Module = ast.parse(text)
        except SyntaxError:
            return [text]
        lines: list[str] = text.splitlines(keepends=True)
        nodes: list[ast.stmt] = self.divide_in_nodes(text, tree)
        groups_nodes: list[
            list[ast.stmt]
            ] = self.group_semantic_nodes(lines, nodes)
        res: list[str] = []
        start: int = 0
        for group in groups_nodes:
            node_end: int = group[-1].end_lineno or group[-1].lineno
            res.append(''.join(lines[start:node_end]))
            start = node_end

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
                    None, file_path, start, end - 1, text[start:end]
                    )
                )
            id += 1
            start = end
        return res


def get_files_content(path: str) -> tuple[
    dict[Path, str], dict[Path, str]
        ]:
    md_files: dict[Path, str] = {}
    py_files: dict[Path, str] = {}

    vllm: Path = Path(path)
    for file in vllm.rglob('*'):
        if file.suffix == '.md':
            try:
                with open(file) as f:
                    md_files[file] = f.read()
            except OSError:
                continue
        elif file.suffix == '.py':
            try:
                with open(file) as f:
                    py_files[file] = f.read()
            except OSError:
                continue
    return (md_files, py_files)


def process_all_chunks(
        read_path: str,
        max_chunk_size: int = 2000
        ) -> list[Chunk]:
    md_files, py_files = get_files_content(read_path)

    md_chunks: MarkdownChunks = MarkdownChunks(max_chunk_size)

    py_chunks: PythonFilesChunks = PythonFilesChunks(max_chunk_size)

    md_chunks.process_chunks(md_files)
    py_chunks.process_chunks(py_files)

    return md_chunks.processed_chunks + py_chunks.processed_chunks


def write_processed_chunks(
        all_chunks: list[Chunk],
        save_path: str
        ) -> None:
    path: Path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data: list[dict[str, Any]] = []

    for i, chunk in enumerate(all_chunks):
        chunk.id = i
        data.append({
            "id": i,
            "file_path": str(chunk.file_path),
            "first_character_index": chunk.first_character_index,
            "last_character_index": chunk.last_character_index,
            "text": chunk.text
        })
    try:
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    except OSError as err:
        print(f"\033[31m{err.__class__.__name__}: {err}\033[0m")
        exit(1)


if __name__ == "__main__":
    def test_py(x: int = 2000) -> None:
        content: str | None = None
        with open('data/raw/vllm-0.10.1/setup.py') as f:
            content = f.read()
        py_chunk = PythonFilesChunks(x)
        for chunk in py_chunk.divide_file_into_chunks(
            Path('data/raw/vllm-0.10.1/setup.py'),
            content
                ):
            print(chunk.text)
            print("----------------------------------------------------------")

    def test_md(x: int = 2000) -> None:
        content: str | None = None
        with open('data/raw/vllm-0.10.1/README.md') as f:
            content = f.read()
        md_chunk = MarkdownChunks(x)
        for chunk in md_chunk.divide_file_into_chunks(
            Path('data/raw/vllm-0.10.1/README.md'),
            content
                ):
            print(chunk.text)
            print("----------------------------------------------------------")

    def test_no_loss_md(x: int = 2000) -> None:
        content: str | None = None
        with open('data/raw/vllm-0.10.1/README.md') as f:
            content = f.read()
        md_chunk = MarkdownChunks(x)
        chunks = md_chunk.divide_file_into_chunks(
            Path('data/raw/vllm-0.10.1/README.md'),
            content
        )
        assert ''.join(chunk.text for chunk in chunks) == content
        assert all(len(chunk.text) <= x for chunk in chunks)

    def test_no_loss_py(x: int = 2000) -> None:
        content: str | None = None
        with open('data/raw/vllm-0.10.1/setup.py') as f:
            content = f.read()
        py_chunk = PythonFilesChunks(x)
        chunks = py_chunk.divide_file_into_chunks(
            Path('data/raw/vllm-0.10.1/setup.py'),
            content
        )
        assert ''.join(chunk.text for chunk in chunks) == content
        assert all(len(chunk.text) <= x for chunk in chunks)

    # test_md(200)
    # test_no_loss_md(200)
    # test_py(200)
    # test_no_loss_py(200)
