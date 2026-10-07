from pathlib import Path
from chunk import MarkdownChunks, PythonFilesChunks
from typing import Any
import json


def get_vllm_files_content(path_to_vllm: str) -> tuple[
    dict[Path, str], dict[Path, str]
        ]:
    md_files: dict[Path, str] = {}
    py_files: dict[Path, str] = {}

    vllm_repo: Path = Path(path_to_vllm)
    for file in vllm_repo.rglob('*'):
        if file.suffix == '.md':
            with open(file) as f:
                md_files[file] = f.read()
        elif file.suffix == '.py':
            with open(file) as f:
                py_files[file] = f.read()
    return (md_files, py_files)


def get_file_content(path_to_file: str) -> dict[Path, str]:
    path: Path = Path(path_to_file)
    with open(path) as f:
        return {path: f.read()}


def main() -> None:
    md_files, py_files = get_vllm_files_content('data/raw/vllm-0.10.1')
    # md_chunks: MarkdownChunks = MarkdownChunks()
    # md_chunks.process_chunks(md_files)
    # md_chunks.process_chunks(
    #     get_file_content('data/raw/vllm-0.10.1/README.md')
    #     )
    py_chunks: PythonFilesChunks = PythonFilesChunks(100)
    # py_chunks.process_chunks(py_files)
    py_chunks.process_chunks(
        get_file_content('data/raw/vllm-0.10.1/setup.py')
        )
    for chunk in py_chunks.processed_chunks:
        to_print: dict[str, Any] = {
            "id": chunk.id,
            "file_path": str(chunk.file_path),
            "first_character_index": chunk.first_character_index,
            "last_character_index": chunk.last_character_index,
            "text": chunk.text
        }
        print(json.dumps(to_print, indent=2))


if __name__ == "__main__":
    main()
