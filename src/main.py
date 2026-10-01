from pathlib import Path
from .chunk import Chunk


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


def markdown_chunking(files: dict[Path, str]) -> list[Chunk]:
    ...


def python_code_chunking(files: dict[Path, str]) -> list[Chunk]:
    ...


def main() -> None:
    md_files, py_files = get_vllm_files_content('data/raw/vllm-0.10.1')
    count = 0
    for path, text in md_files.items():
        print(
            f"\033[35mpath: {path}\033[0m\n"
            f"text:\n\t{text}\n"
        )
        count += 1
    print(f"\033[31m{count}\033[0m")
    print()
    count = 0
    for path, text in py_files.items():
        print(
            f"\033[35mpath: {path}\033[0m\n"
            f"text:\n\t{text}\n"
        )
        count += 1
    print(f"\033[31m{count}\033[0m")


if __name__ == "__main__":
    main()
