import re
from pathlib import Path
from .models import MinimalSource, Chunk


def tokenize(text: str) -> list[str]:
    res: list[str] = re.findall(r"[A-Za-z0-9_]+", text)
    extra: list[str] = []
    for s in res:
        parts: list[
            str
            ] = re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|[0-9]+", s)
        if len(parts) > 1:
            extra.extend(parts)
    res.extend(extra)
    return [s.lower() for s in res]


def read_chunk_text(source: MinimalSource) -> str:
    path: Path = Path(source.file_path)
    try:
        with open(path) as f:
            i: int = source.first_character_index
            j: int = source.last_character_index + 1
            return f.read()[i:j]
    except OSError as err:
        print(f"\033[31m{err.__class__.__name__}: {err}\033[0m")
        return ''


def tokenize_chunk(chunk: Chunk) -> list[str]:
    chunk_text_tokens: list[str] = tokenize(chunk.text)
    chunk_path_tokens: list[str] = tokenize(Path(chunk.file_path).stem)
    chunk_title_tokens: list[str] = (
        tokenize(chunk.title) if chunk.title else []
    )
    return (chunk_text_tokens + chunk_path_tokens + chunk_title_tokens)
