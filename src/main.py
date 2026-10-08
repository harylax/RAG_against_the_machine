from chunk import write_processed_chunks


def main() -> None:
    write_processed_chunks(
        'data/raw/vllm-0.10.1',
        'data/processed/chunks.json',
        1600
        )


if __name__ == "__main__":
    main()
