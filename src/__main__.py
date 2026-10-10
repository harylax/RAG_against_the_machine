from .index import Indexer
from .bm25 import BM25
from .retrieve import Retriever
from .utils import read_chunk_text


def main() -> None:
    Indexer(400)
    bm25 = BM25()
    retriever = Retriever(bm25)
    res = retriever.search('How do I configure OpenAI server?', 3)
    # res = retriever.search('What is rocm version?', 3)
    # print(res.model_dump_json(indent=2))
    # res = retriever.search('', 3)
    # res = retriever.search('rocm', 0)
    print(f"question id: {res.question_id}\n")
    print(f"question: {res.question}\n")
    for source in res.retrieved_sources:
        print(source.model_dump_json(indent=2))
        print('\n-----------------------------------------')
        print(read_chunk_text(source))
        print('-----------------------------------------\n')


if __name__ == "__main__":
    main()
