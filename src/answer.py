from ollama import chat  # type: ignore
from fire import Fire  # type: ignore


def get_response(prompt: str) -> str:
    response = chat(
        model='qwen3:0.6b',
        messages=[
            {'role': 'user', 'content': prompt}
        ]
    )
    return response['message']['content']


if __name__ == "__main__":
    Fire(get_response)
