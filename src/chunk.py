class Chunk:
    def __init__(
        self,
        id: int,
        file_path: str,
        first_character_index: int,
        last_character_index: int,
        text: str
    ):
        self.id: int = id
        self.file_path: str = file_path
        self.first_character_index: int = first_character_index
        self.last_character_index: int = last_character_index
        self.text: str = text
