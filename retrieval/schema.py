from dataclasses import dataclass, asdict
from enum import Enum

class ChunkType(str, Enum):
    FUNCTION = "function_def"
    METHOD = "method_def"
    CLASS = "class_def"
    PLAINTEXT = "plaintext"
    MARKDOWN = "markdown"
    REPO_OVERVIEW = "repository_structure"

@dataclass
class ChunkSchema:
    id: str # for postgre to access
    content: str
    file_path: str
    chunk_type: ChunkType
    signature: str
    start_line: str
    end_line: str

    def to_metadata(self) -> dict:
        return {
            "file_path": self.file_path,
            "chunk_type": self.chunk_type.value,
            "signature": self.signature,
            "start_line": self.start_line,
            "end_line": self.end_line
        }
    
@dataclass
class RetrievedChunkSchema:
    id: str
    content: str
    file_path: str
    chunk_type: ChunkType
    signature: str
    start_line: str
    end_line: str
    score: float

class RetrievalMode(str, Enum):
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"