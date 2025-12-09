from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
from app.ingest.parser import DocumentChunk

class BaseIndex(ABC):
    @abstractmethod
    def add_documents(self, documents: List[DocumentChunk]):
        pass

    @abstractmethod
    def search(self, query: str, k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        pass
    
    @abstractmethod
    def save(self, path: str):
        pass
    
    @abstractmethod
    def load(self, path: str):
        pass
