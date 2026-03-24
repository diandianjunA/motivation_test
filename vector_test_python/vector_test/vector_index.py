from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import numpy as np


class VectorIndex(ABC):
    """
    Abstract base class for vector index.
    Defines the standard interface for vector indexing implementations.
    """

    def __init__(self) -> None:
        pass

    @abstractmethod
    def build(self, vecs: np.ndarray, ids: np.ndarray, num_threads: int = 4) -> None:
        """
        Build index from vectors and their IDs.
        
        Args:
            vecs: numpy array of shape (n, dim) containing vector data
            ids: numpy array of shape (n,) containing vector IDs
            num_threads: number of threads for parallel insertion
        """
        pass

    @abstractmethod
    def build_from_file(self, dataset_path: str, num_threads: int = 4) -> None:
        """
        Build index from a data file.
        
        Args:
            dataset_path: path to the dataset file in fbin format
            num_threads: number of threads for parallel insertion
        """
        pass

    @abstractmethod
    def insert(self, vec: np.ndarray, ids: np.ndarray) -> None:
        """
        Insert vectors into the index.
        
        Args:
            vec: numpy array of shape (dim,) or (n, dim) for single or multiple vectors
            ids: numpy array of shape (n,) containing vector IDs
        """
        pass

    @abstractmethod
    def search(self, query: np.ndarray, top_k: int) -> Tuple[List[int], List[float]]:
        """
        Search for top-k nearest neighbors.
        
        Args:
            query: numpy array of shape (dim,) or (n, dim) for single or multiple queries
            top_k: number of nearest neighbors to return
            
        Returns:
            Tuple of (ids, distances) where:
                ids: list of lists containing top-k IDs
                distances: list of lists containing distances to top-k neighbors
        """
        pass

    @abstractmethod
    def load(self, index_path: str) -> None:
        """
        Load index from file.
        
        Args:
            index_path: path to the index file
        """
        pass

    @abstractmethod
    def save(self, index_path: str) -> None:
        """
        Save index to file.
        
        Args:
            index_path: path where to save the index
        """
        pass

    @abstractmethod
    def get_index_type(self) -> str:
        """
        Get the type of the index.
        
        Returns:
            String identifier for the index type
        """
        pass
