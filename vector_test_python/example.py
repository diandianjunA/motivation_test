"""
Example: Using vector_test Python package

This example demonstrates how to create a custom VectorIndex implementation
and use the VectorTest framework for performance testing.
"""

import numpy as np
from typing import List, Tuple

from vector_test import VectorIndex, VectorTest, read_fbin


class SimpleBruteForceIndex(VectorIndex):
    """
    Simple brute-force vector index implementation for demonstration.
    """
    
    def __init__(self) -> None:
        super().__init__()
        self._vectors: np.ndarray = None
        self._ids: np.ndarray = None
        self._dim: int = 0
    
    def build(self, vecs: np.ndarray, ids: np.ndarray) -> None:
        """Build index from vectors and IDs."""
        self._vectors = vecs.astype(np.float32)
        self._ids = ids.astype(np.uint32)
        self._dim = vecs.shape[1]
    
    def build_from_file(self, dataset_path: str) -> None:
        """Build index from fbin file."""
        data, info = read_fbin(dataset_path)
        ids = np.arange(data.shape[0], dtype=np.uint32)
        self.build(data, ids)
    
    def insert(self, vec: np.ndarray, ids: np.ndarray) -> None:
        """Insert vectors into index."""
        vec = vec.reshape(-1, self._dim) if vec.ndim == 1 else vec
        ids = ids.reshape(-1)
        
        if self._vectors is None:
            self._vectors = vec.astype(np.float32)
            self._ids = ids.astype(np.uint32)
        else:
            self._vectors = np.vstack([self._vectors, vec])
            self._ids = np.concatenate([self._ids, ids])
    
    def search(self, query: np.ndarray, top_k: int) -> Tuple[List[int], List[float]]:
        """Search for top-k nearest neighbors using brute force."""
        query = query.reshape(1, -1) if query.ndim == 1 else query
        
        results_ids = []
        results_distances = []
        
        for q in query:
            distances = np.linalg.norm(self._vectors - q, axis=1)
            top_indices = np.argsort(distances)[:top_k]
            
            results_ids.append([int(self._ids[i]) for i in top_indices])
            results_distances.append([float(distances[i]) for i in top_indices])
        
        if len(results_ids) == 1:
            return results_ids[0], results_distances[0]
        return results_ids, results_distances
    
    def load(self, index_path: str) -> None:
        """Load index from file (simplified for demo)."""
        data, info = read_fbin(index_path)
        self.build(data, np.arange(data.shape[0], dtype=np.uint32))
    
    def save(self, index_path: str) -> None:
        """Save index to file (simplified for demo)."""
        np.save(index_path + '_vectors.npy', self._vectors)
        np.save(index_path + '_ids.npy', self._ids)
    
    def get_index_type(self) -> str:
        return "SimpleBruteForce"


def main():
    """Example usage of vector_test package."""
    print("=" * 50)
    print("VectorTest Python Package Example")
    print("=" * 50)
    
    index = SimpleBruteForceIndex()
    
    print("\n1. Creating sample data...")
    dim = 128
    num_vectors = 10000
    vecs = np.random.rand(num_vectors, dim).astype(np.float32)
    ids = np.arange(num_vectors, dtype=np.uint32)
    
    print(f"   Generated {num_vectors} vectors with dimension {dim}")
    
    print("\n2. Building index...")
    index.build(vecs, ids)
    print("   Index built successfully")
    
    print("\n3. Searching...")
    query = np.random.rand(dim).astype(np.float32)
    top_k = 10
    result_ids, result_distances = index.search(query, top_k)
    print(f"   Top {top_k} results:")
    for i, (id_, dist) in enumerate(zip(result_ids, result_distances)):
        print(f"   {i+1}. ID: {id_}, Distance: {dist:.4f}")
    
    print("\n4. Testing insert...")
    new_vec = np.random.rand(dim).astype(np.float32)
    new_id = num_vectors
    index.insert(new_vec, np.array([new_id], dtype=np.uint32))
    print(f"   Inserted vector with ID {new_id}")
    
    print("\n" + "=" * 50)
    print("Example completed!")
    print("=" * 50)


if __name__ == "__main__":
    main()
