"""
Shine Vector Index implementation using HTTP API.
This module provides a VectorIndex implementation that communicates with
the Shine HTTP server via REST API.
"""

import numpy as np
import requests
from typing import List, Tuple
import time

from vector_test import VectorIndex


class ShineClient:
    """
    Client for Shine HTTP API.
    Provides methods to interact with the Shine vector index service.
    """

    def __init__(self, host: str = "localhost", port: int = 8080, timeout: int = 300):
        """
        Initialize Shine client.

        Args:
            host: Shine server host
            port: Shine server port
            timeout: Request timeout in seconds (default 300s for large vectors)
        """
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout

    def health_check(self) -> bool:
        """Check if the server is healthy."""
        try:
            response = requests.get(f"{self.base_url}/status", timeout=5)
            return response.json().get("status") == "running"
        except Exception:
            return False

    def wait_for_server(self, max_retries: int = 30, retry_interval: int = 1) -> bool:
        """Wait for server to become available."""
        for i in range(max_retries):
            if self.health_check():
                return True
            time.sleep(retry_interval)
        return False

    def insert_vectors(self, vectors: np.ndarray, ids: np.ndarray, batch_size: int = 100) -> int:
        """
        Insert vectors into the index via batch API.

        Args:
            vectors: numpy array of shape (n, dim)
            ids: numpy array of shape (n,) containing vector IDs
            batch_size: number of vectors per request

        Returns:
            Total number of successfully inserted vectors
        """
        total_inserted = 0
        n = len(vectors)
        for offset in range(0, n, batch_size):
            end = min(offset + batch_size, n)
            batch = []
            for i in range(offset, end):
                batch.append({"id": int(ids[i]), "values": vectors[i].tolist()})
            response = requests.post(
                f"{self.base_url}/vectors",
                json={"vectors": batch},
                timeout=self.timeout
            )
            result = response.json()
            total_inserted += result.get("inserted", 0)
        return total_inserted

    def query_vector(self, query: np.ndarray, k: int = 10) -> List[int]:
        """
        Query the index for nearest neighbors.

        Args:
            query: numpy array of shape (dim,) or (n, dim)
            k: number of neighbors to return

        Returns:
            List of vector IDs of the nearest neighbors
        """
        if query.ndim == 1:
            query = query.reshape(1, -1)

        payload = {
            "vector": query[0].tolist(),
            "k": k
        }

        response = requests.post(
            f"{self.base_url}/search",
            json=payload,
            timeout=self.timeout
        )
        result = response.json()
        return result.get("results", [])

    def save_index(self, path: str = "") -> bool:
        """
        Save index to file.

        Args:
            path: path to save the index

        Returns:
            True if successful
        """
        payload = {"path": path}

        response = requests.post(
            f"{self.base_url}/index/store",
            json=payload,
            timeout=self.timeout
        )
        result = response.json()
        return result.get("status") == "ok"

    def load_index(self, path: str = "") -> bool:
        """
        Load index from file.

        Args:
            path: path to load the index from

        Returns:
            True if successful
        """
        payload = {"path": path}

        response = requests.post(
            f"{self.base_url}/index/load",
            json=payload,
            timeout=self.timeout
        )
        result = response.json()
        return result.get("status") == "ok"

    def get_status(self) -> dict:
        """Get server status information."""
        response = requests.get(f"{self.base_url}/status", timeout=5)
        return response.json()


class ShineVectorIndex(VectorIndex):
    """
    VectorIndex implementation for Shine vector database.
    This class wraps the Shine HTTP API to provide a VectorIndex interface.
    """

    def __init__(self, host: str = "localhost", port: int = 8080,
                 timeout: int = 300, wait_for_server: bool = True):
        """
        Initialize ShineVectorIndex.

        Args:
            host: Shine server host
            port: Shine server port
            timeout: Request timeout in seconds (default 300s for large vectors)
            wait_for_server: Whether to wait for server to become available
        """
        super().__init__()
        self.client = ShineClient(host, port, timeout)
        self._host = host
        self._port = port
        
        if wait_for_server:
            if not self.client.wait_for_server():
                raise RuntimeError(f"Shine server at {host}:{port} is not available")

    def build(self, vecs: np.ndarray, ids: np.ndarray, num_threads: int = 4) -> None:
        """
        Build index by inserting all vectors.
        
        Args:
            vecs: numpy array of shape (n, dim)
            ids: numpy array of shape (n,) containing vector IDs
            num_threads: number of threads for parallel insertion
        """
        total = len(vecs)
        batch_size = 100
        print(f"Building index with {total} vectors using batch insert (batch_size={batch_size})...")

        inserted = self.client.insert_vectors(vecs.astype(np.float32), ids, batch_size=batch_size)

        print(f"  Built {inserted}/{total} vectors")

    def build_from_file(self, dataset_path: str, num_threads: int = 4) -> None:
        """
        Build index from fbin file.
        Note: Shine doesn't support direct file loading, so we read the file
        and insert vectors one by one using the HTTP API.
        
        Args:
            dataset_path: path to the dataset file in fbin format
            num_threads: number of threads for parallel insertion
        """
        from vector_test import read_fbin
        
        print(f"Reading data from {dataset_path}...")
        data, info = read_fbin(dataset_path)
        num_vectors, dim = info
        
        print(f"Building index with {num_vectors} vectors of dimension {dim}...")
        
        ids = np.arange(num_vectors, dtype=np.uint32)
        self.build(data, ids, num_threads)
        
        print(f"Index built with {num_vectors} vectors")

    def insert(self, vec: np.ndarray, ids: np.ndarray) -> None:
        """
        Insert vectors into the index.

        Args:
            vec: numpy array of shape (dim,) or (n, dim)
            ids: numpy array of shape (n,) containing vector IDs
        """
        if vec.ndim == 1:
            vec = vec.reshape(1, -1)
            ids = ids.reshape(-1)

        self.client.insert_vectors(vec.astype(np.float32), ids)

    def search(self, query: np.ndarray, top_k: int) -> Tuple[List[int], List[float]]:
        """
        Search for top-k nearest neighbors.
        
        Args:
            query: numpy array of shape (dim,) or (n, dim)
            top_k: number of nearest neighbors to return
            
        Returns:
            Tuple of (ids, distances)
        """
        if query.ndim == 1:
            query = query.reshape(1, -1)
        
        ids_list = []
        distances_list = []
        
        for i, q in enumerate(query):
            ids = self.client.query_vector(q.astype(np.float32), k=top_k)
            ids_list.append(ids)
            distances_list.append([0.0] * top_k)
        
        if len(ids_list) == 1:
            return ids_list[0], distances_list[0]
        return ids_list, distances_list

    def load(self, index_path: str) -> None:
        """
        Load index from file.
        
        Args:
            index_path: path to the index file
        """
        if not self.client.load_index(index_path):
            raise RuntimeError(f"Failed to load index from {index_path}")

    def save(self, index_path: str) -> None:
        """
        Save index to file.
        
        Args:
            index_path: path where to save the index
        """
        if not self.client.save_index(index_path):
            raise RuntimeError(f"Failed to save index to {index_path}")

    def get_index_type(self) -> str:
        """Get the type of the index."""
        return "Shine"
