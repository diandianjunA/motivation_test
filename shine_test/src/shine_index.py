"""
Shine Vector Index implementation using HTTP API.
This module provides a VectorIndex implementation that communicates with
the Shine HTTP server via REST API.
"""

import numpy as np
import requests
from typing import List, Tuple, Optional
import time
import os
import json

from vector_test import VectorIndex


class ShineClient:
    """
    Client for Shine HTTP API.
    Provides methods to interact with the Shine vector index service.
    """

    def __init__(self, host: str = "localhost", port: int = 8080, timeout: int = 60):
        """
        Initialize Shine client.
        
        Args:
            host: Shine server host
            port: Shine server port
            timeout: Request timeout in seconds
        """
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout

    def health_check(self) -> bool:
        """Check if the server is healthy."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.json().get("success", False)
        except Exception:
            return False

    def wait_for_server(self, max_retries: int = 30, retry_interval: int = 1) -> bool:
        """Wait for server to become available."""
        for i in range(max_retries):
            if self.health_check():
                return True
            time.sleep(retry_interval)
        return False

    def insert_vector(self, vector: np.ndarray, vector_id: Optional[int] = None) -> int:
        """
        Insert a vector into the index.
        
        Args:
            vector: numpy array of shape (dim,)
            vector_id: optional vector ID
            
        Returns:
            The ID of the inserted vector
        """
        payload = {"vector": vector.tolist()}
        if vector_id is not None:
            payload["id"] = vector_id

        response = requests.post(
            f"{self.base_url}/insert",
            json=payload,
            timeout=self.timeout
        )
        result = response.json()
        if result.get("success"):
            return result.get("id", vector_id if vector_id is not None else 0)
        else:
            raise RuntimeError(f"Insert failed: {result.get('error')}")

    def insert_vectors_batch(self, vectors: np.ndarray, start_id: int = 0) -> List[int]:
        """
        Insert multiple vectors into the index.
        
        Args:
            vectors: numpy array of shape (n, dim)
            start_id: starting vector ID
            
        Returns:
            List of inserted vector IDs
        """
        ids = []
        for i, vec in enumerate(vectors):
            vec_id = self.insert_vector(vec, start_id + i)
            ids.append(vec_id)
        return ids

    def query_vector(self, query: np.ndarray, k: int = 10, ef_search: int = 128) -> Tuple[List[int], List[float]]:
        """
        Query the index for nearest neighbors.
        
        Args:
            query: numpy array of shape (dim,) or (n, dim)
            k: number of neighbors to return
            ef_search: search parameter
            
        Returns:
            Tuple of (ids, distances)
        """
        if query.ndim == 1:
            query = query.reshape(1, -1)

        payload = {
            "vector": query[0].tolist(),
            "k": k,
            "ef_search": ef_search
        }

        response = requests.post(
            f"{self.base_url}/query",
            json=payload,
            timeout=self.timeout
        )
        result = response.json()
        
        if result.get("success"):
            results = result.get("results", [])
            ids = [r.get("id") for r in results]
            distances = [r.get("distance") for r in results]
            return ids, distances
        else:
            raise RuntimeError(f"Query failed: {result.get('error')}")

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
            f"{self.base_url}/save",
            json=payload,
            timeout=self.timeout
        )
        result = response.json()
        return result.get("success", False)

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
            f"{self.base_url}/load",
            json=payload,
            timeout=self.timeout
        )
        result = response.json()
        return result.get("success", False)

    def get_info(self) -> dict:
        """Get system information."""
        response = requests.get(f"{self.base_url}/info", timeout=5)
        return response.json()


class ShineVectorIndex(VectorIndex):
    """
    VectorIndex implementation for Shine vector database.
    This class wraps the Shine HTTP API to provide a VectorIndex interface.
    """

    def __init__(self, host: str = "localhost", port: int = 8080, 
                 timeout: int = 60, wait_for_server: bool = True):
        """
        Initialize ShineVectorIndex.
        
        Args:
            host: Shine server host
            port: Shine server port
            timeout: Request timeout in seconds
            wait_for_server: Whether to wait for server to become available
        """
        super().__init__()
        self.client = ShineClient(host, port, timeout)
        self._host = host
        self._port = port
        
        if wait_for_server:
            if not self.client.wait_for_server():
                raise RuntimeError(f"Shine server at {host}:{port} is not available")

    def build(self, vecs: np.ndarray, ids: np.ndarray) -> None:
        """
        Build index by inserting all vectors.
        
        Args:
            vecs: numpy array of shape (n, dim)
            ids: numpy array of shape (n,) containing vector IDs
        """
        for i, (vec, vec_id) in enumerate(zip(vecs, ids)):
            self.client.insert_vector(vec.astype(np.float32), int(vec_id))
            if (i + 1) % 1000 == 0:
                print(f"  Built {i + 1}/{len(vecs)} vectors")

    def build_from_file(self, dataset_path: str) -> None:
        """
        Build index from fbin file.
        Note: Shine doesn't support direct file loading, so we read the file
        and insert vectors one by one using the HTTP API.
        
        Args:
            dataset_path: path to the dataset file in fbin format
        """
        from vector_test import read_fbin
        
        print(f"Reading data from {dataset_path}...")
        data, info = read_fbin(dataset_path)
        num_vectors, dim = info
        
        print(f"Building index with {num_vectors} vectors of dimension {dim}...")
        
        ids = np.arange(num_vectors, dtype=np.uint32)
        self.build(data, ids)
        
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
            ids = ids.reshape(1, -1)
        
        for v, vec_id in zip(vec, ids):
            self.client.insert_vector(v.astype(np.float32), int(vec_id))

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
        
        for q in query:
            ids, distances = self.client.query_vector(q.astype(np.float32), k=top_k)
            ids_list.append(ids)
            distances_list.append(distances)
        
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
