import struct
import numpy as np
from typing import Tuple, Optional


def read_fbin(file_path: str) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Read bigANN .bin format vector file.
    
    File format:
        - Header: [uint32 num_vectors][uint32 dim]
        - Data: num_vectors * dim * float32
    
    Args:
        file_path: path to the .bin file
        
    Returns:
        Tuple of (data, (num_vectors, dim))
            data: numpy array of shape (num_vectors, dim) with dtype float32
    """
    with open(file_path, 'rb') as f:
        num_vectors = struct.unpack('I', f.read(4))[0]
        dim = struct.unpack('I', f.read(4))[0]
        
        data = np.frombuffer(
            f.read(num_vectors * dim * 4),
            dtype=np.float32
        ).reshape(num_vectors, dim)
        
    return data, (num_vectors, dim)


def read_bin(file_path: str) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Read ground truth file in .bin format.
    
    File format:
        - Header: [uint32 num_queries][uint32 k]
        - Data: num_queries * k * uint32 (neighbor IDs)
    
    Args:
        file_path: path to the ground truth .bin file
        
    Returns:
        Tuple of (data, (num_queries, k))
            data: numpy array of shape (num_queries, k) with dtype uint32
    """
    with open(file_path, 'rb') as f:
        num_queries = struct.unpack('I', f.read(4))[0]
        k = struct.unpack('I', f.read(4))[0]
        
        data = np.frombuffer(
            f.read(num_queries * k * 4),
            dtype=np.uint32
        ).reshape(num_queries, k)
        
    return data, (num_queries, k)


def rand_vec(dim: int, count: int) -> np.ndarray:
    """
    Generate random vectors with values in [0, 1).
    
    Args:
        dim: dimension of each vector
        count: number of vectors to generate
        
    Returns:
        numpy array of shape (count, dim) with dtype float32
    """
    return np.random.rand(count, dim).astype(np.float32)


def rand_vec_inplace(vec: np.ndarray) -> None:
    """
    Fill an existing array with random values in [0, 1).
    
    Args:
        vec: numpy array to fill
    """
    vec[:] = np.random.rand(*vec.shape).astype(np.float32)
