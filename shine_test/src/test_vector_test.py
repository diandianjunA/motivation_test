#!/usr/bin/env python3
"""
Test script for Shine vector index using vector_test framework.

Usage:
    python test_vector_test.py --option build --data_path <path> --index_path <path>
    python test_vector_test.py --option dynamic --index_path <path> ...
    python test_vector_test.py --option storage --index_path <path> ...
    python test_vector_test.py --option recall --index_path <path> ...
"""

import sys
import os
import argparse
import numpy as np

vector_test_path = os.path.join(os.path.dirname(__file__), '..', 'vector_test_python')
if os.path.exists(vector_test_path):
    sys.path.insert(0, vector_test_path)

from vector_test import VectorTest, read_config
from src.shine_index import ShineVectorIndex


def main():
    parser = argparse.ArgumentParser(description="Test Shine vector index using vector_test framework")
    
    parser.add_argument("--host", default="localhost", help="Shine server host")
    parser.add_argument("--port", type=int, default=8080, help="Shine server port")
    parser.add_argument("--option", required=True, 
                        choices=["build", "storage", "dynamic", "recall"],
                        help="Test option")
    parser.add_argument("--data_path", default="", help="Path to training data (fbin format)")
    parser.add_argument("--index_path", default="", help="Path to save/load index")
    parser.add_argument("--query_data", default="", help="Path to query data (fbin format)")
    parser.add_argument("--groundtruth", default="", help="Path to ground truth data (bin format)")
    parser.add_argument("--dim", type=int, default=0, help="Vector dimension")
    parser.add_argument("--topk", type=int, default=10, help="Top-k for search")
    parser.add_argument("--threads", type=int, default=4, help="Number of threads for dynamic test")
    parser.add_argument("--test_scale", type=int, default=1000, help="Number of operations for test")
    parser.add_argument("--read_ratio", type=float, default=0.5, help="Read ratio for dynamic test")
    parser.add_argument("--total_test", action="store_true", help="Run total test with different read ratios")
    parser.add_argument("--log_path", default="./logs", help="Log file path")
    parser.add_argument("--wait_server", action="store_true", default=False, 
                        help="Wait for server to be available")
    
    args = parser.parse_args()
    
    config = {
        'option': args.option,
        'data_path': args.data_path,
        'index_path': args.index_path,
        'query_data': args.query_data,
        'groundtruth': args.groundtruth,
        'dim': str(args.dim),
        'topk': str(args.topk),
        'threads': str(args.threads),
        'test_scale': str(args.test_scale),
        'read_ratio': str(args.read_ratio),
        'total_test': 'true' if args.total_test else 'false',
        'log_path': args.log_path,
    }
    
    print("=" * 50)
    print("Shine Vector Index Test")
    print("=" * 50)
    print(f"Host: {args.host}:{args.port}")
    print(f"Option: {args.option}")
    for key, value in config.items():
        if value:
            print(f"  {key}: {value}")
    
    index = ShineVectorIndex(
        host=args.host,
        port=args.port,
        wait_for_server=args.wait_server
    )
    
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        for key, value in config.items():
            if value:
                f.write(f"[test]\n")
                f.write(f"{key} = {value}\n")
        config_path = f.name
    
    try:
        test = VectorTest(config_path, index)
    finally:
        os.unlink(config_path)
    
    print("\nTest completed!")


if __name__ == "__main__":
    main()
