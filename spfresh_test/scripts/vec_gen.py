import numpy as np
import struct
import os
from tqdm import tqdm  # 用于显示进度条

def generate_fvecs(filename, num_vectors, dim, batch_size=10000, dtype='float32'):
    """
    高效生成随机向量并保存为fvecs格式文件
    
    参数:
    filename: 输出文件路径
    num_vectors: 向量数量
    dim: 向量维度
    batch_size: 批量处理大小 (默认10000)
    dtype: 数据类型 (默认float32)
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # 计算总批次数
    num_batches = (num_vectors + batch_size - 1) // batch_size
    
    # 预分配维度头数据 (每个向量4字节)
    dim_header = struct.pack('i', dim)
    
    with open(filename, 'wb') as f:
        # 使用进度条显示生成进度
        for batch_idx in tqdm(range(num_batches), desc="生成向量"):
            # 计算当前批次的实际大小
            current_batch_size = min(batch_size, num_vectors - batch_idx * batch_size)
            
            # 批量生成随机向量 (更高效)
            vectors = np.random.random((current_batch_size, dim)).astype(dtype)
            
            # 写入批次数据
            for i in range(current_batch_size):
                f.write(dim_header)  # 写入维度头
                f.write(vectors[i].tobytes())  # 写入向量数据

def generate_fvecs_mmap(filename, num_vectors, dim, dtype='float32'):
    """
    使用内存映射技术生成超大向量集 (适用于超大文件)
    
    参数:
    filename: 输出文件路径
    num_vectors: 向量数量
    dim: 向量维度
    dtype: 数据类型 (默认float32)
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # 计算文件大小 (字节)
    vector_size = 4 + dim * np.dtype(dtype).itemsize  # 4字节头 + 向量数据
    file_size = num_vectors * vector_size
    
    # 创建并预分配文件空间
    with open(filename, 'wb') as f:
        f.seek(file_size - 1)
        f.write(b'\0')
    
    # 使用内存映射写入
    with open(filename, 'r+b') as f:
        mm = np.memmap(f, dtype='uint8', mode='r+', shape=(file_size,))
        
        # 预打包维度头
        dim_header = np.frombuffer(struct.pack('i', dim), dtype='uint8')
        
        # 进度条
        for i in tqdm(range(num_vectors), desc="内存映射生成"):
            # 计算当前向量在文件中的位置
            offset = i * vector_size
            
            # 写入维度头
            mm[offset:offset+4] = dim_header
            
            # 生成并写入向量数据
            vector = np.random.random(dim).astype(dtype)
            mm[offset+4:offset+vector_size] = np.frombuffer(vector.tobytes(), dtype='uint8')

if __name__ == "__main__":
    # 示例使用
    output_file = "/data1/xjs/random_dataset/vec1024dim100K.fvecs"  # 输出文件路径
    vector_dim = 1024                           # 向量维度
    vector_count = 100000                     # 1000000个向量
    
    print(f"开始生成 {vector_count} 个 {vector_dim} 维向量...")
    
    # 根据数据量选择合适的方法
    if vector_count * vector_dim > 10**9:  # 超过10亿元素使用内存映射
        print("使用内存映射方法 (超大数据集)")
        generate_fvecs_mmap(output_file, vector_count, vector_dim)
    else:
        print("使用批量处理方法")
        generate_fvecs(output_file, vector_count, vector_dim, batch_size=50000)
    
    print(f"生成完成! 文件已保存到 {output_file}")
    print(f"预计文件大小: {(4 + vector_dim * 4) * vector_count / (1024**3):.2f} GB")