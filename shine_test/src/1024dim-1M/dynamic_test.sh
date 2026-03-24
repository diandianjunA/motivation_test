# 动态读写测试
python ../test_vector_test.py \
    --option dynamic \
    --host 192.168.6.201 \
    --port 8080 \
    --index_path /data/xjs/index/shine_index/1024dim1M \
    --dim 1024 \
    --threads 16 \
    --test_scale 5000 \
    --read_ratio 0.5