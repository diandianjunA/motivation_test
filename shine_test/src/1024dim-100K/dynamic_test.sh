# 动态读写测试
python test_vector_test.py \
    --option dynamic \
    --host localhost \
    --port 8080 \
    --index_path /path/to/index \
    --dim 128 \
    --threads 4 \
    --test_scale 1000 \
    --read_ratio 0.5