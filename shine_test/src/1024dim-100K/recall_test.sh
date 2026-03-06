python test_vector_test.py \
    --option recall \
    --host localhost \
    --port 8080 \
    --index_path /data/xjs/index/shine_index/1024dim100K \
    --query_data /data/xjs/random_dataset/1024dim100K/queries/query-test.fbin \
    --groundtruth /data/xjs/random_dataset/1024dim100K/queries/groundtruth-test.bin \
    --topk 10