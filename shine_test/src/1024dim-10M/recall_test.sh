python ../test_vector_test.py \
    --option recall \
    --host 192.168.6.201 \
    --port 8080 \
    --index_path /data/xjs/index/shine_index/1024dim10M \
    --query_data /data/xjs/random_dataset/1024dim10M/queries/query-test.fbin \
    --groundtruth /data/xjs/random_dataset/1024dim10M/queries/groundtruth-test.bin \
    --threads "${THREADS:-8}" \
    --topk 10
