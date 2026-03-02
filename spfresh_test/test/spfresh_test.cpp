#include "gtest/gtest.h"
#include "spfresh_index.h"

// TEST(SPFreshIndexTest, GeneralTest) {
//     SPFreshIndex index("/home/xjs/experiment/motivation_test/spfresh_test/config/index_conf/1024dim100K.ini");
//     int dimension = 1024;
    
//     int init_count = 100000;
//     std::vector<float> init_vecs(init_count * dimension);
//     std::vector<uint32_t> init_ids(init_count);
//     for (int i = 0; i < init_count; ++i) {
//         init_ids[i] = i;
//     }
//     for (int i = 0; i < init_count; ++i) {
//         for (int j = 0; j < dimension; ++j) {
//             init_vecs[i * dimension + j] = (float)rand() / RAND_MAX;
//         }
//     }
//     index.build(init_vecs, init_ids);

//     int insert_count = 10;
//     std::vector<float> insert_vecs(insert_count * dimension);
//     std::vector<uint32_t> insert_ids(insert_count);
//     for (int i = 0; i < insert_count; ++i) {
//         insert_ids[i] = i + 1000000;
//     }

//     for (int i = 0; i < insert_count; ++i) {
//         for (int j = 0; j < dimension; ++j) {
//             insert_vecs[i * dimension + j] = (float)rand() / RAND_MAX;
//         }
//     }
//     index.insert(insert_vecs, insert_ids);

//     std::vector<float> query_vec(dimension);
//     for (int i = 0; i < dimension; ++i) {
//         query_vec[i] = (float)rand() / RAND_MAX;
//     }
//     std::vector<uint32_t> search_ids(insert_count);
//     std::vector<float> distances(insert_count);
//     index.search(query_vec, insert_count, search_ids, distances);

//     for (int i = 0; i < insert_count; ++i) {
//         printf("search_ids[%d] = %d, distances[%d] = %f\n", i, search_ids[i], i, distances[i]);
//     }
// }

// TEST(DISABLED_SPFreshIndexTest, GeneralTest) {
TEST(SPFreshIndexTest, GeneralTest) {
    SPFreshIndex index("/home/xjs/experiment/motivation_test/spfresh_test/config/index_conf/1024dim100K.ini");
    int dimension = 1024;
    index.build("/data1/xjs/random_dataset/vec1024dim100K.fvecs");

    int insert_count = 10;
    std::vector<float> insert_vecs(insert_count * dimension);
    std::vector<uint32_t> insert_ids(insert_count);
    for (int i = 0; i < insert_count; ++i) {
        insert_ids[i] = i + 100000;
    }

    for (int i = 0; i < insert_count; ++i) {
        for (int j = 0; j < dimension; ++j) {
            insert_vecs[i * dimension + j] = (float)rand() / RAND_MAX;
        }
    }
    index.insert(insert_vecs, insert_ids);

    std::vector<float> query_vec(dimension);
    for (int i = 0; i < dimension; ++i) {
        query_vec[i] = (float)rand() / RAND_MAX;
    }
    std::vector<uint32_t> search_ids(insert_count);
    std::vector<float> distances(insert_count);
    index.search(query_vec, insert_count, search_ids, distances);

    for (int i = 0; i < insert_count; ++i) {
        printf("search_ids[%d] = %d, distances[%d] = %f\n", i, search_ids[i], i, distances[i]);
    }

    index.save("/data1/xjs/index/spfresh_index/1024dim100K");
}

TEST(SPFreshIndexTest, LoadTest) {
    SPFreshIndex index("/home/xjs/experiment/motivation_test/spfresh_test/config/index_conf/1024dim100K.ini");
    int dimension = 1024;
    index.load("/data1/xjs/index/spfresh_index/1024dim100K");

    int insert_count = 10;
    std::vector<float> insert_vecs(insert_count * dimension);
    std::vector<uint32_t> insert_ids(insert_count);
    for (int i = 0; i < insert_count; ++i) {
        insert_ids[i] = i + 100000;
    }

    for (int i = 0; i < insert_count; ++i) {
        for (int j = 0; j < dimension; ++j) {
            insert_vecs[i * dimension + j] = (float)rand() / RAND_MAX;
        }
    }
    index.insert(insert_vecs, insert_ids);

    std::vector<float> query_vec(dimension);
    for (int i = 0; i < dimension; ++i) {
        query_vec[i] = (float)rand() / RAND_MAX;
    }
    std::vector<uint32_t> search_ids(insert_count);
    std::vector<float> distances(insert_count);
    index.search(query_vec, insert_count, search_ids, distances);

    for (int i = 0; i < insert_count; ++i) {
        printf("search_ids[%d] = %d, distances[%d] = %f\n", i, search_ids[i], i, distances[i]);
    }
}

int main(int argc, char* argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}