#include "random_data_generator.h"
#include "ground_truth_calculator.h"

// 实验配置管理器
class ExperimentGenerator {
private:
    DataConfig config_;
    RandomDataGenerator data_gen_;
    GroundTruthCalculator gt_calc_;
    
public:
    ExperimentGenerator(const DataConfig& config, 
                       DistanceMetric metric = DistanceMetric::L2);
    
    // 生成完整的数据集和ground truth
    void generate_experiment(const std::string& exp_name, 
                           bool save_vectors = true);
    
    // 改进的时间格式化函数
    std::string format_time_detailed(double ms);
    
private:
    void save_config(const std::string& filename);
    
    void print_statistics(const std::vector<std::vector<float>>& database,
                         const std::vector<std::vector<float>>& queries,
                         const std::vector<std::vector<GroundTruthCalculator::Neighbor>>& ground_truth);
};