#include "config.h"
#include "logger.h"
#include <fstream>
#include <sstream>

std::map<std::string, std::string> readConfig(const std::string& filename) {
    std::ifstream file(filename);
    std::map<std::string, std::string> config;
    if (!file.is_open()) {
        GlobalLogger->error("Failed to open config file: {}", filename);
        throw std::runtime_error("Failed to open config file: " + filename);
    }

    std::string line;
    while (std::getline(file, line)) {
        // 移除行尾回车符（处理Windows/Linux文件差异）
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        
        // 移除行内注释（分号后的所有内容）
        size_t commentPos = line.find(';');
        if (commentPos != std::string::npos) {
            line = line.substr(0, commentPos);
        }
        
        // 跳过空行（可能只剩空格）
        if (line.empty()) continue;
        
        // 跳过行首空白
        size_t start = line.find_first_not_of(" \t");
        if (start == std::string::npos) continue; // 全空白行
        
        // 跳过纯注释行（行首是分号）
        if (line[start] == ';') continue;
        
        std::stringstream ss(line);
        std::string key, value;
        if (std::getline(ss, key, '=') && std::getline(ss, value)) {
            // 去除键/值首尾空白
            auto trim = [](std::string& str) {
                // 去除首部空白
                size_t start = str.find_first_not_of(" \t");
                if (start != std::string::npos) {
                    str = str.substr(start);
                }
                
                // 去除尾部空白
                size_t end = str.find_last_not_of(" \t");
                if (end != std::string::npos) {
                    str = str.substr(0, end + 1);
                }
            };
            
            trim(key);
            trim(value);
            
            if (!key.empty()) {
                config[key] = value;
            }
        }
    }
    return config;
}
