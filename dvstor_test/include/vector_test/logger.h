#pragma once
#include <spdlog/spdlog.h>

extern std::shared_ptr<spdlog::logger> GlobalLogger;

// 初始化全局日志记录器
void init_global_logger();

// 设置日志级别
void set_log_level(spdlog::level::level_enum log_level);

// 添加日志输出到文件
void add_file_sink(const std::string& log_path);