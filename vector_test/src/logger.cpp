#include "logger.h"
#include "spdlog/sinks/basic_file_sink.h"
#include <spdlog/sinks/stdout_color_sinks.h>

std::shared_ptr<spdlog::logger> GlobalLogger;

void init_global_logger() {
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    GlobalLogger = std::make_shared<spdlog::logger>("GlobalLogger", console_sink);
}

void set_log_level(spdlog::level::level_enum log_level) {
    GlobalLogger->set_level(log_level);
}

void add_file_sink(const std::string& log_path) {
    auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(log_path, true);
    GlobalLogger->sinks().push_back(file_sink);
}