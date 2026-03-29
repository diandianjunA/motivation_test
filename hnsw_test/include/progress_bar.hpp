#pragma once

#include <chrono>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>

class ProgressBar {
private:
    std::string message;
    size_t total;
    size_t current = 0;
    size_t bar_width = 50;
    std::chrono::time_point<std::chrono::steady_clock> start_time;
    bool show_time = true;
    bool show_speed = true;
    std::mutex mtx;

public:
    ProgressBar(const std::string& msg, size_t total_items, bool show_time = true, bool show_speed = true)
        : message(msg), total(total_items), show_time(show_time), show_speed(show_speed) {
        start_time = std::chrono::steady_clock::now();
    }

    void set_current(size_t new_current) {
        std::lock_guard<std::mutex> lock(mtx);
        current = new_current;
        if (current > total) {
            current = total;
        }
    }

    void display() {
        std::lock_guard<std::mutex> lock(mtx);
        if (total == 0) {
            return;
        }

        const float percentage = 100.0f * static_cast<float>(current) / static_cast<float>(total);
        const size_t pos = bar_width * current / total;
        const auto now = std::chrono::steady_clock::now();
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();

        std::cout << "\r" << message << " [";
        for (size_t i = 0; i < bar_width; ++i) {
            if (i < pos) {
                std::cout << "=";
            } else if (i == pos) {
                std::cout << ">";
            } else {
                std::cout << " ";
            }
        }
        std::cout << "] " << std::fixed << std::setprecision(1) << std::setw(5) << percentage << "%";

        if (show_speed && elapsed > 0) {
            const double speed = static_cast<double>(current) * 1000.0 / static_cast<double>(elapsed);
            std::cout << " (" << format_number(speed) << " it/s)";
        }

        if (show_time && elapsed > 0 && current > 0) {
            const double items_per_ms = static_cast<double>(current) / static_cast<double>(elapsed);
            const double remaining_ms = static_cast<double>(total - current) / items_per_ms;
            std::cout << " ETA: " << format_time(remaining_ms);
        }

        std::cout << std::flush;
    }

    void finish() {
        set_current(total);
        display();
        std::cout << std::endl;

        const auto end_time = std::chrono::steady_clock::now();
        const auto total_time =
            std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        if (total_time > 0) {
            const double speed = static_cast<double>(total) * 1000.0 / static_cast<double>(total_time);
            std::cout << "Completed in " << format_time(total_time) << " (" << format_number(speed) << " it/s)"
                      << std::endl;
        }
    }

private:
    static std::string format_time(double ms) {
        if (ms < 1000.0) {
            return std::to_string(static_cast<int>(ms)) + "ms";
        }
        if (ms < 60000.0) {
            return std::to_string(static_cast<int>(ms / 1000.0)) + "s";
        }
        if (ms < 3600000.0) {
            const int minutes = static_cast<int>(ms / 60000.0);
            const int seconds = static_cast<int>((ms - minutes * 60000.0) / 1000.0);
            return std::to_string(minutes) + "m" + std::to_string(seconds) + "s";
        }
        const int hours = static_cast<int>(ms / 3600000.0);
        const int minutes = static_cast<int>((ms - hours * 3600000.0) / 60000.0);
        return std::to_string(hours) + "h" + std::to_string(minutes) + "m";
    }

    static std::string format_number(double num) {
        std::stringstream ss;
        if (num >= 1000000.0) {
            ss << std::fixed << std::setprecision(1) << (num / 1000000.0) << "M";
        } else if (num >= 1000.0) {
            ss << std::fixed << std::setprecision(1) << (num / 1000.0) << "K";
        } else {
            ss << std::fixed << std::setprecision(1) << num;
        }
        return ss.str();
    }
};
