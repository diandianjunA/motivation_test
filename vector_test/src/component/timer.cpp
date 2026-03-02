#include "component/timer.h"

Timer::Timer() {}

Timer::~Timer() {}

void Timer::start() {
    start_time = std::chrono::high_resolution_clock::now();
}

void Timer::pause() {
    end_time = std::chrono::high_resolution_clock::now();
    duration += end_time - start_time;
}

void Timer::resume() {
    start_time = std::chrono::high_resolution_clock::now();
}

void Timer::stop() {
    end_time = std::chrono::high_resolution_clock::now();
    duration += end_time - start_time;
}

double Timer::elapsed() const {
    return duration.count();
}