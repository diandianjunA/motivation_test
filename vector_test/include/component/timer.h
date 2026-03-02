#include <chrono>

class Timer {
public:
    Timer();

    ~Timer();

    void start();

    void pause();

    void resume();

    void stop();

    double elapsed() const; // 返回经过的时间，单位为秒
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
    std::chrono::time_point<std::chrono::high_resolution_clock> end_time;
    std::chrono::duration<double> duration;
};
