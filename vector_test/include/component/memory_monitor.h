#include <fstream>
#include <sstream>
#include <thread>
#include <mutex>
#include <vector>
#include <atomic>
#include <chrono>
#include <iomanip>
#include <unistd.h>

class MemoryMonitor {
public:
    MemoryMonitor(pid_t pid = 0) 
        : targetPid(pid == 0 ? getpid() : pid), 
          running(false), 
          totalMemoryUsage(0), 
          sampleCount(0) {};
    
    ~MemoryMonitor();

    // 获取当前内存使用信息
    struct MemoryInfo {
        pid_t pid;             // 进程ID
        unsigned long rss;     // 实际物理内存使用量 (KB)
        unsigned long vsz;     // 虚拟内存大小 (KB)
        unsigned long shared;  // 共享内存大小 (KB)
        unsigned long text;    // 代码段大小 (KB)
        unsigned long data;    // 数据段大小 (KB)
        double rssPercentage; // RSS占系统总内存的百分比
    };

    void start();

    void stop();

    MemoryInfo getCurrentMemoryUsage();

    double getAverageMemoryUsage();

    pid_t getTargetPid() const;

    void setTargetPid(pid_t newPid);

private:
    pid_t targetPid;          // 要监控的进程ID
    std::atomic<bool> running;
    std::thread monitorThread;
    std::mutex dataMutex;
    double totalMemoryUsage;  // 累计RSS百分比
    unsigned long sampleCount; // 采样次数

    void monitorLoop();
    
        // 获取系统总内存
    unsigned long getSystemTotalMemory();

    MemoryInfo parseMemInfo();
};