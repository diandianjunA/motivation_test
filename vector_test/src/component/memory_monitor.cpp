#include "component/memory_monitor.h"
#include "logger.h"

MemoryMonitor::~MemoryMonitor() {
    stop();
}

// 启动监控
void MemoryMonitor::start() {
    if (running) return;
    running = true;
    monitorThread = std::thread(&MemoryMonitor::monitorLoop, this);
}

// 停止监控
void MemoryMonitor::stop() {
    if (!running) return;
    running = false;
    if (monitorThread.joinable()) {
        monitorThread.join();
    }
}

MemoryMonitor::MemoryInfo MemoryMonitor::getCurrentMemoryUsage() {
    return parseMemInfo();
}

// 获取平均内存使用率
double MemoryMonitor::getAverageMemoryUsage() {
    std::lock_guard<std::mutex> lock(dataMutex);
    if (sampleCount == 0) return 0.0;
    return totalMemoryUsage / sampleCount;
}

// 获取目标进程ID
pid_t MemoryMonitor::getTargetPid() const {
    return targetPid;
}

// 设置新的目标进程ID
void MemoryMonitor::setTargetPid(pid_t newPid) {
    stop();
    {
        std::lock_guard<std::mutex> lock(dataMutex);
        targetPid = newPid;
        totalMemoryUsage = 0;
        sampleCount = 0;
    }
    start();
}

// 监控循环
void MemoryMonitor::monitorLoop() {
    while (running) {
        auto memInfo = parseMemInfo();
        {
            std::lock_guard<std::mutex> lock(dataMutex);
            totalMemoryUsage += memInfo.rssPercentage;
            sampleCount++;
        }
        // 输出每秒钟的内存占用大小，比例等信息
        GlobalLogger->info("MemoryMonitor: pid: {}, rss: {} kB, vsz: {} kB, rssPercentage: {}%", memInfo.pid, memInfo.rss, memInfo.vsz, memInfo.rssPercentage);
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

unsigned long MemoryMonitor::getSystemTotalMemory() {
    std::ifstream file("/proc/meminfo");
    std::string line;
    unsigned long total = 0;

    while (std::getline(file, line)) {
        if (line.find("MemTotal:") != std::string::npos) {
            std::istringstream iss(line);
            std::string key;
            unsigned long value;
            std::string unit;
            iss >> key >> value >> unit;
            total = value;
            break;
        }
    }
    return total;
}

MemoryMonitor::MemoryInfo MemoryMonitor::parseMemInfo() {
    MemoryInfo info = {};
    info.pid = targetPid;

    // 获取系统总内存
    unsigned long systemTotal = getSystemTotalMemory();

    // 打开进程的状态文件
    std::ostringstream filename;
    filename << "/proc/" << targetPid << "/status";
    std::ifstream file(filename.str());
    if (!file.is_open()) {
        ;
        return info;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.find("VmRSS:") != std::string::npos) {
            std::istringstream iss(line);
            std::string key;
            unsigned long value;
            std::string unit;
            iss >> key >> value >> unit;
            info.rss = value; // 单位kB
        }
        else if (line.find("VmSize:") != std::string::npos) {
            std::istringstream iss(line);
            std::string key;
            unsigned long value;
            std::string unit;
            iss >> key >> value >> unit;
            info.vsz = value; // 单位kB
        }
        else if (line.find("RssShmem:") != std::string::npos) {
            std::istringstream iss(line);
            std::string key;
            unsigned long value;
            std::string unit;
            iss >> key >> value >> unit;
            info.shared = value; // 单位kB
        }
        else if (line.find("VmExe:") != std::string::npos) {
            std::istringstream iss(line);
            std::string key;
            unsigned long value;
            std::string unit;
            iss >> key >> value >> unit;
            info.text = value; // 单位kB
        }
        else if (line.find("VmData:") != std::string::npos) {
            std::istringstream iss(line);
            std::string key;
            unsigned long value;
            std::string unit;
            iss >> key >> value >> unit;
            info.data = value; // 单位kB
        }
    }

    // 计算RSS占系统总内存的百分比
    if (systemTotal > 0 && info.rss > 0) {
        info.rssPercentage = (info.rss * 100.0) / systemTotal;
    }

    return info;
}
