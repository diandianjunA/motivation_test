#include "utils.hh"

#include <cmath>
#include <map>

void lib_failure(const str&& message) {
  std::cerr << "[ERROR]: " << message << std::endl;
  std::exit(EXIT_FAILURE);
}

std::string get_ip(const str& node_name) {
  static const std::map<str, str> node_to_ip{
    {"cluster1", "127.0.0.1"},
    {"cluster2", "192.168.6.201"},
    {"cluster3", "192.168.6.202"},
  };

  const auto it = node_to_ip.find(node_name);
  if (it != node_to_ip.end()) {
    return it->second;
  }
  return node_name;
}

f64 compute_throughput(i32 message_size,
                       i32 repeats,
                       Timepoint start,
                       Timepoint end) {
  return message_size / (ToSeconds(end - start).count() / repeats) /
         std::pow(1000, 2);
}

f64 compute_latency(i32 repeats,
                    Timepoint start,
                    Timepoint end,
                    bool is_read_or_atomic) {
  i32 rtt_factor = is_read_or_atomic ? 1 : 2;
  return ToMicroSeconds(end - start).count() / repeats / rtt_factor;
}

void print_status(str&& status) {
  std::cerr << "[STATUS]: " << status << std::endl;
}
