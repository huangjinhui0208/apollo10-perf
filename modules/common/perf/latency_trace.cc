#include "modules/common/perf/latency_trace.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "cyber/common/log.h"
#include "cyber/time/clock.h"
#include "cyber/time/time.h"
#include "modules/common/latency_recorder/latency_recorder.h"
#include "modules/common/perf/perf_gflags.h"

namespace apollo {
namespace perf {
namespace {

std::mutex g_recorder_mu;
std::unordered_map<std::string, std::unique_ptr<apollo::common::LatencyRecorder>>
    g_recorders;

std::vector<std::string> SplitCsv(const std::string& text) {
  std::vector<std::string> result;
  std::stringstream ss(text);
  std::string item;
  while (std::getline(ss, item, ',')) {
    item.erase(item.begin(), std::find_if(item.begin(), item.end(), [](int ch) {
                 return !std::isspace(ch);
               }));
    item.erase(std::find_if(item.rbegin(), item.rend(), [](int ch) {
                 return !std::isspace(ch);
               }).base(),
               item.end());
    if (!item.empty()) {
      result.emplace_back(item);
    }
  }
  return result;
}

bool EndsWith(const std::string& text, const std::string& suffix) {
  if (suffix.size() > text.size()) {
    return false;
  }
  return std::equal(suffix.rbegin(), suffix.rend(), text.rbegin());
}

bool IsEventEnabled(const std::string& event_name) {
  if (!FLAGS_latency_trace_enable_event_filter) {
    return true;
  }

  static std::string cached_suffix_flag;
  static std::vector<std::string> cached_suffixes;
  static std::string cached_substring_flag;
  static std::vector<std::string> cached_substrings;
  static std::mutex cache_mu;

  {
    std::lock_guard<std::mutex> lk(cache_mu);
    if (cached_suffix_flag != FLAGS_latency_trace_block_suffixes) {
      cached_suffix_flag = FLAGS_latency_trace_block_suffixes;
      cached_suffixes = SplitCsv(cached_suffix_flag);
    }
    if (cached_substring_flag != FLAGS_latency_trace_block_substrings) {
      cached_substring_flag = FLAGS_latency_trace_block_substrings;
      cached_substrings = SplitCsv(cached_substring_flag);
    }
  }

  for (const auto& suffix : cached_suffixes) {
    if (EndsWith(event_name, suffix)) {
      return false;
    }
  }

  for (const auto& token : cached_substrings) {
    if (event_name.find(token) != std::string::npos) {
      return false;
    }
  }

  return true;
}

void FlushAllRecordersAtExit() {
  apollo::common::LatencyRecorder::ShutdownProcess();
}

struct ExitFlusherRegister {
  ExitFlusherRegister() { std::atexit(FlushAllRecordersAtExit); }
} g_exit_flusher_register;

apollo::common::LatencyRecorder* GetRecorder(const std::string& event_name) {
  std::lock_guard<std::mutex> lk(g_recorder_mu);
  auto it = g_recorders.find(event_name);
  if (it == g_recorders.end()) {
    it = g_recorders
             .emplace(event_name,
                      std::make_unique<apollo::common::LatencyRecorder>(
                          event_name))
             .first;
  }
  return it->second.get();
}

}  // namespace

uint64_t NowNs() { return apollo::cyber::Clock::Now().ToNanosecond(); }

uint64_t TraceIdFromHeader(const apollo::common::Header& header) {
  if (header.has_lidar_timestamp() && header.lidar_timestamp() != 0) {
    return header.lidar_timestamp();
  }
  if (header.has_camera_timestamp() && header.camera_timestamp() != 0) {
    return header.camera_timestamp();
  }
  if (header.has_radar_timestamp() && header.radar_timestamp() != 0) {
    return header.radar_timestamp();
  }
  // if (header.has_timestamp_sec() && header.timestamp_sec() > 0.0) {
  //   return static_cast<uint64_t>(header.timestamp_sec() * 1e9);
  // }
  return 0;
}

void AppendLatency(const std::string& event_name,
                   uint64_t trace_id,
                   uint64_t begin_ns,
                   uint64_t end_ns) {
  if (!FLAGS_enable_latency_trace || trace_id == 0) {
    return;
  }

  // Filter before GetRecorder(event_name). This is important: filtering after
  // GetRecorder still creates extra LatencyRecorder instances and does not
  // reduce /apollo/common/latency_records burst sources.
  if (!IsEventEnabled(event_name)) {
    return;
  }

  if (end_ns <= begin_ns) {
    if (begin_ns == std::numeric_limits<uint64_t>::max()) {
      return;
    }
    end_ns = begin_ns + 1;
  }

  apollo::cyber::Time begin(begin_ns);
  apollo::cyber::Time end(end_ns);
  GetRecorder(event_name)->AppendLatencyRecord(trace_id, begin, end);
}

void AppendPoint(const std::string& event_name,
                 uint64_t trace_id,
                 uint64_t t_ns) {
  if (t_ns == std::numeric_limits<uint64_t>::max()) {
    return;
  }
  AppendLatency(event_name, trace_id, t_ns, t_ns + 1);
}

void FlushAllRecorders() {
  apollo::common::LatencyRecorder::FlushProcess();
  apollo::common::LatencyRecorder::LogGlobalStats("FlushAllRecorders");
}

}  // namespace perf
}  // namespace apollo
