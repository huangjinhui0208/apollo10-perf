/******************************************************************************
 * Copyright 2019 The Apollo Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 *****************************************************************************/

#include "modules/common/latency_recorder/latency_recorder.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unistd.h>

#include "cyber/common/global_data.h"
#include "cyber/common/log.h"
#include "gflags/gflags.h"

#include "modules/common/perf/perf_gflags.h"

using apollo::cyber::Time;

namespace apollo {
namespace common {
namespace {

struct CsvEvent {
  std::string module_name;
  uint64_t trace_id_ns = 0;
  uint64_t begin_time_ns = 0;
  uint64_t end_time_ns = 0;
};

std::string ShellQuote(const std::string& s) {
  std::string out = "'";
  for (char c : s) {
    if (c == '\'') {
      out += "'\\''";
    } else {
      out += c;
    }
  }
  out += "'";
  return out;
}

bool EnsureDir(const std::string& dir) {
  if (dir.empty()) {
    return false;
  }
  const std::string cmd = "mkdir -p " + ShellQuote(dir);
  return std::system(cmd.c_str()) == 0;
}

std::string ReadFirstLine(const std::string& file_path) {
  std::ifstream ifs(file_path);
  if (!ifs.is_open()) {
    return "";
  }

  std::string line;
  std::getline(ifs, line);

  while (!line.empty() &&
         (line.back() == '\n' || line.back() == '\r' ||
          line.back() == ' ' || line.back() == '\t')) {
    line.pop_back();
  }
  return line;
}

std::string SanitizeName(std::string s) {
  for (auto& c : s) {
    const bool ok =
        (c >= 'a' && c <= 'z') ||
        (c >= 'A' && c <= 'Z') ||
        (c >= '0' && c <= '9') ||
        c == '_' || c == '-' || c == '.';
    if (!ok) {
      c = '_';
    }
  }
  if (s.empty()) {
    s = "unknown";
  }
  return s;
}

std::string ProgramName() {
  const std::string& binary_name =
      apollo::cyber::common::GlobalData::Instance()->ProcessGroup();
  if (!binary_name.empty()) {
    return SanitizeName(binary_name);
  }

  return "apollo_process";
}

std::string ResolveOutputRoot() {
  const char* env = std::getenv("APOLLO_LATENCY_OUTPUT_DIR");
  if (env != nullptr && std::string(env).size() > 0) {
    return std::string(env);
  }
  return FLAGS_latency_output_dir;
}

std::string ResolveRunId(const std::string& output_root) {
  const char* env = std::getenv("APOLLO_LATENCY_RUN_ID");
  if (env != nullptr && std::string(env).size() > 0) {
    return std::string(env);
  }

  // This file is written by run_latency_experiment.sh.
  // It solves the Dreamview/cyber_launch case where module processes
  // may not inherit shell environment variables.
  const std::string active_run_file = output_root + "/active_run_id";
  const std::string active_run_id = ReadFirstLine(active_run_file);
  if (!active_run_id.empty()) {
    return active_run_id;
  }

  return FLAGS_latency_run_id;
}

std::string CsvLine(const std::string& run_id, const CsvEvent& e) {
  const uint64_t dur_ns =
      (e.end_time_ns >= e.begin_time_ns) ? (e.end_time_ns - e.begin_time_ns) : 0;

  return run_id + "," +
         e.module_name + "," +
         std::to_string(e.trace_id_ns) + "," +
         std::to_string(e.begin_time_ns) + "," +
         std::to_string(e.end_time_ns) + "," +
         std::to_string(dur_ns) + "\n";
}

class LocalLatencyCsvSink {
 public:
  static LocalLatencyCsvSink* Instance() {
    // Intentionally leaked singleton. This avoids static destruction order
    // problems with latency_trace.cc's atexit flusher.
    static LocalLatencyCsvSink* sink = new LocalLatencyCsvSink();
    return sink;
  }

  void Append(const std::string& module_name,
              uint64_t trace_id_ns,
              uint64_t begin_time_ns,
              uint64_t end_time_ns) {
    if (!FLAGS_latency_record_enable_local_csv) {
      return;
    }

    InitOnce();

    CsvEvent e;
    e.module_name = module_name;
    e.trace_id_ns = trace_id_ns;
    e.begin_time_ns = begin_time_ns;
    e.end_time_ns = end_time_ns;

    const uint64_t accepted =
        accepted_records_.fetch_add(1, std::memory_order_relaxed) + 1;

    size_t queue_size = 0;
    {
      std::lock_guard<std::mutex> lk(mu_);
      queue_.emplace_back(std::move(e));
      queue_size = queue_.size();
      UpdatePeakQueueSizeLocked(queue_size);
    }

    if (FLAGS_latency_record_local_queue_warn_size > 0 &&
        queue_size >
            static_cast<size_t>(FLAGS_latency_record_local_queue_warn_size)) {
      AERROR_EVERY(1000)
          << "[LATENCY_LOCAL_CSV_QUEUE_LARGE] "
          << "pid=" << getpid()
          << " queue_size=" << queue_size
          << " accepted_records=" << accepted
          << " written_records=" << written_records_.load();
    }

    cv_.notify_one();
  }

  void Flush() {
    if (!FLAGS_latency_record_enable_local_csv) {
      return;
    }

    InitOnce();

    const uint64_t target_records =
        accepted_records_.load(std::memory_order_acquire);

    const uint64_t request_id =
        flush_request_id_.fetch_add(1, std::memory_order_acq_rel) + 1;

    cv_.notify_one();

    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(10);

    while (std::chrono::steady_clock::now() < deadline) {
      const uint64_t written =
          written_records_.load(std::memory_order_acquire);
      const uint64_t done_flush =
          flush_done_id_.load(std::memory_order_acquire);

      if (written >= target_records && done_flush >= request_id) {
        return;
      }

      cv_.notify_one();
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    AERROR << "[LATENCY_LOCAL_CSV_FLUSH_TIMEOUT] "
           << "pid=" << getpid()
           << " target_records=" << target_records
           << " written_records=" << written_records_.load()
           << " request_id=" << request_id
           << " flush_done_id=" << flush_done_id_.load();
  }

  void Shutdown() {
    if (!FLAGS_latency_record_enable_local_csv) {
      return;
    }

    InitOnce();
    Flush();

    {
      std::lock_guard<std::mutex> lk(mu_);
      running_.store(false, std::memory_order_release);
    }
    cv_.notify_one();

    if (writer_thread_.joinable()) {
      writer_thread_.join();
    }

    LogStats("Shutdown");
  }

  void LogStats(const std::string& reason) {
    AINFO << "[LATENCY_LOCAL_CSV_TOTAL] "
          << "reason=" << reason
          << " pid=" << getpid()
          << " run_id=" << run_id_
          << " file=" << file_path_
          << " accepted_records=" << accepted_records_.load()
          << " written_records=" << written_records_.load()
          << " peak_queue_size=" << peak_queue_size_.load()
          << " flush_request_id=" << flush_request_id_.load()
          << " flush_done_id=" << flush_done_id_.load();
  }

 private:
  LocalLatencyCsvSink() = default;

  void InitOnce() {
    std::call_once(init_once_, [this]() {
      output_root_ = ResolveOutputRoot();
      run_id_ = ResolveRunId(output_root_);

      const std::string run_dir = output_root_ + "/" + run_id_;
      const std::string parts_dir = run_dir + "/parts";

      if (!EnsureDir(parts_dir)) {
        AERROR << "[LATENCY_LOCAL_CSV] failed to create parts dir: "
               << parts_dir;
      }

      const std::string program = ProgramName();
      file_path_ = parts_dir + "/latency_events." + program + "." +
                   std::to_string(static_cast<int>(getpid())) + ".csv";

      ofs_.open(file_path_, std::ios::out | std::ios::trunc);
      if (!ofs_.is_open()) {
        AERROR << "[LATENCY_LOCAL_CSV] failed to open file: " << file_path_;
        return;
      }

      ofs_ << "run_id,module_name,trace_id_ns,begin_time_ns,end_time_ns,"
              "duration_ns\n";
      ofs_.flush();

      running_.store(true, std::memory_order_release);
      writer_thread_ = std::thread(&LocalLatencyCsvSink::WriterLoop, this);

      AINFO << "[LATENCY_LOCAL_CSV_START] "
            << "pid=" << getpid()
            << " run_id=" << run_id_
            << " output_root=" << output_root_
            << " file=" << file_path_
            << " flush_interval_ms="
            << FLAGS_latency_record_local_flush_interval_ms;
    });
  }

  void UpdatePeakQueueSizeLocked(size_t queue_size) {
    uint64_t old_peak = peak_queue_size_.load(std::memory_order_relaxed);
    while (queue_size > old_peak &&
           !peak_queue_size_.compare_exchange_weak(
               old_peak, static_cast<uint64_t>(queue_size),
               std::memory_order_relaxed)) {
    }
  }

  void WriterLoop() {
    std::deque<CsvEvent> local;
    auto last_flush = std::chrono::steady_clock::now();
    uint64_t last_status_written = 0;
    auto last_status = std::chrono::steady_clock::now();

    while (true) {
      const uint64_t current_request =
          flush_request_id_.load(std::memory_order_acquire);

      {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait_for(
            lk,
            std::chrono::milliseconds(
                FLAGS_latency_record_local_flush_interval_ms),
            [this, current_request]() {
              return !queue_.empty() ||
                     !running_.load(std::memory_order_acquire) ||
                     flush_done_id_.load(std::memory_order_acquire) <
                         current_request;
            });

        local.swap(queue_);
      }

      while (!local.empty()) {
        if (ofs_.is_open()) {
          ofs_ << CsvLine(run_id_, local.front());
        }
        local.pop_front();
        written_records_.fetch_add(1, std::memory_order_relaxed);
      }

      const auto now = std::chrono::steady_clock::now();
      const auto elapsed_ms =
          std::chrono::duration_cast<std::chrono::milliseconds>(
              now - last_flush)
              .count();

      const uint64_t requested =
          flush_request_id_.load(std::memory_order_acquire);
      const bool need_forced_flush =
          flush_done_id_.load(std::memory_order_acquire) < requested;

      if (ofs_.is_open() &&
          (need_forced_flush ||
           elapsed_ms >= FLAGS_latency_record_local_flush_interval_ms)) {
        ofs_.flush();
        last_flush = now;
        flush_done_id_.store(requested, std::memory_order_release);
      }

      const auto status_elapsed_sec =
          std::chrono::duration_cast<std::chrono::seconds>(
              now - last_status)
              .count();

      if (status_elapsed_sec >= FLAGS_latency_record_local_status_interval_sec) {
        const uint64_t written = written_records_.load();
        if (written != last_status_written) {
          AINFO << "[LATENCY_LOCAL_CSV_STATUS] "
                << "pid=" << getpid()
                << " accepted_records=" << accepted_records_.load()
                << " written_records=" << written
                << " peak_queue_size=" << peak_queue_size_.load()
                << " file=" << file_path_;
          last_status_written = written;
        }
        last_status = now;
      }

      if (!running_.load(std::memory_order_acquire)) {
        std::lock_guard<std::mutex> lk(mu_);
        if (queue_.empty()) {
          break;
        }
      }
    }

    // Final drain.
    {
      std::lock_guard<std::mutex> lk(mu_);
      local.swap(queue_);
    }

    while (!local.empty()) {
      if (ofs_.is_open()) {
        ofs_ << CsvLine(run_id_, local.front());
      }
      local.pop_front();
      written_records_.fetch_add(1, std::memory_order_relaxed);
    }

    if (ofs_.is_open()) {
      ofs_.flush();
      ofs_.close();
    }

    const uint64_t requested =
        flush_request_id_.load(std::memory_order_acquire);
    flush_done_id_.store(requested, std::memory_order_release);

    LogStats("WriterLoopExit");
  }

  std::once_flag init_once_;

  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<CsvEvent> queue_;

  std::thread writer_thread_;
  std::ofstream ofs_;

  std::atomic<bool> running_{false};

  std::atomic<uint64_t> accepted_records_{0};
  std::atomic<uint64_t> written_records_{0};
  std::atomic<uint64_t> peak_queue_size_{0};

  std::atomic<uint64_t> flush_request_id_{0};
  std::atomic<uint64_t> flush_done_id_{0};

  std::string output_root_;
  std::string run_id_;
  std::string file_path_;
};

}  // namespace

LatencyRecorder::LatencyRecorder(const std::string& module_name)
    : module_name_(module_name) {}

LatencyRecorder::~LatencyRecorder() {}

void LatencyRecorder::AppendLatencyRecord(const uint64_t message_id,
                                          const Time& begin_time,
                                          const Time& end_time) {
  if (begin_time >= end_time) {
    static const int kErrorReduceBase = 1000;
    if (!cyber::common::GlobalData::Instance()->IsRealityMode()) {
      AERROR_EVERY(kErrorReduceBase)
          << "latency begin_time: " << begin_time
          << " >= end_time: " << end_time
          << ", " << kErrorReduceBase << " times";
      return;
    }

    AERROR << "latency begin_time: " << begin_time
           << " >= end_time: " << end_time;
    return;
  }

  if (message_id == 0) {
    return;
  }

  if (FLAGS_latency_record_enable_local_csv) {
    LocalLatencyCsvSink::Instance()->Append(
        module_name_,
        message_id,
        begin_time.ToNanosecond(),
        end_time.ToNanosecond());
    return;
  }

  // Legacy Cyber publishing is intentionally removed from the strict path.
  // If you need the old /apollo/common/latency_records monitor path later,
  // add it behind FLAGS_latency_record_enable_cyber_publish.
}

void LatencyRecorder::Flush() {
  FlushProcess();
}

void LatencyRecorder::FlushProcess() {
  LocalLatencyCsvSink::Instance()->Flush();
}

void LatencyRecorder::ShutdownProcess() {
  LocalLatencyCsvSink::Instance()->Shutdown();
}

void LatencyRecorder::LogGlobalStats(const std::string& reason) {
  LocalLatencyCsvSink::Instance()->LogStats(reason);
}

}  // namespace common
}  // namespace apollo
