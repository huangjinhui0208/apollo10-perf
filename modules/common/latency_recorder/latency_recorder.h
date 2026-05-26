/******************************************************************************
 * Copyright 2019 The Apollo Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 *****************************************************************************/

#pragma once

#include <string>

#include "cyber/time/time.h"

namespace apollo {
namespace common {

class LatencyRecorder {
 public:
  explicit LatencyRecorder(const std::string& module_name);
  ~LatencyRecorder();

  void AppendLatencyRecord(const uint64_t message_id,
                           const apollo::cyber::Time& begin_time,
                           const apollo::cyber::Time& end_time);

  // Keep this method for compatibility with existing latency_trace.cc.
  // In local CSV mode it flushes the process-level local CSV sink.
  void Flush();

  // Flush all process-local latency records to disk.
  static void FlushProcess();

  // Flush and stop the process-local writer thread.
  // This is intended for atexit path.
  static void ShutdownProcess();

  // Print process-level counters.
  static void LogGlobalStats(const std::string& reason);

 private:
  LatencyRecorder() = default;

  std::string module_name_;
};

}  // namespace common
}  // namespace apollo
