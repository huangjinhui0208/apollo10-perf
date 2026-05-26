#pragma once

#include "gflags/gflags.h"

DECLARE_bool(enable_latency_trace);
DECLARE_string(latency_run_id);
DECLARE_string(latency_output_dir);
DECLARE_bool(enable_resource_sampler);
DECLARE_int32(resource_sample_hz);

// New local CSV latency sink.
// When enabled, Apollo module processes write latency records directly
// into per-process CSV part files. This bypasses /apollo/common/latency_records
// and latency_csv_exporter.
DECLARE_bool(latency_record_enable_local_csv);

// Legacy Cyber topic publishing. Keep false for strict experiment collection.
DECLARE_bool(latency_record_enable_cyber_publish);

DECLARE_int32(latency_record_local_flush_interval_ms);
DECLARE_int32(latency_record_local_status_interval_sec);
DECLARE_int64(latency_record_local_queue_warn_size);

// When enabled, low-value high-frequency events are filtered before
// LatencyRecorder objects are created.
DECLARE_bool(latency_trace_enable_event_filter);

DECLARE_string(latency_trace_block_suffixes);
DECLARE_string(latency_trace_block_substrings);
