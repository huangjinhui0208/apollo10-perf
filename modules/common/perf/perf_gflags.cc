#include "modules/common/perf/perf_gflags.h"

DEFINE_bool(enable_latency_trace, true, "Enable per-frame latency trace.");

// This run id is only a fallback. In experiment runs, the local latency writer
// first reads APOLLO_LATENCY_RUN_ID or ${latency_output_dir}/active_run_id.
DEFINE_string(latency_run_id, "manual_run", "Run id.");

DEFINE_string(latency_output_dir, "/apollo_workspace/data/log/latency",
              "Output dir.");

DEFINE_bool(enable_resource_sampler, false, "Enable resource sampler.");
DEFINE_int32(resource_sample_hz, 100, "Resource sample frequency in Hz.");

DEFINE_bool(latency_record_enable_local_csv, true,
            "Write latency records directly to per-process local CSV files.");

DEFINE_bool(latency_record_enable_cyber_publish, false,
            "Also publish latency records to /apollo/common/latency_records. "
            "Keep false for strict experiment collection.");

DEFINE_int32(latency_record_local_flush_interval_ms, 500,
             "Flush interval in milliseconds for local latency CSV writer.");

DEFINE_int32(latency_record_local_status_interval_sec, 30,
             "Status log interval in seconds for local latency CSV writer.");

DEFINE_int64(latency_record_local_queue_warn_size, 200000,
             "Warn when local latency CSV queue exceeds this size. "
             "The queue is intentionally not dropped.");

DEFINE_bool(latency_trace_enable_event_filter, true,
            "Enable event filtering before creating LatencyRecorder.");

DEFINE_string(latency_trace_block_suffixes,
              ".tick,.input,.output_pub,.checkinput_enter,.detect.ok",
              "Comma-separated event-name suffixes to drop.");

DEFINE_string(latency_trace_block_substrings, "",
              "Comma-separated event-name substrings to drop.");
