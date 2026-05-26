#pragma once

#include <cstdint>
#include <string>

#include "modules/common_msgs/basic_msgs/header.pb.h"

namespace apollo {
namespace perf {

uint64_t NowNs();

uint64_t TraceIdFromHeader(const apollo::common::Header& header);

// Record an interval event.
void AppendLatency(const std::string& event_name,
                   uint64_t trace_id,
                   uint64_t begin_ns,
                   uint64_t end_ns);

// Record a point event. Internally it is stored as [t_ns, t_ns + 1].
void AppendPoint(const std::string& event_name,
                 uint64_t trace_id,
                 uint64_t t_ns);

void FlushAllRecorders();

}  // namespace perf
}  // namespace apollo
