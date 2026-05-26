/******************************************************************************
 * Copyright 2023 The Apollo Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *****************************************************************************/

#include "modules/perception/lidar_detection_filter/lidar_detection_filter_component.h"

#include "cyber/common/log.h"
#include "cyber/profiler/profiler.h"
#include "modules/common/perf/latency_trace.h"

namespace apollo {
namespace perception {
namespace lidar {

bool LidarDetectionFilterComponent::Init() {
  LidarDetectionFilterComponentConfig comp_config;
  if (!GetProtoConfig(&comp_config)) {
    AERROR << "Get LidarDetectionFilterComponentConfig file failed";
    return false;
  }
  AINFO << "Lidar Detection Filter Component Configs: "
        << comp_config.DebugString();

  output_channel_name_ = comp_config.output_channel_name();
  writer_ = node_->CreateWriter<LidarFrameMessage>(output_channel_name_);

  use_object_filter_bank_ = comp_config.use_object_filter_bank();
  if (use_object_filter_bank_) {
    auto plugin_param = comp_config.plugin_param();
    ObjectFilterInitOptions filter_bank_init_options;
    filter_bank_init_options.config_path = plugin_param.config_path();
    filter_bank_init_options.config_file = plugin_param.config_file();
    ACHECK(filter_bank_.Init(filter_bank_init_options));
  }

  return true;
}

// bool LidarDetectionFilterComponent::Proc(
//     const std::shared_ptr<LidarFrameMessage>& message) {
//   PERF_FUNCTION()
//   // internal proc
//   bool status = InternalProc(message);
//   if (status) {
//     writer_->Write(message);
//     AINFO << "Send lidar detection filter message.";
//   }
//   return status;
// }
bool LidarDetectionFilterComponent::Proc(
    const std::shared_ptr<LidarFrameMessage>& message) {
  PERF_FUNCTION();

  const uint64_t t0 = apollo::perf::NowNs();

  uint64_t trace_id = 0;
  if (message && message->lidar_frame_ != nullptr &&
      message->lidar_frame_->timestamp > 0.0) {
    trace_id = static_cast<uint64_t>(message->lidar_frame_->timestamp * 1e9);
  }

  apollo::perf::AppendPoint(
      "perception.lidar_detection_filter.input", trace_id, t0);

  AINFO << "[LIDAR_DETECTION_FILTER_INPUT] "
        << "trace_id=" << trace_id
        << " filter_proc_entry_ns=" << t0
        << " input_channel=/perception/lidar/detection";

  const uint64_t p0 = apollo::perf::NowNs();
  bool status = InternalProc(message);
  const uint64_t p1 = apollo::perf::NowNs();

  apollo::perf::AppendLatency(
      "perception.lidar_detection_filter.proc", trace_id, p0, p1);

  if (status) {
    const uint64_t w0 = apollo::perf::NowNs();
    writer_->Write(message);
    const uint64_t w1 = apollo::perf::NowNs();

    apollo::perf::AppendLatency(
        "perception.lidar_detection_filter.write", trace_id, w0, w1);

    AINFO << "[LIDAR_DETECTION_FILTER_OUTPUT] "
          << "trace_id=" << trace_id
          << " filter_output_begin_ns=" << w0
          << " filter_output_end_ns=" << w1
          << " output_channel=/perception/lidar/detection_filter"
          << " status=1";
  } else {
    AINFO << "[LIDAR_DETECTION_FILTER_OUTPUT] "
          << "trace_id=" << trace_id
          << " filter_output_begin_ns=0"
          << " filter_output_end_ns=0"
          << " output_channel=/perception/lidar/detection_filter"
          << " status=0";
  }

  const uint64_t t1 = apollo::perf::NowNs();
  apollo::perf::AppendLatency(
      "perception.lidar_detection_filter.total", trace_id, t0, t1);

  return status;
}


// bool LidarDetectionFilterComponent::InternalProc(
//     const std::shared_ptr<LidarFrameMessage>& in_message) {
//   ObjectFilterOptions filter_options;

//   PERF_BLOCK("filter_bank")
//   if (use_object_filter_bank_) {
//     if (!filter_bank_.Filter(filter_options, in_message->lidar_frame_.get())) {
//       AINFO << "Lidar detection filter banck error.";
//       return false;
//     }
//   }
//   PERF_BLOCK_END

//   return true;
// }
bool LidarDetectionFilterComponent::InternalProc(
    const std::shared_ptr<LidarFrameMessage>& in_message) {
  uint64_t trace_id = 0;
  if (in_message != nullptr) {
    if (in_message->lidar_timestamp_ != 0) {
      trace_id = in_message->lidar_timestamp_;
    } else if (in_message->lidar_frame_ != nullptr &&
               in_message->lidar_frame_->timestamp > 0.0) {
      trace_id = static_cast<uint64_t>(in_message->lidar_frame_->timestamp * 1e9);
    } else if (in_message->timestamp_ > 0.0) {
      trace_id = static_cast<uint64_t>(in_message->timestamp_ * 1e9);
    }
  }

  ObjectFilterOptions filter_options;

  const uint64_t f0 = apollo::perf::NowNs();
  PERF_BLOCK("filter_bank")
  if (use_object_filter_bank_) {
    if (!filter_bank_.Filter(filter_options, in_message->lidar_frame_.get())) {
      AINFO << "Lidar detection filter bank error.";
      return false;
    }
  }
  PERF_BLOCK_END
  const uint64_t f1 = apollo::perf::NowNs();

  apollo::perf::AppendLatency("perception.lidar_detection_filter.filter_bank",
                              trace_id, f0, f1);
  return true;
}


}  // namespace lidar
}  // namespace perception
}  // namespace apollo
