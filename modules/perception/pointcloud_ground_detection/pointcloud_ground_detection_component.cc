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

#include "modules/perception/pointcloud_ground_detection/pointcloud_ground_detection_component.h"

#include "cyber/profiler/profiler.h"
#include "modules/common/perf/latency_trace.h"

namespace apollo {
namespace perception {
namespace lidar {

using apollo::cyber::common::GetAbsolutePath;

bool PointCloudGroundDetectComponent::Init() {
  PointCloudGroundDetectComponentConfig comp_config;
  if (!GetProtoConfig(&comp_config)) {
    AERROR << "Get PointCloudGroundDetectComponentConfig file failed";
    return false;
  }
  AINFO << "PointCloud Ground Detect Component Configs: "
        << comp_config.DebugString();
  output_channel_name_ = comp_config.output_channel_name();
  writer_ =
      node_->CreateWriter<onboard::LidarFrameMessage>(output_channel_name_);

  // groun detector
  auto plugin_param = comp_config.plugin_param();
  std::string ground_detector_name = plugin_param.name();
  ground_detector_ =
      BaseGroundDetectorRegisterer::GetInstanceByName(ground_detector_name);
  CHECK_NOTNULL(ground_detector_);

  GroundDetectorInitOptions ground_detector_init_options;
  ground_detector_init_options.config_path = plugin_param.config_path();
  ground_detector_init_options.config_file = plugin_param.config_file();
  ACHECK(ground_detector_->Init(ground_detector_init_options))
      << "Failed to init ground detection.";
  return true;
}

// bool PointCloudGroundDetectComponent::Proc(
//     const std::shared_ptr<LidarFrameMessage>& message) {
//   PERF_FUNCTION()
//   // internal proc
//   bool status = InternalProc(message);
//   if (status) {
//     writer_->Write(message);
//     AINFO << "Send pointcloud ground detect output message.";
//   }
//   return true;
// }
bool PointCloudGroundDetectComponent::Proc(
    const std::shared_ptr<LidarFrameMessage>& message) {
  PERF_FUNCTION();

  // uint64_t trace_id = 0;
  // if (message != nullptr) {
  //   if (message->lidar_timestamp_ != 0) {
  //     trace_id = message->lidar_timestamp_;
  //   } else if (message->timestamp_ > 0.0) {
  //     trace_id = static_cast<uint64_t>(message->timestamp_ * 1e9);
  //   }
  // }
  uint64_t trace_id = 0;
  size_t upstream_num_points = 0;

  if (message && message->lidar_frame_ != nullptr &&
      message->lidar_frame_->timestamp > 0.0) {
    trace_id = static_cast<uint64_t>(message->lidar_frame_->timestamp * 1e9);
  }

  if (message && message->lidar_frame_ != nullptr &&
      message->lidar_frame_->cloud != nullptr) {
    upstream_num_points = message->lidar_frame_->cloud->size();
  }

  const uint64_t t0 = apollo::perf::NowNs();
  apollo::perf::AppendPoint("perception.pointcloud_ground_detection.tick",
                            trace_id, t0);

  const uint64_t t1 = apollo::perf::NowNs();
  bool status = InternalProc(message);
  const uint64_t t2 = apollo::perf::NowNs();
  apollo::perf::AppendLatency("perception.pointcloud_ground_detection.proc",
                              trace_id, t1, t2);

  if (status) {
  const uint64_t w0 = apollo::perf::NowNs();
  writer_->Write(message);
  const uint64_t w1 = apollo::perf::NowNs();

  apollo::perf::AppendLatency(
      "perception.pointcloud_ground_detection.write", trace_id, w0, w1);

  AINFO << "[LIDAR_DETECTION_UPSTREAM_OUTPUT] "
        << "trace_id=" << trace_id
        << " frame_stamp_ns=" << trace_id
        << " upstream_output_begin_ns=" << w0
        << " upstream_output_end_ns=" << w1
        << " upstream_num_points=" << upstream_num_points
        << " output_channel=/perception/lidar/pointcloud_ground_detection";
  }
  
  const uint64_t tt0 = apollo::perf::NowNs();
  apollo::perf::AppendLatency("perception.pointcloud_ground_detection.total",
                                trace_id, t0, tt0);
  return status;
}


// bool PointCloudGroundDetectComponent::InternalProc(
//     const std::shared_ptr<LidarFrameMessage>& message) {
//   auto lidar_frame_ref = message->lidar_frame_;
//   PERF_BLOCK("ground_detector")
//   GroundDetectorOptions ground_detector_options;
//   if (!ground_detector_->Detect(
//       ground_detector_options, lidar_frame_ref.get())) {
//     AERROR << "Ground detect error.";
//     return false;
//   }
//   PERF_BLOCK_END

//   return true;
// }
bool PointCloudGroundDetectComponent::InternalProc(
    const std::shared_ptr<LidarFrameMessage>& message) {
  uint64_t trace_id = 0;
  if (message != nullptr) {
    if (message->lidar_timestamp_ != 0) {
      trace_id = message->lidar_timestamp_;
    } else if (message->timestamp_ > 0.0) {
      trace_id = static_cast<uint64_t>(message->timestamp_ * 1e9);
    }
  }

  GroundDetectorOptions options;
  const uint64_t g0 = apollo::perf::NowNs();
  PERF_BLOCK("ground_detector")
  if (!ground_detector_->Detect(options, message->lidar_frame_.get())) {
    AERROR << "Ground detector detect error!";
    return false;
  }
  PERF_BLOCK_END
  const uint64_t g1 = apollo::perf::NowNs();
  apollo::perf::AppendLatency("perception.pointcloud_ground_detection.ground_detector",
                              trace_id, g0, g1);
  return true;
}


}  // namespace lidar
}  // namespace perception
}  // namespace apollo
