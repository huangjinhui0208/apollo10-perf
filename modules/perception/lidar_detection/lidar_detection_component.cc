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

#include "modules/perception/lidar_detection/lidar_detection_component.h"
#include "modules/common/perf/latency_trace.h"
#include "cyber/common/log.h"
#include "cyber/profiler/profiler.h"

namespace apollo {
namespace perception {
namespace lidar {

bool LidarDetectionComponent::Init() {
  LidarDetectionComponentConfig comp_config;
  if (!GetProtoConfig(&comp_config)) {
    AERROR << "Get LidarDetectionComponentConfig file failed";
    return false;
  }
  AINFO << "Lidar Detection Component Configs: " << comp_config.DebugString();

  sensor_name_ = comp_config.sensor_name();
  // writer
  output_channel_name_ = comp_config.output_channel_name();
  writer_ =
      node_->CreateWriter<onboard::LidarFrameMessage>(output_channel_name_);

  use_object_builder_ = comp_config.use_object_builder();

  // detector init
  auto plugin_param = comp_config.plugin_param();
  std::string detector_name = plugin_param.name();
  BaseLidarDetector* detector =
      BaseLidarDetectorRegisterer::GetInstanceByName(detector_name);
  CHECK_NOTNULL(detector);
  detector_.reset(detector);

  LidarDetectorInitOptions detection_init_options;
  detection_init_options.sensor_name = sensor_name_;
  detection_init_options.config_path = plugin_param.config_path();
  detection_init_options.config_file = plugin_param.config_file();
  ACHECK(detector_->Init(detection_init_options))
      << "lidar detector init error";

  // TODO(zero): Fix paddle doesn't output log to file
  FLAGS_logtostderr = 0;

  // object builder init
  if (use_object_builder_) {
    ObjectBuilderInitOptions builder_init_options;
    ACHECK(builder_.Init(builder_init_options));
  }

  AINFO << "Successfully init lidar detection component.";
  return true;
}

bool LidarDetectionComponent::Proc(
    const std::shared_ptr<LidarFrameMessage>& message) {
  PERF_FUNCTION();

  const uint64_t t0 = apollo::perf::NowNs();
  uint64_t trace_id = 0;
  if (message && message->lidar_frame_ != nullptr &&
      message->lidar_frame_->timestamp > 0.0) {
    trace_id = static_cast<uint64_t>(message->lidar_frame_->timestamp * 1e9);
  }
  // 计算点云数量，记录输入性能数据
  size_t input_num_points = 0;
  if (message && message->lidar_frame_ != nullptr &&
      message->lidar_frame_->cloud != nullptr) {
    input_num_points = message->lidar_frame_->cloud->size();
  }

AINFO << "[LIDAR_DETECTION_INPUT] "
      << "trace_id=" << trace_id
      << " frame_stamp_ns=" << trace_id
      << " lidar_proc_entry_ns=" << t0
      << " input_num_points=" << input_num_points
      << " input_channel=/perception/lidar/pointcloud_ground_detection";

  apollo::perf::AppendPoint("perception.lidar_detection.input", trace_id, t0);

  const uint64_t t1 = apollo::perf::NowNs();
  bool status = InternalProc(message);
  const uint64_t t2 = apollo::perf::NowNs();

  apollo::perf::AppendLatency("perception.lidar_detection.proc",
                              trace_id, t1, t2);

  // if (status) {
  //   const uint64_t t3 = apollo::perf::NowNs();
  //   writer_->Write(message);
  //   const uint64_t t4 = apollo::perf::NowNs();

  //   apollo::perf::AppendLatency("perception.lidar_detection.write",
  //                               trace_id, t3, t4);
   
  // }
  if (status) {
  const uint64_t t3 = apollo::perf::NowNs();
  writer_->Write(message);
  const uint64_t t4 = apollo::perf::NowNs();

  apollo::perf::AppendLatency("perception.lidar_detection.write",
                              trace_id, t3, t4);

  AINFO << "[LIDAR_DETECTION_OUTPUT] "
        << "trace_id=" << trace_id
        << " output_write_begin_ns=" << t3
        << " output_write_end_ns=" << t4
        << " output_channel=/perception/lidar/detection"
        << " status=1";
  } else {
    AINFO << "[LIDAR_DETECTION_OUTPUT] "
          << "trace_id=" << trace_id
          << " output_write_begin_ns=0"
          << " output_write_end_ns=0"
          << " output_channel=/perception/lidar/detection"
          << " status=0";
    apollo::perf::AppendPoint("perception.lidar_detection.no_output",
                              trace_id, t2);
  }
  const uint64_t tt0 = apollo::perf::NowNs();
  apollo::perf::AppendLatency("perception.lidar_detection.total",
                                trace_id, t0, tt0);
  return status;
}

bool LidarDetectionComponent::InternalProc(
    const std::shared_ptr<LidarFrameMessage>& in_message) {

  uint64_t trace_id = 0;
  if (in_message && in_message->lidar_frame_ != nullptr &&
      in_message->lidar_frame_->timestamp > 0.0) {
    trace_id = static_cast<uint64_t>(in_message->lidar_frame_->timestamp * 1e9);
  }

  const uint64_t d0 = apollo::perf::NowNs();
  PERF_BLOCK("lidar_detector")
  apollo::perf::AppendPoint("perception.lidar_detection.detect.enter", trace_id,
                            d0);
  const uint64_t d_prepare0 = apollo::perf::NowNs();
  LidarDetectorOptions detection_options;
  const uint64_t d_prepare1 = apollo::perf::NowNs();
  apollo::perf::AppendLatency("perception.lidar_detection.detect.prepare",
                              trace_id, d_prepare0, d_prepare1);

  const uint64_t d_detect0 = apollo::perf::NowNs();
  if (!detector_->Detect(detection_options, in_message->lidar_frame_.get())) {
    const uint64_t d_detect1 = apollo::perf::NowNs();
    apollo::perf::AppendLatency("perception.lidar_detection.detect.call",
                                trace_id, d_detect0, d_detect1);
    apollo::perf::AppendPoint("perception.lidar_detection.detect.fail", trace_id,
                              d_detect1);
    AERROR << "Lidar detector detect error!";
    return false;
  }
  const uint64_t d_detect1 = apollo::perf::NowNs();
  apollo::perf::AppendLatency("perception.lidar_detection.detect.call", trace_id,
                              d_detect0, d_detect1);
  apollo::perf::AppendPoint("perception.lidar_detection.detect.ok", trace_id,
                            d_detect1);
  PERF_BLOCK_END
  const uint64_t d1 = apollo::perf::NowNs();
  apollo::perf::AppendLatency("perception.lidar_detection.detector",
                              trace_id, d0, d1);

  const uint64_t b0 = apollo::perf::NowNs();
  PERF_BLOCK("object_builder")
  if (use_object_builder_) {
    ObjectBuilderOptions builder_options;
    if (!builder_.Build(builder_options, in_message->lidar_frame_.get())) {
      AERROR << "Lidar detector, object builder error.";
      return false;
    }
  }
  PERF_BLOCK_END
  const uint64_t b1 = apollo::perf::NowNs();
  apollo::perf::AppendLatency("perception.lidar_detection.object_builder",
                              trace_id, b0, b1);

  return true;
}

}  // namespace lidar
}  // namespace perception
}  // namespace apollo
