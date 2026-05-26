/******************************************************************************
 * Copyright 2024 The Apollo Authors. All Rights Reserved.
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

#include "modules/perception/lidar_cpdet_detection/lidar_cpdetection_component.h"

#include "cyber/common/log.h"
#include "cyber/profiler/profiler.h"
#include "modules/common/perf/latency_trace.h"

namespace apollo {
namespace perception {
namespace lidar {

bool LidarCPDetectionComponent::Init() {
  LidarCPDetectionComponentConfig comp_config;
  if (!GetProtoConfig(&comp_config)) {
    AERROR << "Get LidarCPDetectionComponentConfig file failed";
    return false;
  }
  AINFO << "Lidar CPDetection Component Configs: " << comp_config.DebugString();

  sensor_name_ = comp_config.sensor_name();
  // writer
  output_channel_name_ = comp_config.output_channel_name();
  writer_ =
      node_->CreateWriter<onboard::LidarFrameMessage>(output_channel_name_);

  use_object_builder_ = comp_config.use_object_builder();

  // detector init
  auto plugin_param = comp_config.plugin_param();
  std::string detector_name = plugin_param.name();
  BaseCPDetector* detector =
      BaseCPDetectorRegisterer::GetInstanceByName(detector_name);
  CHECK_NOTNULL(detector);
  detector_.reset(detector);

  CPDetectorInitOptions detection_init_options;
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

bool LidarCPDetectionComponent::Proc(
    const std::shared_ptr<LidarFrameMessage>& message) {
  PERF_FUNCTION()

  // 提取 trace_id
  uint64_t trace_id = 0;
  if (message != nullptr && message->timestamp_ > 0.0) {
    trace_id = static_cast<uint64_t>(message->timestamp_ * 1e9);
  }

  // tick - 帧到达时间点
  const uint64_t t0 = apollo::perf::NowNs();
  apollo::perf::AppendPoint("perception.lidar_cpdetection.tick",
                            trace_id, t0);

  // internal proc - 内部处理
  const uint64_t i0 = apollo::perf::NowNs();
  bool status = InternalProc(message, trace_id);
  const uint64_t i1 = apollo::perf::NowNs();
  apollo::perf::AppendLatency("perception.lidar_cpdetection.internal_proc",
                              trace_id, i0, i1);

  if (status) {
    // Write - 发送消息
    const uint64_t w0 = apollo::perf::NowNs();
    writer_->Write(message);
    const uint64_t w1 = apollo::perf::NowNs();
    apollo::perf::AppendLatency("perception.lidar_cpdetection.write",
                                trace_id, w0, w1);
    apollo::perf::AppendLatency("perception.lidar_cpdetection.total",
                                trace_id, t0, w1);

    AINFO << "Send Lidar detection output message.";
  }
  return status;
}

bool LidarCPDetectionComponent::InternalProc(
    const std::shared_ptr<LidarFrameMessage>& in_message,
    const uint64_t trace_id) {
  // 1. Detect - 目标检测
  const uint64_t d0 = apollo::perf::NowNs();
  PERF_BLOCK("lidar_detector")
  CPDetectorOptions detection_options;
  if (!detector_->Detect(detection_options, in_message->lidar_frame_.get())) {
    AERROR << "Lidar detector detect error!";
    return false;
  }
  PERF_BLOCK_END
  const uint64_t d1 = apollo::perf::NowNs();
  apollo::perf::AppendLatency("perception.lidar_cpdetection.detect",
                              trace_id, d0, d1);

  // 2. ObjectBuilder - 目标构建
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
  apollo::perf::AppendLatency("perception.lidar_cpdetection.object_builder",
                              trace_id, b0, b1);

  return true;
}

}  // namespace lidar
}  // namespace perception
}  // namespace apollo
