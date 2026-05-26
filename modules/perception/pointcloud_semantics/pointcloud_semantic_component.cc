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

#include "modules/perception/pointcloud_semantics/pointcloud_semantic_component.h"

#include "cyber/profiler/profiler.h"
#include "modules/common/perf/latency_trace.h"

namespace apollo {
namespace perception {
namespace lidar {

bool PointCloudSemanticComponent::Init() {
    PointCloudSemanticComponentConfig comp_config;
    if (!GetProtoConfig(&comp_config)) {
        AERROR << "Get PointCloudSemanticComponentConfig file failed";
        return false;
    }
    AINFO << "PointCloud semantic Component Configs: " << comp_config.DebugString();

    // writer
    output_channel_name_ = comp_config.output_channel_name();
    writer_ = node_->CreateWriter<LidarFrameMessage>(output_channel_name_);

    // init parser
    auto plugin_param = comp_config.plugin_param();
    std::string parser_name = plugin_param.name();
    BasePointCloudParser* parser = BasePointCloudParserRegisterer::GetInstanceByName(parser_name);
    CHECK_NOTNULL(parser);
    parser_.reset(parser);
    PointCloudParserInitOptions init_options;
    init_options.config_path = plugin_param.config_path();
    init_options.config_file = plugin_param.config_file();
    ACHECK(parser_->Init(init_options)) << "PointCloud parser init error";

    AINFO << "Successfully init pointcloud parser component.";
    return true;
}

bool PointCloudSemanticComponent::Proc(const std::shared_ptr<LidarFrameMessage>& message) {
  PERF_FUNCTION()

  // 提取 trace_id
  uint64_t trace_id = 0;
  if (message != nullptr && message->timestamp_ > 0.0) {
    trace_id = static_cast<uint64_t>(message->timestamp_ * 1e9);
  }

  // tick - 帧到达时间点
  const uint64_t t0 = apollo::perf::NowNs();
  apollo::perf::AppendPoint("perception.pointcloud_semantic.tick",
                            trace_id, t0);

  // internal proc - 内部处理
  const uint64_t i0 = apollo::perf::NowNs();
  bool status = InternalProc(message, trace_id);
  const uint64_t i1 = apollo::perf::NowNs();
  apollo::perf::AppendLatency("perception.pointcloud_semantic.internal_proc",
                              trace_id, i0, i1);

  if (status) {
    // Write - 发送消息
    const uint64_t w0 = apollo::perf::NowNs();
    writer_->Write(message);
    const uint64_t w1 = apollo::perf::NowNs();
    apollo::perf::AppendLatency("perception.pointcloud_semantic.write",
                                trace_id, w0, w1);
    apollo::perf::AppendLatency("perception.pointcloud_semantic.total",
                                trace_id, t0, w1);

    AINFO << "Send PointCloud Semantic message.";
  } else {
    AERROR << "PointCloudSemanticComponent InternalProc error.";
  }
  return status;
}

bool PointCloudSemanticComponent::InternalProc(const std::shared_ptr<LidarFrameMessage>& in_message,
                                                const uint64_t trace_id) {
  // 1. Parse - 语义解析
  const uint64_t p0 = apollo::perf::NowNs();
  PERF_BLOCK("pointcloud_parser")
  PointCloudParserOptions parser_options;
  if (!parser_->Parse(parser_options, in_message->lidar_frame_.get())) {
    AERROR << "PointCloud parser error!";
    return false;
  }
  PERF_BLOCK_END
  const uint64_t p1 = apollo::perf::NowNs();
  apollo::perf::AppendLatency("perception.pointcloud_semantic.parse",
                              trace_id, p0, p1);

  return true;
}

}  // namespace lidar
}  // namespace perception
}  // namespace apollo
