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

#include "modules/perception/pointcloud_preprocess/pointcloud_preprocess_component.h"

#include "cyber/profiler/profiler.h"
#include "cyber/time/clock.h"
#include "modules/perception/common/algorithm/sensor_manager/sensor_manager.h"
#include "modules/perception/common/lidar/common/lidar_frame_pool.h"
#include "modules/common/perf/latency_trace.h"

namespace apollo {
namespace perception {
namespace lidar {

using apollo::cyber::Clock;
using apollo::cyber::common::GetAbsolutePath;

std::atomic<uint32_t> PointCloudPreprocessComponent::seq_num_{0};

bool PointCloudPreprocessComponent::Init() {
  PointCloudPreprocessComponentConfig comp_config;
  if (!GetProtoConfig(&comp_config)) {
    AERROR << "Get PointCloudPreprocessComponentConfig file failed";
    return false;
  }
  AINFO << "PointCloud Preprocess Component Configs: "
        << comp_config.DebugString();

  output_channel_name_ = comp_config.output_channel_name();
  sensor_name_ = comp_config.sensor_name();
  // Use sensor_name as lidar to novatel child_frame_id. If not equal
  // we can use `sensor_info_` to get child_frame_id
  lidar2novatel_tf2_child_frame_id_ = comp_config.sensor_name();
  lidar_query_tf_offset_ =
      static_cast<float>(comp_config.lidar_query_tf_offset());

  auto plugin_param = comp_config.plugin_param();
  std::string cloud_preprocessor_name = plugin_param.name();
  config_path_ = plugin_param.config_path();
  config_file_ = plugin_param.config_file();
  cloud_preprocessor_ = BasePointCloudPreprocessorRegisterer::GetInstanceByName(
      cloud_preprocessor_name);
  CHECK_NOTNULL(cloud_preprocessor_);
  // writer
  writer_ =
      node_->CreateWriter<onboard::LidarFrameMessage>(output_channel_name_);

  if (!InitAlgorithmPlugin()) {
    AERROR << "Failed to init pointcloud preprocess component plugin.";
    return false;
  }

  return true;
}

bool PointCloudPreprocessComponent::Proc(
    const std::shared_ptr<drivers::PointCloud>& message) {
  PERF_FUNCTION();

  const uint64_t t0 = apollo::perf::NowNs();
  const uint64_t trace_id =
      static_cast<uint64_t>(message->measurement_time() * 1e9);

  apollo::perf::AppendPoint("perception.pointcloud_preprocess.tick",
                            trace_id, t0);

  auto out_message = std::make_shared<onboard::LidarFrameMessage>();

  const uint64_t t1 = apollo::perf::NowNs();
  bool status = InternalProc(message, out_message);
  const uint64_t t2 = apollo::perf::NowNs();

  apollo::perf::AppendLatency("perception.pointcloud_preprocess.proc",
                              trace_id, t1, t2);

  if (status) {
    const uint64_t t3 = apollo::perf::NowNs();
    writer_->Write(out_message);
    const uint64_t t4 = apollo::perf::NowNs();

    apollo::perf::AppendLatency("perception.pointcloud_preprocess.write",
                                trace_id, t3, t4);
    apollo::perf::AppendLatency("perception.pointcloud_preprocess.total",
                                trace_id, t0, t4);
  }

  return status;
}

bool PointCloudPreprocessComponent::InitAlgorithmPlugin() {
  ACHECK(algorithm::SensorManager::Instance()->GetSensorInfo(sensor_name_,
                                                             &sensor_info_));
  // pointcloud preprocessor init
  PointCloudPreprocessorInitOptions preprocessor_init_options;
  preprocessor_init_options.sensor_name = sensor_name_;
  preprocessor_init_options.config_path = config_path_;
  preprocessor_init_options.config_file = config_file_;
  ACHECK(cloud_preprocessor_->Init(preprocessor_init_options));

  // static transform
  lidar2world_trans_.Init(lidar2novatel_tf2_child_frame_id_);
  return true;
}

bool PointCloudPreprocessComponent::InternalProc(
    const std::shared_ptr<const drivers::PointCloud>& in_message,
    const std::shared_ptr<onboard::LidarFrameMessage>& out_message) {
  uint64_t trace_id = 0;
  if (in_message != nullptr && in_message->measurement_time() > 0.0)
  {
    trace_id = static_cast<uint64_t>(in_message->measurement_time() * 1e9);
  }
  const uint64_t c0 = apollo::perf::NowNs();

  uint32_t seq_num = seq_num_.fetch_add(1);
  const double timestamp = in_message->measurement_time();
  const double cur_time = Clock::NowInSeconds();
  const double start_latency = (cur_time - timestamp) * 1e3;
  AINFO << std::setprecision(16) << "FRAME:Preprocess:Start:msg_time["
        << timestamp << "]:sensor[" << sensor_name_ << "]:cur_time[" << cur_time
        << "]:cur_latency[" << start_latency << "]";

  out_message->timestamp_ = timestamp;
  out_message->lidar_timestamp_ = in_message->header().lidar_timestamp();
  out_message->seq_num_ = seq_num;
  out_message->process_stage_ = onboard::ProcessStage::LIDAR_DETECTION;
  out_message->error_code_ = apollo::common::ErrorCode::OK;

  auto& frame = out_message->lidar_frame_;
  frame = lidar::LidarFramePool::Instance().Get();
  frame->cloud = base::PointFCloudPool::Instance().Get();
  frame->timestamp = timestamp;
  frame->sensor_info = sensor_info_;
  const uint64_t c1 = apollo::perf::NowNs();
  apollo::perf::AppendLatency("perception.pointcloud_preprocess.copyinfo",
                              trace_id, c0, c1);

  const uint64_t tf0 = apollo::perf::NowNs();                            
  Eigen::Affine3d pose = Eigen::Affine3d::Identity();
  Eigen::Affine3d pose_novatel = Eigen::Affine3d::Identity();
  const double lidar_query_tf_timestamp =
      timestamp - lidar_query_tf_offset_ * 0.001;
  if (!lidar2world_trans_.GetSensor2worldTrans(lidar_query_tf_timestamp, &pose,
                                               &pose_novatel)) {
    out_message->error_code_ = apollo::common::ErrorCode::PERCEPTION_ERROR_TF;
    AERROR << "Failed to get pose at time: " << lidar_query_tf_timestamp;
    return false;
  }

  frame->lidar2world_pose = pose;
  frame->novatel2world_pose = pose_novatel;
  const uint64_t tf1 = apollo::perf::NowNs();
  apollo::perf::AppendLatency("perception.pointcloud_preprocess.tf",
                              trace_id, tf0, tf1);
  // pointcloud preprocessor
  const uint64_t p0 = apollo::perf::NowNs();
  PointCloudPreprocessorOptions preprocessor_option;
  if (!lidar2world_trans_.GetExtrinsics(
      &preprocessor_option.sensor2novatel_extrinsics)) {
    AERROR << "Get sensor2novatel extrinsics error.";
    return false;
  }
  frame->lidar2novatel_extrinsics =
      preprocessor_option.sensor2novatel_extrinsics;

  PERF_BLOCK("cloud_preprocessor")
  if (!cloud_preprocessor_->Preprocess(
      preprocessor_option, in_message, frame.get())) {
    AERROR << "Pointcloud preprocess error.";
    return false;
  }
  PERF_BLOCK_END
  const uint64_t p1 = apollo::perf::NowNs();
  apollo::perf::AppendLatency("perception.pointcloud_preprocess.preprocess",
                              trace_id, p0, p1);

  return true;
}

}  // namespace lidar
}  // namespace perception
}  // namespace apollo
