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

#include "modules/perception/pointcloud_map_based_roi/pointcloud_map_based_roi_component.h"

#include "cyber/profiler/profiler.h"
#include "modules/perception/common/lidar/common/config_util.h"
#include "modules/common/perf/latency_trace.h"

namespace apollo {
namespace perception {
namespace lidar {

bool PointCloudMapROIComponent::Init() {
  PointCloudMapROIComponentConfig comp_config;
  if (!GetProtoConfig(&comp_config)) {
    AERROR << "Get PointCloudMapROIComponentConfig file failed";
    return false;
  }
  AINFO << "PointCloud map based roi Component Configs: "
        << comp_config.DebugString();
  // writer
  output_channel_name_ = comp_config.output_channel_name();
  writer_ = node_->CreateWriter<LidarFrameMessage>(output_channel_name_);

  // Scene manager
  ACHECK(SceneManager::Instance().Init());

  // map manager init
  use_map_manager_ =
      comp_config.use_map_manager() && comp_config.enable_hdmap();
  if (use_map_manager_) {
    MapManagerInitOptions map_manager_init_options;
    map_manager_init_options.config_path =
      comp_config.map_manager_config_path();
    map_manager_init_options.config_file =
      comp_config.map_manager_config_file();
    if (!map_manager_.Init(map_manager_init_options)) {
      AINFO << "Failed to init map manager.";
      use_map_manager_ = false;
    }
  }

  // init roi filter
  auto& plugin = comp_config.plugin_param();
  roi_filter_ = apollo::cyber::plugin_manager::PluginManager::Instance()
                    ->CreateInstance<BaseROIFilter>(
                        ConfigUtil::GetFullClassName(plugin.name()));

  CHECK_NOTNULL(roi_filter_);
  ROIFilterInitOptions roi_filter_init_options;
  roi_filter_init_options.config_path = plugin.config_path();
  roi_filter_init_options.config_file = plugin.config_file();
  ACHECK(roi_filter_->Init(roi_filter_init_options))
      << "Failed to init roi filter.";

  return true;
}

// bool PointCloudMapROIComponent::Proc(
//     const std::shared_ptr<LidarFrameMessage>& message) {
//   PERF_FUNCTION()
//   // internal proc
//   bool status = InternalProc(message);
//   if (status) {
//     writer_->Write(message);
//     AINFO << "Send pointcloud map based roi output message.";
//   }
//   return status;
// }
bool PointCloudMapROIComponent::Proc(
    const std::shared_ptr<LidarFrameMessage>& message) {
  PERF_FUNCTION();

  uint64_t trace_id = 0;
  if (message != nullptr) {
    if (message->lidar_timestamp_ != 0) {
      trace_id = message->lidar_timestamp_;
    } else if (message->timestamp_ > 0.0) {
      trace_id = static_cast<uint64_t>(message->timestamp_ * 1e9);
    }
  }

  const uint64_t t0 = apollo::perf::NowNs();
  apollo::perf::AppendPoint("perception.pointcloud_map_based_roi.tick",
                            trace_id, t0);

  const uint64_t t1 = apollo::perf::NowNs();
  bool status = InternalProc(message);
  const uint64_t t2 = apollo::perf::NowNs();
  apollo::perf::AppendLatency("perception.pointcloud_map_based_roi.proc",
                              trace_id, t1, t2);

  if (status) {
    const uint64_t t3 = apollo::perf::NowNs();
    writer_->Write(message);
    const uint64_t t4 = apollo::perf::NowNs();
    apollo::perf::AppendLatency("perception.pointcloud_map_based_roi.write",
                                trace_id, t3, t4);
    
  }
  const uint64_t tt0 = apollo::perf::NowNs();
  apollo::perf::AppendLatency("perception.pointcloud_map_based_roi.total",
                                trace_id, t0, tt0);
  return status;
}

// bool PointCloudMapROIComponent::InternalProc(
//     const std::shared_ptr<LidarFrameMessage>& message) {
//   // map update
//   PERF_BLOCK("map_manager")
//   if (use_map_manager_) {
//     MapManagerOptions map_manager_options;
//     if (!map_manager_.Update(map_manager_options,
//                              message->lidar_frame_.get())) {
//       AINFO << "Failed to update map structure.";
//       return false;
//     }
//   }
//   PERF_BLOCK_END

//   ROIFilterOptions roi_filter_options;
//   auto lidar_frame_ref = message->lidar_frame_.get();
//   auto original_cloud = lidar_frame_ref->cloud;
//   PERF_BLOCK("roi_filter")
//   if (lidar_frame_ref->hdmap_struct != nullptr &&
//       roi_filter_->Filter(roi_filter_options, lidar_frame_ref)) {
//     // do nothing
//   } else {
//     AINFO << "Fail to call roi filter, use origin cloud.";
//     lidar_frame_ref->roi_indices.indices.resize(original_cloud->size());
//     // we manually fill roi indices with all cloud point indices
//     std::iota(lidar_frame_ref->roi_indices.indices.begin(),
//               lidar_frame_ref->roi_indices.indices.end(), 0);
//   }
//   PERF_BLOCK_END

//   return true;
// }
bool PointCloudMapROIComponent::InternalProc(
    const std::shared_ptr<LidarFrameMessage>& message) {
  uint64_t trace_id = 0;
  if (message != nullptr) {
    if (message->lidar_timestamp_ != 0) {
      trace_id = message->lidar_timestamp_;
    } else if (message->timestamp_ > 0.0) {
      trace_id = static_cast<uint64_t>(message->timestamp_ * 1e9);
    }
  }

  const uint64_t m0 = apollo::perf::NowNs();
  PERF_BLOCK("map_manager")
  if (use_map_manager_) {
    MapManagerOptions map_manager_options;
    if (!map_manager_.Update(map_manager_options, message->lidar_frame_.get())) {
      AINFO << "Failed to update map structure.";
      return false;
    }
  }
  PERF_BLOCK_END
  const uint64_t m1 = apollo::perf::NowNs();
  apollo::perf::AppendLatency("perception.pointcloud_map_based_roi.map_manager",
                              trace_id, m0, m1);

  ROIFilterOptions roi_filter_options;
  auto lidar_frame_ref = message->lidar_frame_.get();
  auto original_cloud = lidar_frame_ref->cloud;

  const uint64_t r0 = apollo::perf::NowNs();
  PERF_BLOCK("roi_filter")
  if (lidar_frame_ref->hdmap_struct != nullptr &&
      roi_filter_->Filter(roi_filter_options, lidar_frame_ref)) {
  } else {
    AINFO << "Fail to call roi filter, use origin cloud.";
    lidar_frame_ref->roi_indices.indices.resize(original_cloud->size());
    std::iota(lidar_frame_ref->roi_indices.indices.begin(),
              lidar_frame_ref->roi_indices.indices.end(), 0);
  }
  PERF_BLOCK_END
  const uint64_t r1 = apollo::perf::NowNs();
  apollo::perf::AppendLatency("perception.pointcloud_map_based_roi.roi_filter",
                              trace_id, r0, r1);
  return true;
}


}  // namespace lidar
}  // namespace perception
}  // namespace apollo
