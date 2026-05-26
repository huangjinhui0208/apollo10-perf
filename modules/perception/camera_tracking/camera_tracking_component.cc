/******************************************************************************
 * Copyright 2023 The Apollo Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *****************************************************************************/

#include "modules/perception/camera_tracking/camera_tracking_component.h"

#include "modules/perception/camera_tracking/proto/camera_tracking_component.pb.h"

#include "cyber/common/file.h"
#include "cyber/profiler/profiler.h"
#include "cyber/time/clock.h"
#include "modules/common/perf/latency_trace.h"

namespace apollo {
namespace perception {
namespace camera {

using Clock = apollo::cyber::Clock;

bool CameraTrackingComponent::Init() {
  CameraTrackingComponentConfig camera_tracking_component_config;
  if (!GetProtoConfig(&camera_tracking_component_config)) {
    return false;
  }
  AINFO << "Camera Tracking Component config: "
        << camera_tracking_component_config.DebugString();
  output_channel_name_ = camera_tracking_component_config.output_channel_name();
  writer_ =
      node_->CreateWriter<onboard::SensorFrameMessage>(output_channel_name_);

  sensor_manager_ = algorithm::SensorManager::Instance();

  PluginParam plugin_param = camera_tracking_component_config.plugin_param();
  std::string tracker_name = plugin_param.name();
  camera_obstacle_tracker_.reset(
      BaseObstacleTrackerRegisterer::GetInstanceByName(tracker_name));
  if (nullptr == camera_obstacle_tracker_) {
    AERROR << "Failed to get camera obstacle tracker instance.";
    return false;
  }

  ObstacleTrackerInitOptions tracker_init_options;
  tracker_init_options.config_path = plugin_param.config_path();
  tracker_init_options.config_file = plugin_param.config_file();
  tracker_init_options.image_width =
      camera_tracking_component_config.image_width();
  tracker_init_options.image_height =
      camera_tracking_component_config.image_height();
  tracker_init_options.gpu_id = camera_tracking_component_config.gpu_id();
  if (!camera_obstacle_tracker_->Init(tracker_init_options)) {
    AERROR << "Failed to init camera obstacle tracker.";
    return false;
  }
  return true;
}

// bool CameraTrackingComponent::Proc(
//     const std::shared_ptr<onboard::CameraFrame>& message) {
//   PERF_FUNCTION()
//   AINFO << std::setprecision(16)
//         << "Enter Tracking component, message timestamp: " << message->timestamp
//         << " current timestamp: " << Clock::NowInSeconds();

//   auto out_message = std::make_shared<onboard::SensorFrameMessage>();
//   bool status = InternalProc(message, out_message);

//   if (status) {
//     writer_->Write(out_message);
//     AINFO << "Send camera tracking output message.";
//   }
//   return status;
// }

bool CameraTrackingComponent::Proc(
    const std::shared_ptr<onboard::CameraFrame>& message) {
  PERF_FUNCTION()

  uint64_t trace_id = 0;
  if (message != nullptr && message->timestamp > 0.0) {
    trace_id = static_cast<uint64_t>(message->timestamp * 1e9);
  }

  const uint64_t t0 = apollo::perf::NowNs();
  apollo::perf::AppendPoint("perception.camera_tracking.tick",
                            trace_id, t0);

  AINFO << std::setprecision(16)
        << "Enter Tracking component, message timestamp: " << message->timestamp
        << " current timestamp: " << Clock::NowInSeconds();

  auto out_message = std::make_shared<onboard::SensorFrameMessage>();

  const uint64_t t1 = apollo::perf::NowNs();
  bool status = InternalProc(message, out_message);
  const uint64_t t2 = apollo::perf::NowNs();
  apollo::perf::AppendLatency("perception.camera_tracking.proc",
                              trace_id, t1, t2);

  if (status) {
    const uint64_t w0 = apollo::perf::NowNs();
    writer_->Write(out_message);
    const uint64_t w1 = apollo::perf::NowNs();
    apollo::perf::AppendLatency("perception.camera_tracking.write",
                                trace_id, w0, w1);
    AINFO << "Send camera tracking output message.";
  }
  const uint64_t tt0 = apollo::perf::NowNs();
  apollo::perf::AppendLatency("perception.camera_tracking.total",
                                trace_id, t0, tt0);
  return status;
}

// bool LidarDetectionFilterComponent::InternalProc(
//     const std::shared_ptr<LidarFrameMessage>& in_message) {
//   uint64_t trace_id = 0;
//   if (in_message != nullptr) {
//     if (in_message->lidar_timestamp_ != 0) {
//       trace_id = in_message->lidar_timestamp_;
//     } else if (in_message->lidar_frame_ != nullptr &&
//                in_message->lidar_frame_->timestamp > 0.0) {
//       trace_id = static_cast<uint64_t>(in_message->lidar_frame_->timestamp * 1e9);
//     } else if (in_message->timestamp_ > 0.0) {
//       trace_id = static_cast<uint64_t>(in_message->timestamp_ * 1e9);
//     }
//   }

//   ObjectFilterOptions filter_options;

//   const uint64_t f0 = apollo::perf::NowNs();
//   PERF_BLOCK("filter_bank")
//   if (use_object_filter_bank_) {
//     if (!filter_bank_.Filter(filter_options, in_message->lidar_frame_.get())) {
//       AINFO << "Lidar detection filter bank error.";
//       return false;
//     }
//   }
//   PERF_BLOCK_END
//   const uint64_t f1 = apollo::perf::NowNs();

//   apollo::perf::AppendLatency("perception.lidar_detection_filter.filter_bank",
//                               trace_id, f0, f1);
//   return true;
// }


// bool CameraTrackingComponent::InternalProc(
//     const std::shared_ptr<const onboard::CameraFrame>& in_message,
//     std::shared_ptr<onboard::SensorFrameMessage> out_message) {
//   out_message->timestamp_ = in_message->timestamp;
//   std::shared_ptr<CameraTrackingFrame> tracking_frame;
//   tracking_frame.reset(new CameraTrackingFrame);
//   tracking_frame->frame_id = in_message->frame_id;
//   tracking_frame->timestamp = in_message->timestamp;
//   tracking_frame->data_provider = in_message->data_provider;
//   tracking_frame->detected_objects = in_message->detected_objects;
//   tracking_frame->feature_blob = in_message->feature_blob;
//   tracking_frame->track_feature_blob.reset(new base::Blob<float>());
//   tracking_frame->project_matrix.setIdentity();
//   tracking_frame->camera2world_pose = in_message->camera2world_pose;

//   // Tracking
//   PERF_BLOCK("camera_tracking")
//   bool res = camera_obstacle_tracker_->Process(tracking_frame);
//   PERF_BLOCK_END

//   if (!res) {
//     out_message->error_code_ =
//         apollo::common::ErrorCode::PERCEPTION_ERROR_PROCESS;
//     AERROR << "Camera tracking process error!";
//     return false;
//   }

//   base::SensorInfo sensor_info;
//   std::string camera_name = in_message->data_provider->sensor_name();
//   if (!(sensor_manager_->GetSensorInfo(camera_name, &sensor_info))) {
//     AERROR << "Failed to get sensor info, sensor name: " << camera_name;
//     return false;
//   }

//   auto& frame = out_message->frame_;
//   frame = base::FramePool::Instance().Get();
//   // todo(wxt): check if sensor_info needed
//   // frame->sensor_info = lidar_frame->sensor_info;
//   frame->sensor_info = sensor_info;
//   frame->timestamp = in_message->timestamp;
//   frame->objects = tracking_frame->tracked_objects;
//   // todo(wxt): check if sensor2world_pose needed
//   frame->sensor2world_pose = tracking_frame->camera2world_pose;
//   return true;
// }
bool CameraTrackingComponent::InternalProc(
   const std::shared_ptr<const onboard::CameraFrame>& in_message, std::shared_ptr<onboard::SensorFrameMessage> out_message) {
     uint64_t trace_id = 0; 
     if (in_message != nullptr && in_message->timestamp > 0.0) { trace_id = static_cast<uint64_t>(in_message->timestamp * 1e9); 
    }

    out_message->timestamp_ = in_message->timestamp;
    const uint64_t c1 = apollo::perf::NowNs();
    apollo::perf::AppendPoint("perception.camera_tracking.copy_timestamp", trace_id, c1);
    const uint64_t cr0 = apollo::perf::NowNs();
    std::shared_ptr<CameraTrackingFrame> tracking_frame; tracking_frame.reset(new CameraTrackingFrame);
     tracking_frame->frame_id = in_message->frame_id; tracking_frame->timestamp = in_message->timestamp; tracking_frame->data_provider = in_message->data_provider; 
     tracking_frame->detected_objects = in_message->detected_objects; 
     tracking_frame->feature_blob = in_message->feature_blob; tracking_frame->track_feature_blob.reset(new base::Blob<float>()); tracking_frame->project_matrix.setIdentity(); 
     tracking_frame->camera2world_pose = in_message->camera2world_pose;

    
    const uint64_t cr1 = apollo::perf::NowNs();
    apollo::perf::AppendLatency("perception.camera_tracking.create_frame", trace_id, cr0, cr1);

    const uint64_t tr0 = apollo::perf::NowNs(); 
    PERF_BLOCK("camera_tracking") 
    bool res = camera_obstacle_tracker_->Process(tracking_frame); 
    PERF_BLOCK_END const uint64_t tr1 = apollo::perf::NowNs(); 
    apollo::perf::AppendLatency("perception.camera_tracking.tracker", trace_id, tr0, tr1); 
     if (!res) { 
      out_message->error_code_ = apollo::common::ErrorCode::PERCEPTION_ERROR_PROCESS; 
      AERROR << "Camera tracking process error!"; 
      return false; 
    }

    const uint64_t s0 = apollo::perf::NowNs();
     base::SensorInfo sensor_info; 
     std::string camera_name = in_message->data_provider->sensor_name(); 
     if (!(sensor_manager_->GetSensorInfo(camera_name, &sensor_info))) 
     { 
      AERROR << "Failed to get sensor info, sensor name: " << camera_name; 
      return false; 
    }
    const uint64_t s1 = apollo::perf::NowNs();
    apollo::perf::AppendLatency("perception.camera_tracking.get_sensor_info", trace_id, s0, s1);

    const uint64_t b0 = apollo::perf::NowNs();
     auto& frame = out_message->frame_; 
     frame = base::FramePool::Instance().Get(); 
     frame->sensor_info = sensor_info; 
     frame->timestamp = in_message->timestamp; 
     frame->objects = tracking_frame->tracked_objects; 
     frame->sensor2world_pose = tracking_frame->camera2world_pose;
    const uint64_t b1 = apollo::perf::NowNs();
    apollo::perf::AppendLatency("perception.camera_tracking.build_output", trace_id, b0, b1);
     return true; }


}  // namespace camera
}  // namespace perception
}  // namespace apollo
