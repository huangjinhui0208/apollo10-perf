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
#include "modules/perception/camera_detection_multi_stage/camera_detection_multi_stage_component.h"

#include "cyber/profiler/profiler.h"
#include "modules/perception/common/algorithm/sensor_manager/sensor_manager.h"
#include "modules/perception/common/base/camera.h"
#include "modules/perception/common/camera/common/data_provider.h"
#include "modules/common/perf/latency_trace.h"

namespace apollo {
namespace perception {
namespace camera {

bool CameraDetectionMultiStageComponent::InitObstacleDetector(
    const CameraDetectionMultiStage& detection_param) {
  ObstacleDetectorInitOptions init_options;
  // Init conf file
  auto plugin_param = detection_param.plugin_param();
  init_options.config_path = plugin_param.config_path();
  init_options.config_file = plugin_param.config_file();
  init_options.gpu_id = detection_param.gpu_id();
  timestamp_offset_ = detection_param.timestamp_offset();

  // Init camera params
  std::string camera_name = detection_param.camera_name();
  base::BaseCameraModelPtr model =
      algorithm::SensorManager::Instance()->GetUndistortCameraModel(
          camera_name);
  ACHECK(model) << "Can't find " << camera_name
                << " in data/conf/sensor_meta.pb.txt";
  auto pinhole = static_cast<base::PinholeCameraModel*>(model.get());
  init_options.intrinsic = pinhole->get_intrinsic_params();
  camera_k_matrix_ = init_options.intrinsic;
  init_options.image_height = model->get_height();
  init_options.image_width = model->get_width();
  image_height_ = model->get_height();
  image_width_ = model->get_width();

  // Init detector
  detector_.reset(
      BaseObstacleDetectorRegisterer::GetInstanceByName(plugin_param.name()));
  detector_->Init(init_options);
  return true;
}

bool CameraDetectionMultiStageComponent::InitCameraFrame(
    const CameraDetectionMultiStage& detection_param) {
  DataProvider::InitOptions init_options;
  init_options.image_height = image_height_;
  init_options.image_width = image_width_;
  init_options.do_undistortion = detection_param.enable_undistortion();
  init_options.sensor_name = detection_param.camera_name();
  init_options.device_id = detection_param.gpu_id();
  AINFO << "init_options.device_id: " << init_options.device_id
        << " camera_name: " << init_options.sensor_name;

  data_provider_ = std::make_shared<camera::DataProvider>();
  data_provider_->Init(init_options);

  return true;
}

bool CameraDetectionMultiStageComponent::InitTransformWrapper(
    const CameraDetectionMultiStage& detection_param) {
  trans_wrapper_.reset(new onboard::TransformWrapper());
  // tf_camera_frame_id
  trans_wrapper_->Init(detection_param.camera_name());
  return true;
}

bool CameraDetectionMultiStageComponent::Init() {
  CameraDetectionMultiStage detection_param;
  if (!GetProtoConfig(&detection_param)) {
    AERROR << "Load camera detection 3d component config failed!";
    return false;
  }

  camera_name_ = detection_param.camera_name();
  perf_prefix_ =
      "perception.camera_detection_multi_stage." + camera_name_;

  AINFO << "[CAMERA_MULTI_STAGE_PERF] camera_name=" << camera_name_
        << ", perf_prefix=" << perf_prefix_;

  InitObstacleDetector(detection_param);

  InitCameraFrame(detection_param);

  InitTransformWrapper(detection_param);

  writer_ = node_->CreateWriter<onboard::CameraFrame>(
      detection_param.channel().output_obstacles_channel_name());
  return true;
}

// bool CameraDetectionMultiStageComponent::Proc(
//     const std::shared_ptr<apollo::drivers::Image>& msg) {
//   PERF_FUNCTION()
//   std::shared_ptr<onboard::CameraFrame> out_message(new (std::nothrow)
//                                                         onboard::CameraFrame);
//   bool status = InternalProc(msg, out_message);
//   if (status) {
//     writer_->Write(out_message);
//     AINFO << "Send camera detection 2d output message.";
//   }

//   return status;
// }
bool CameraDetectionMultiStageComponent::Proc(
    const std::shared_ptr<apollo::drivers::Image>& msg) {
  PERF_FUNCTION();

  uint64_t trace_id = 0;
  if (msg != nullptr && msg->measurement_time() > 0.0) {
    trace_id = static_cast<uint64_t>(msg->measurement_time() * 1e9);
  }

  const uint64_t t0 = apollo::perf::NowNs();
  apollo::perf::AppendPoint(PerfEvent("tick"), trace_id, t0);

  std::shared_ptr<onboard::CameraFrame> out_message(
      new (std::nothrow) onboard::CameraFrame());

  const uint64_t p0 = apollo::perf::NowNs();
  bool status = InternalProc(msg, out_message);
  const uint64_t p1 = apollo::perf::NowNs();

  apollo::perf::AppendLatency(PerfEvent("proc.total"), trace_id, p0, p1);

  if (status) {
    const uint64_t w0 = apollo::perf::NowNs();
    writer_->Write(out_message);
    const uint64_t w1 = apollo::perf::NowNs();

    apollo::perf::AppendLatency(PerfEvent("write"), trace_id, w0, w1);
    AINFO << "Send camera detection 2d output message. camera="
          << camera_name_;
  }

  const uint64_t t1 = apollo::perf::NowNs();
  apollo::perf::AppendLatency(PerfEvent("total"), trace_id, t0, t1);

  return status;
}



// bool CameraDetectionMultiStageComponent::InternalProc(
//     const std::shared_ptr<apollo::drivers::Image>& msg,
//     const std::shared_ptr<onboard::CameraFrame>& out_message) {
//   out_message->data_provider = data_provider_;
//   // Fill image
//   // todo(daohu527): need use real memory size
//   out_message->data_provider->FillImageData(
//       image_height_, image_width_,
//       reinterpret_cast<const uint8_t*>(msg->data().data()), msg->encoding());

//   out_message->camera_k_matrix = camera_k_matrix_;

//   const double msg_timestamp = msg->measurement_time() + timestamp_offset_;
//   out_message->timestamp = msg_timestamp;

//   // Get sensor to world pose from TF
//   Eigen::Affine3d camera2world;
//   if (!trans_wrapper_->GetSensor2worldTrans(msg_timestamp, &camera2world)) {
//     const std::string err_str =
//         absl::StrCat("failed to get camera to world pose, ts: ", msg_timestamp,
//                      " frame_id: ", msg->frame_id());
//     AERROR << err_str;
//     return false;
//   }

//   out_message->frame_id = frame_id_;
//   ++frame_id_;

//   out_message->camera2world_pose = camera2world;

//   // Detect
//   PERF_BLOCK("camera_2d_detector")
//   detector_->Detect(out_message.get());
//   PERF_BLOCK_END
//   return true;
// }
bool CameraDetectionMultiStageComponent::InternalProc(
    const std::shared_ptr<apollo::drivers::Image>& msg,
    const std::shared_ptr<onboard::CameraFrame>& out_message) {
  uint64_t trace_id = 0;
  if (msg != nullptr && msg->measurement_time() > 0.0) {
    trace_id = static_cast<uint64_t>(msg->measurement_time() * 1e9);
  }

  out_message->data_provider = data_provider_;

  // 1. Fill image：建议使用重载版本，在 DataProvider 内部分解
  bool fill_ok = out_message->data_provider->FillImageData(
      image_height_, image_width_,
      reinterpret_cast<const uint8_t*>(msg->data().data()),
      msg->encoding(),
      perf_prefix_,
      trace_id);

  if (!fill_ok) {
    const uint64_t fail_t = apollo::perf::NowNs();
    apollo::perf::AppendPoint(PerfEvent("fill_image.failed"), trace_id, fail_t);
    AERROR << "FillImageData failed, camera=" << camera_name_
           << ", encoding=" << msg->encoding()
           << ", trace_id=" << trace_id;
    return false;
  }

  // 2. 设置 frame 元信息
  const uint64_t meta0 = apollo::perf::NowNs();
  out_message->camera_k_matrix = camera_k_matrix_;
  const double msg_timestamp = msg->measurement_time() + timestamp_offset_;
  out_message->timestamp = msg_timestamp;
  out_message->frame_id = frame_id_;
  ++frame_id_;
  const uint64_t meta1 = apollo::perf::NowNs();

  apollo::perf::AppendLatency(PerfEvent("set_frame_meta"),
                              trace_id, meta0, meta1);

  // 3. TF
  Eigen::Affine3d camera2world;
  const uint64_t tf0 = apollo::perf::NowNs();
  bool tf_ok = trans_wrapper_->GetSensor2worldTrans(msg_timestamp,
                                                    &camera2world);
  const uint64_t tf1 = apollo::perf::NowNs();

  apollo::perf::AppendLatency(PerfEvent("get_tf"), trace_id, tf0, tf1);

  if (!tf_ok) {
    const uint64_t fail_t = apollo::perf::NowNs();
    apollo::perf::AppendPoint(PerfEvent("get_tf.failed"), trace_id, fail_t);

    const std::string err_str = absl::StrCat(
        "failed to get camera to world pose, ts: ",
        msg_timestamp, " frame_id: ", msg->frame_id(),
        " camera: ", camera_name_);
    AERROR << err_str;
    return false;
  }

  out_message->camera2world_pose = camera2world;

  // 4. Detector 总耗时
  const uint64_t d0 = apollo::perf::NowNs();
  PERF_BLOCK("camera_2d_detector")
  bool det_ok = detector_->Detect(out_message.get());
  PERF_BLOCK_END
  const uint64_t d1 = apollo::perf::NowNs();

  apollo::perf::AppendLatency(PerfEvent("detector.total"),
                              trace_id, d0, d1);

  if (!det_ok) {
    const uint64_t fail_t = apollo::perf::NowNs();
    apollo::perf::AppendPoint(PerfEvent("detector.failed"), trace_id, fail_t);
    AERROR << "Detector failed, camera=" << camera_name_
           << ", trace_id=" << trace_id;
    return false;
  }

  return true;
}



}  // namespace camera
}  // namespace perception
}  // namespace apollo
