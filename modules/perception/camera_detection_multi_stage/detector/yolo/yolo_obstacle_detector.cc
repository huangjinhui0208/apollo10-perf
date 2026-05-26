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
#include <iomanip>
#include <sstream>

#include "modules/perception/common/camera/common/camera_gpu_profile_utils.h"
#include "modules/perception/camera_detection_multi_stage/detector/yolo/yolo_obstacle_detector.h"

#include "cyber/common/file.h"
#include "cyber/common/log.h"
#include "modules/common/perf/latency_trace.h"
#include "cyber/profiler/profiler.h"
#include "modules/perception/common/camera/common/util.h"
#include "modules/perception/common/inference/utils/resize.h"
#include "modules/perception/common/util.h"

namespace apollo {
namespace perception {
namespace camera {

#if GPU_PLATFORM == NVIDIA
bool YoloObstacleDetector::InitGpuProfileResources() {
  if (!gpu_profile_enabled_) {
    gpu_event_ready_ = false;
    return false;
  }


  auto create_event = [ ](cudaEvent_t* event, const char* name) -> bool {

    if (*event == nullptr) {
      if (cudaEventCreate(event) != cudaSuccess) {
        *event = nullptr;
        AWARN << "[CAM_GPU_PROFILE] cudaEventCreate failed: " << name;
        return false;
      }
    }
    return true;
  };

  bool ok = true;
  ok = create_event(&infer_event_start_, "infer_event_start") && ok;
  ok = create_event(&infer_event_stop_, "infer_event_stop") && ok;
  ok = create_event(&getobj_event_start_, "getobj_event_start") && ok;
  ok = create_event(&getobj_event_stop_, "getobj_event_stop") && ok;

  gpu_event_ready_ = ok;
  return ok;
}

void YoloObstacleDetector::ReleaseGpuProfileResources() {
  if (infer_event_start_ != nullptr) {
    cudaEventDestroy(infer_event_start_);
    infer_event_start_ = nullptr;
  }
  if (infer_event_stop_ != nullptr) {
    cudaEventDestroy(infer_event_stop_);
    infer_event_stop_ = nullptr;
  }
  if (getobj_event_start_ != nullptr) {
    cudaEventDestroy(getobj_event_start_);
    getobj_event_start_ = nullptr;
  }
  if (getobj_event_stop_ != nullptr) {
    cudaEventDestroy(getobj_event_stop_);
    getobj_event_stop_ = nullptr;
  }

  gpu_event_ready_ = false;
}
#endif


void YoloObstacleDetector::LoadInputShape(const yolo::ModelParam &model_param) {
  float offset_ratio = model_param.offset_ratio();
  float cropped_ratio = model_param.cropped_ratio();
  int resized_width = model_param.resized_width();
  int aligned_pixel = model_param.aligned_pixel();

  int image_width = options_.image_width;
  int image_height = options_.image_height;

  offset_y_ = static_cast<int>(std::round(offset_ratio * image_height));
  float roi_ratio = cropped_ratio * image_height / image_width;
  width_ = (resized_width + aligned_pixel / 2) / aligned_pixel * aligned_pixel;
  height_ = static_cast<int>(width_ * roi_ratio + aligned_pixel / 2) /
            aligned_pixel * aligned_pixel;

  AINFO << "image_height=" << image_height << ", image_width=" << image_width
        << ", roi_ratio=" << roi_ratio;
  AINFO << "offset_y=" << offset_y_ << ", height=" << height_
        << ", width=" << width_;
}

void YoloObstacleDetector::LoadParam(const yolo::ModelParam &model_param) {
  confidence_threshold_ = model_param.confidence_threshold();
  light_vis_conf_threshold_ = model_param.light_vis_conf_threshold();
  light_swt_conf_threshold_ = model_param.light_swt_conf_threshold();

  auto min_dims = model_param.min_dims();
  min_dims_.min_2d_height = min_dims.min_2d_height();
  min_dims_.min_3d_height = min_dims.min_3d_height();
  min_dims_.min_3d_width = min_dims.min_3d_width();
  min_dims_.min_3d_length = min_dims.min_3d_length();

  ori_cycle_ = model_param.ori_cycle();

  border_ratio_ = model_param.border_ratio();

  // init NMS
  const auto &nms_param = model_param.nms_param();
  nms_.sigma = nms_param.sigma();
  nms_.type = nms_param.type();
  nms_.threshold = nms_param.threshold();
  nms_.inter_cls_nms_thresh = nms_param.inter_cls_nms_thresh();
  nms_.inter_cls_conf_thresh = nms_param.inter_cls_conf_thresh();
}

void YoloObstacleDetector::InitYoloBlob() {
  auto obj_blob_scale1 = net_->get_blob("obj_pred");
  auto obj_blob_scale2 = net_->get_blob("det2_obj_blob");
  auto obj_blob_scale3 = net_->get_blob("det3_obj_blob");
  int output_height_scale1 = obj_blob_scale1->shape(1);
  int output_width_scale1 = obj_blob_scale1->shape(2);
  int obj_size = output_height_scale1 * output_width_scale1 *
                 static_cast<int>(anchors_.size()) / anchorSizeFactor;
  // todo(daohu527): why just obj_blob_scale2 != nullptr
  if (obj_blob_scale2) {
    int output_height_scale2 = obj_blob_scale2->shape(1);
    int output_width_scale2 = obj_blob_scale2->shape(2);
    int output_height_scale3 = obj_blob_scale3->shape(1);
    int output_width_scale3 = obj_blob_scale3->shape(2);
    obj_size = (output_height_scale1 * output_width_scale1 +
                output_height_scale2 * output_width_scale2 +
                output_height_scale3 * output_width_scale3) *
               static_cast<int>(anchors_.size()) / anchorSizeFactor / numScales;
  }

  yolo_blobs_.res_box_blob.reset(
      new base::Blob<float>(1, 1, obj_size, kBoxBlockSize));
  yolo_blobs_.res_cls_blob.reset(new base::Blob<float>(
      1, 1, static_cast<int>(types_.size() + 1), obj_size));
  yolo_blobs_.res_cls_blob->cpu_data();
  overlapped_.reset(
      new base::Blob<bool>(std::vector<int>{obj_k_, obj_k_}, true));
  overlapped_->cpu_data();
  overlapped_->gpu_data();
  idx_sm_.reset(new base::Blob<int>(std::vector<int>{obj_k_}, true));
  yolo_blobs_.anchor_blob.reset(
      new base::Blob<float>(1, 1, static_cast<int>(anchors_.size() / 2), 2));
  yolo_blobs_.expand_blob.reset(
      new base::Blob<float>(1, 1, 1, static_cast<int>(expands_.size())));
  auto expand_cpu_data = yolo_blobs_.expand_blob->mutable_cpu_data();
  memcpy(expand_cpu_data, expands_.data(), expands_.size() * sizeof(float));
  auto anchor_cpu_data = yolo_blobs_.anchor_blob->mutable_cpu_data();
  memcpy(anchor_cpu_data, anchors_.data(), anchors_.size() * sizeof(float));
  yolo_blobs_.anchor_blob->gpu_data();

  yolo_blobs_.det1_loc_blob = net_->get_blob("loc_pred");
  yolo_blobs_.det1_obj_blob = net_->get_blob("obj_pred");
  yolo_blobs_.det1_cls_blob = net_->get_blob("cls_pred");
  yolo_blobs_.det1_ori_conf_blob = net_->get_blob("detect1_ori_conf_pred");
  yolo_blobs_.det1_ori_blob = net_->get_blob("ori_pred");
  yolo_blobs_.det1_dim_blob = net_->get_blob("dim_pred");
  yolo_blobs_.det2_loc_blob = net_->get_blob("detect2_loc_pred");
  yolo_blobs_.det2_obj_blob = net_->get_blob("detect2_obj_pred");
  yolo_blobs_.det2_cls_blob = net_->get_blob("detect2_cls_pred");
  yolo_blobs_.det2_ori_conf_blob = net_->get_blob("detect2_ori_conf_pred");
  yolo_blobs_.det2_ori_blob = net_->get_blob("detect2_ori_pred");
  yolo_blobs_.det2_dim_blob = net_->get_blob("detect2_dim_pred");
  yolo_blobs_.det3_loc_blob = net_->get_blob("detect3_loc_pred");
  yolo_blobs_.det3_obj_blob = net_->get_blob("detect3_obj_pred");
  yolo_blobs_.det3_cls_blob = net_->get_blob("detect3_cls_pred");
  yolo_blobs_.det3_ori_conf_blob = net_->get_blob("detect3_ori_conf_pred");
  yolo_blobs_.det3_ori_blob = net_->get_blob("detect3_ori_pred");
  yolo_blobs_.det3_dim_blob = net_->get_blob("detect3_dim_pred");

  yolo_blobs_.lof_blob = net_->get_blob("lof_pred");
  yolo_blobs_.lor_blob = net_->get_blob("lor_pred");

  yolo_blobs_.brvis_blob = net_->get_blob("brvis_pred");
  yolo_blobs_.brswt_blob = net_->get_blob("brswt_pred");
  yolo_blobs_.ltvis_blob = net_->get_blob("ltvis_pred");
  yolo_blobs_.ltswt_blob = net_->get_blob("ltswt_pred");
  yolo_blobs_.rtvis_blob = net_->get_blob("rtvis_pred");
  yolo_blobs_.rtswt_blob = net_->get_blob("rtswt_pred");

  yolo_blobs_.area_id_blob = net_->get_blob("area_id_pred");
  yolo_blobs_.visible_ratio_blob = net_->get_blob("vis_pred");
  yolo_blobs_.cut_off_ratio_blob = net_->get_blob("cut_pred");

  yolo_blobs_.feat_blob = net_->get_blob("conv3_3");
}

bool YoloObstacleDetector::Init(const ObstacleDetectorInitOptions &options) {
  options_ = options;

  gpu_id_ = options.gpu_id;
  BASE_GPU_CHECK(cudaSetDevice(gpu_id_));
  BASE_GPU_CHECK(cudaStreamCreate(&stream_));

  std::string config_file =
      GetConfigFile(options.config_path, options.config_file);
  if (!cyber::common::GetProtoFromFile(config_file, &model_param_)) {
    AERROR << "Read proto_config failed! " << config_file;
    return false;
  }

  const auto &model_info = model_param_.info();
  std::string model_path = GetModelPath(model_info.name());
  std::string anchors_file =
      GetModelFile(model_path, model_info.anchors_file().file());
  std::string types_file =
      GetModelFile(model_path, model_info.types_file().file());
  std::string expand_file =
      GetModelFile(model_path, model_info.expand_file().file());
  LoadInputShape(model_param_);
  LoadParam(model_param_);
  min_dims_.min_2d_height /= static_cast<float>(height_);

  if (!LoadAnchors(anchors_file, &anchors_)) {
    return false;
  }
  if (!LoadTypes(types_file, &types_)) {
    return false;
  }
  if (!LoadExpand(expand_file, &expands_)) {
    return false;
  }
  ACHECK(expands_.size() == types_.size());

  if (!InitNetwork(model_param_.info(), model_path)) {
    AERROR << "Init network failed!";
    return false;
  }

  InitYoloBlob();
  //初始化event
  #if GPU_PLATFORM == NVIDIA
   InitGpuProfileResources();
  #endif
  return true;
}

// bool YoloObstacleDetector::Detect(onboard::CameraFrame *frame) {
//   if (frame == nullptr) {
//     return false;
//   }

//   if (cudaSetDevice(gpu_id_) != cudaSuccess) {
//     AERROR << "Failed to set device to " << gpu_id_;
//     return false;
//   }

//   auto model_inputs = model_param_.info().inputs();
//   auto input_blob = net_->get_blob(model_inputs[0].name());

//   DataProvider::ImageOptions image_options;
//   image_options.target_color = base::Color::BGR;
//   image_options.crop_roi = base::RectI(0, offset_y_, options_.image_width,
//                                        options_.image_height - offset_y_);
//   image_options.do_crop = true;
//   base::Image8U image;
//   frame->data_provider->GetImage(image_options, &image);

//   inference::ResizeGPU(image, input_blob, frame->data_provider->src_width(), 0);
//   PERF_BLOCK("camera_2d_detector_infer")
//   net_->Infer();
//   PERF_BLOCK_END

//   PERF_BLOCK("camera_2d_detector_get_obj")
//   get_objects_cpu(yolo_blobs_, stream_, types_, nms_, model_param_,
//                   light_vis_conf_threshold_, light_swt_conf_threshold_,
//                   overlapped_.get(), idx_sm_.get(), &frame->detected_objects);
//   PERF_BLOCK_END

//   filter_bbox(min_dims_, &frame->detected_objects);

//   recover_bbox(frame->data_provider->src_width(),
//                frame->data_provider->src_height() - offset_y_, offset_y_,
//                &frame->detected_objects);

//   // appearance features for tracking
//   frame->feature_blob = yolo_blobs_.feat_blob;

//   // post processing
//   int left_boundary =
//       static_cast<int>(border_ratio_ * static_cast<float>(image.cols()));
//   int right_boundary = static_cast<int>((1.0f - border_ratio_) *
//                                         static_cast<float>(image.cols()));
//   for (auto &obj : frame->detected_objects) {
//     // recover alpha
//     obj->camera_supplement.alpha /= ori_cycle_;
//     // get area_id from visible_ratios
//     if (model_param_.num_areas() == 0) {
//       obj->camera_supplement.area_id =
//           get_area_id(obj->camera_supplement.visible_ratios);
//     }
//     // clear cut off ratios
//     auto &box = obj->camera_supplement.box;
//     if (box.xmin >= left_boundary) {
//       obj->camera_supplement.cut_off_ratios[2] = 0;
//     }
//     if (box.xmax <= right_boundary) {
//       obj->camera_supplement.cut_off_ratios[3] = 0;
//     }
//   }

//   return true;
// }
bool YoloObstacleDetector::Detect(onboard::CameraFrame *frame) {
  if (frame == nullptr) {
    return false;
  }

  if (frame->data_provider == nullptr) {
    AERROR << "YoloObstacleDetector::Detect failed: data_provider is nullptr.";
    return false;
  }
  const bool warmup = detect_call_idx_ < gpu_profile_warmup_frames_;
  ++detect_call_idx_;

  std::string camera_name = "unknown_camera";
  if (frame->data_provider != nullptr) {
    camera_name = frame->data_provider->sensor_name();
  }

  uint64_t trace_id = 0;
  if (frame->timestamp > 0.0) {
    trace_id = static_cast<uint64_t>(frame->timestamp * 1e9);
  }

  // std::string camera_name = frame->data_provider->sensor_name();
  if (camera_name.empty()) {
    camera_name = "unknown_camera";
  }

  const std::string perf_prefix =
      "perception.camera_detection_multi_stage." + camera_name + ".detector";

  auto record_latency = [&](const std::string &name,
                            uint64_t begin_ns,
                            uint64_t end_ns) {
    if (trace_id != 0) {
      apollo::perf::AppendLatency(perf_prefix + "." + name,
                                  trace_id,
                                  begin_ns,
                                  end_ns);
    }
  };

  auto record_point = [&](const std::string &name, uint64_t t_ns) {
    if (trace_id != 0) {
      apollo::perf::AppendPoint(perf_prefix + "." + name,
                                trace_id,
                                t_ns);
    }
  };

  const uint64_t total_begin_ns = apollo::perf::NowNs();

  // 1. cudaSetDevice
  const uint64_t set_device_begin_ns = apollo::perf::NowNs();
  if (cudaSetDevice(gpu_id_) != cudaSuccess) {
    const uint64_t set_device_end_ns = apollo::perf::NowNs();
    record_latency("set_device", set_device_begin_ns, set_device_end_ns);
    record_point("set_device.failed", set_device_end_ns);

    AERROR << "Failed to set device to " << gpu_id_
           << ", camera=" << camera_name
           << ", trace_id=" << trace_id;
    return false;
  }
  const uint64_t set_device_end_ns = apollo::perf::NowNs();
  record_latency("set_device", set_device_begin_ns, set_device_end_ns);

  // 2. 获取网络输入 blob
  const uint64_t get_blob_begin_ns = apollo::perf::NowNs();

  auto model_inputs = model_param_.info().inputs();
  if (model_inputs.size() <= 0) {
    const uint64_t get_blob_end_ns = apollo::perf::NowNs();
    record_latency("get_input_blob", get_blob_begin_ns, get_blob_end_ns);
    record_point("get_input_blob.failed", get_blob_end_ns);

    AERROR << "Yolo model has no input blob config."
           << " camera=" << camera_name
           << ", trace_id=" << trace_id;
    return false;
  }

  auto input_blob = net_->get_blob(model_inputs[0].name());
  if (input_blob == nullptr) {
    const uint64_t get_blob_end_ns = apollo::perf::NowNs();
    record_latency("get_input_blob", get_blob_begin_ns, get_blob_end_ns);
    record_point("get_input_blob.failed", get_blob_end_ns);

    AERROR << "Failed to get input blob: " << model_inputs[0].name()
           << ", camera=" << camera_name
           << ", trace_id=" << trace_id;
    return false;
  }

  const uint64_t get_blob_end_ns = apollo::perf::NowNs();
  record_latency("get_input_blob", get_blob_begin_ns, get_blob_end_ns);

  // 3. 准备 DataProvider::ImageOptions
  const uint64_t prepare_options_begin_ns = apollo::perf::NowNs();

  DataProvider::ImageOptions image_options;
  image_options.target_color = base::Color::BGR;
  image_options.crop_roi = base::RectI(
      0,
      offset_y_,
      options_.image_width,
      options_.image_height - offset_y_);
  image_options.do_crop = true;

  const uint64_t prepare_options_end_ns = apollo::perf::NowNs();
  record_latency("prepare_image_options",
                 prepare_options_begin_ns,
                 prepare_options_end_ns);

  // 4. 从 DataProvider 中取图像：BGR + crop
  base::Image8U image;

  const uint64_t get_image_begin_ns = apollo::perf::NowNs();
  const bool get_image_ok = frame->data_provider->GetImage(image_options, &image);
  const uint64_t get_image_end_ns = apollo::perf::NowNs();

  record_latency("get_image", get_image_begin_ns, get_image_end_ns);

  if (!get_image_ok) {
    record_point("get_image.failed", get_image_end_ns);

    AERROR << "Failed to get image from DataProvider."
           << " camera=" << camera_name
           << ", trace_id=" << trace_id
           << ", image_options=" << image_options.ToString();
    return false;
  }

  // 5. Resize 到网络输入尺寸
  const uint64_t resize_begin_ns = apollo::perf::NowNs();

  inference::ResizeGPU(
      image,
      input_blob,
      frame->data_provider->src_width(),
      0);

  const uint64_t resize_end_ns = apollo::perf::NowNs();
  record_latency("resize_gpu", resize_begin_ns, resize_end_ns);

  // 6. YOLO 网络推理
  const uint64_t infer_begin_ns = apollo::perf::NowNs();

  // PERF_BLOCK("camera_2d_detector_infer")
  // net_->Infer();
  // PERF_BLOCK_END


  // detector.infer：当前先用 default stream。
  // 重要：stream=0 是合法 CUDA default stream，不是无效 stream。
  cudaStream_t infer_stream = static_cast<cudaStream_t>(0);
  const bool infer_stream_is_default =
      gpu_profile::IsDefaultStream(infer_stream);

  const char* infer_stream_scope = gpu_profile::StreamScope(infer_stream);
  const char* infer_gpu_event_confidence =
      gpu_profile::GpuEventConfidence(infer_stream, true);

  auto infer_mem_before = gpu_profile::ReadCudaMemSnapshot();

  float infer_gpu_ms = -1.0f;
  bool infer_event_ok = gpu_event_ready_;

  const std::string infer_nvtx_name = gpu_profile::MakeNvtxName(
      "detector.infer", camera_name, trace_id);

  {
    gpu_profile::NvtxRangeGuard infer_nvtx(infer_nvtx_name);

    if (infer_event_ok) {
      if (cudaEventRecord(infer_event_start_, infer_stream) != cudaSuccess) {
        infer_event_ok = false;
        AWARN << "[CAM_GPU_PROFILE] cudaEventRecord(infer start) failed.";
      }
    }

    const uint64_t infer_wall_begin_ns = apollo::perf::NowNs();

    PERF_BLOCK("camera_2d_detector_infer")
    net_->Infer();
    PERF_BLOCK_END

    const uint64_t infer_wall_end_ns = apollo::perf::NowNs();

    if (infer_event_ok) {
      if (cudaEventRecord(infer_event_stop_, infer_stream) != cudaSuccess) {
        infer_event_ok = false;
        AWARN << "[CAM_GPU_PROFILE] cudaEventRecord(infer stop) failed.";
      }
    }

    if (infer_event_ok) {
      if (cudaEventSynchronize(infer_event_stop_) != cudaSuccess) {
        infer_event_ok = false;
        AWARN << "[CAM_GPU_PROFILE] cudaEventSynchronize(infer stop) failed.";
      }
    }

    if (infer_event_ok) {
      if (cudaEventElapsedTime(&infer_gpu_ms,
                               infer_event_start_,
                               infer_event_stop_) != cudaSuccess) {
        infer_event_ok = false;
        AWARN << "[CAM_GPU_PROFILE] cudaEventElapsedTime(infer) failed.";
      }
    }

    auto infer_mem_after = gpu_profile::ReadCudaMemSnapshot();

    const double infer_wall_ms =
        static_cast<double>(infer_wall_end_ns - infer_wall_begin_ns) / 1.0e6;

    const double mem_before_used_mib = infer_mem_before.used_mib();
    const double mem_after_used_mib = infer_mem_after.used_mib();
    const double mem_delta_used_mib =
        (infer_mem_before.ok && infer_mem_after.ok)
            ? (mem_after_used_mib - mem_before_used_mib)
            : -1.0;

    if (trace_id != 0) {
      apollo::perf::AppendLatency(
          "perception.camera_detection_multi_stage." + camera_name +
              ".detector.infer",
          trace_id, infer_wall_begin_ns, infer_wall_end_ns);
    }

    AINFO << std::fixed << std::setprecision(3)
          << "[CAM_GPU_PROFILE] "
          << "trace_id=" << trace_id
          << " camera=" << camera_name
          << " stage=detector.infer"
          << " warmup=" << (warmup ? 1 : 0)
          << " infer_stream_addr="
          << reinterpret_cast<uintptr_t>(infer_stream)
          << " infer_stream_is_default="
          << (infer_stream_is_default ? 1 : 0)
          << " stream_scope=" << infer_stream_scope
          << " stream_source=default_fallback"
          << " gpu_event_confidence=" << infer_gpu_event_confidence
          << " wall_ms=" << infer_wall_ms
          << " gpu_event_ok=" << (infer_event_ok ? 1 : 0)
          << " gpu_event_ms=" << infer_gpu_ms
          << " wall_minus_gpu_ms="
          << (infer_event_ok ? infer_wall_ms - infer_gpu_ms : -1.0)
          << " mem_before_used_mib=" << mem_before_used_mib
          << " mem_after_used_mib=" << mem_after_used_mib
          << " mem_delta_used_mib=" << mem_delta_used_mib
          << " cuda_last_error="
          << cudaGetErrorString(cudaGetLastError());
  }


  const uint64_t infer_end_ns = apollo::perf::NowNs();
  record_latency("infer", infer_begin_ns, infer_end_ns);

  // 7. YOLO 后处理：解码预测结果、NMS、生成 detected_objects
  const uint64_t get_objects_begin_ns = apollo::perf::NowNs();

  // PERF_BLOCK("camera_2d_detector_get_obj")
  // get_objects_cpu(yolo_blobs_,
  //                 stream_,
  //                 types_,
  //                 nms_,
  //                 model_param_,
  //                 light_vis_conf_threshold_,
  //                 light_swt_conf_threshold_,
  //                 overlapped_.get(),
  //                 idx_sm_.get(),
  //                 &frame->detected_objects);
  // PERF_BLOCK_END

  // get_objects_cpu 使用 YoloObstacleDetector::stream_。
  // Apollo 源码中 stream_ 在 Init() 里 cudaStreamCreate(&stream_)，
  // 并传给 get_objects_cpu(...)。
  cudaStream_t getobj_stream = stream_;
  const bool getobj_stream_is_default =
      gpu_profile::IsDefaultStream(getobj_stream);

  const char* getobj_stream_scope = gpu_profile::StreamScope(getobj_stream);
  const char* getobj_gpu_event_confidence =
      gpu_profile::GpuEventConfidence(getobj_stream, true);

  auto getobj_mem_before = gpu_profile::ReadCudaMemSnapshot();

  float getobj_gpu_ms = -1.0f;
  bool getobj_event_ok = gpu_event_ready_;

  const std::string getobj_nvtx_name = gpu_profile::MakeNvtxName(
      "detector.get_objects_cpu", camera_name, trace_id);

  {
    gpu_profile::NvtxRangeGuard getobj_nvtx(getobj_nvtx_name);

    if (getobj_event_ok) {
      if (cudaEventRecord(getobj_event_start_, getobj_stream) != cudaSuccess) {
        getobj_event_ok = false;
        AWARN << "[CAM_GPU_PROFILE] cudaEventRecord(getobj start) failed.";
      }
    }

    const uint64_t getobj_wall_begin_ns = apollo::perf::NowNs();

    PERF_BLOCK("camera_2d_detector_get_obj")
    get_objects_cpu(yolo_blobs_,
                    stream_,
                    types_,
                    nms_,
                    model_param_,
                    light_vis_conf_threshold_,
                    light_swt_conf_threshold_,
                    overlapped_.get(),
                    idx_sm_.get(),
                    &frame->detected_objects);
    PERF_BLOCK_END

    const uint64_t getobj_wall_end_ns = apollo::perf::NowNs();

    if (getobj_event_ok) {
      if (cudaEventRecord(getobj_event_stop_, getobj_stream) != cudaSuccess) {
        getobj_event_ok = false;
        AWARN << "[CAM_GPU_PROFILE] cudaEventRecord(getobj stop) failed.";
      }
    }

    if (getobj_event_ok) {
      if (cudaEventSynchronize(getobj_event_stop_) != cudaSuccess) {
        getobj_event_ok = false;
        AWARN << "[CAM_GPU_PROFILE] cudaEventSynchronize(getobj stop) failed.";
      }
    }

    if (getobj_event_ok) {
      if (cudaEventElapsedTime(&getobj_gpu_ms,
                               getobj_event_start_,
                               getobj_event_stop_) != cudaSuccess) {
        getobj_event_ok = false;
        AWARN << "[CAM_GPU_PROFILE] cudaEventElapsedTime(getobj) failed.";
      }
    }

    auto getobj_mem_after = gpu_profile::ReadCudaMemSnapshot();

    const double getobj_wall_ms =
        static_cast<double>(getobj_wall_end_ns - getobj_wall_begin_ns) / 1.0e6;

    const double mem_before_used_mib = getobj_mem_before.used_mib();
    const double mem_after_used_mib = getobj_mem_after.used_mib();
    const double mem_delta_used_mib =
        (getobj_mem_before.ok && getobj_mem_after.ok)
            ? (mem_after_used_mib - mem_before_used_mib)
            : -1.0;

    if (trace_id != 0) {
      apollo::perf::AppendLatency(
          "perception.camera_detection_multi_stage." + camera_name +
              ".detector.get_objects_cpu",
          trace_id, getobj_wall_begin_ns, getobj_wall_end_ns);
    }

    AINFO << std::fixed << std::setprecision(3)
          << "[CAM_GPU_PROFILE] "
          << "trace_id=" << trace_id
          << " camera=" << camera_name
          << " stage=detector.get_objects_cpu"
          << " warmup=" << (warmup ? 1 : 0)
          << " stream_addr=" << reinterpret_cast<uintptr_t>(getobj_stream)
          << " stream_is_default=" << (getobj_stream_is_default ? 1 : 0)
          << " stream_scope=" << getobj_stream_scope
          << " stream_source=yolo_stream_"
          << " gpu_event_confidence=" << getobj_gpu_event_confidence
          << " wall_ms=" << getobj_wall_ms
          << " gpu_event_ok=" << (getobj_event_ok ? 1 : 0)
          << " gpu_event_ms=" << getobj_gpu_ms
          << " wall_minus_gpu_ms="
          << (getobj_event_ok ? getobj_wall_ms - getobj_gpu_ms : -1.0)
          << " detected_object_count=" << frame->detected_objects.size()
          << " mem_before_used_mib=" << mem_before_used_mib
          << " mem_after_used_mib=" << mem_after_used_mib
          << " mem_delta_used_mib=" << mem_delta_used_mib
          << " cuda_last_error="
          << cudaGetErrorString(cudaGetLastError());
  }


  const uint64_t get_objects_end_ns = apollo::perf::NowNs();
  record_latency("get_objects_cpu",
                 get_objects_begin_ns,
                 get_objects_end_ns);

  // 8. bbox 过滤
  const uint64_t filter_bbox_begin_ns = apollo::perf::NowNs();

  filter_bbox(min_dims_, &frame->detected_objects);

  const uint64_t filter_bbox_end_ns = apollo::perf::NowNs();
  record_latency("filter_bbox",
                 filter_bbox_begin_ns,
                 filter_bbox_end_ns);

  // 9. bbox 坐标恢复
  const uint64_t recover_bbox_begin_ns = apollo::perf::NowNs();

  recover_bbox(frame->data_provider->src_width(),
               frame->data_provider->src_height() - offset_y_,
               offset_y_,
               &frame->detected_objects);

  const uint64_t recover_bbox_end_ns = apollo::perf::NowNs();
  record_latency("recover_bbox",
                 recover_bbox_begin_ns,
                 recover_bbox_end_ns);

  // 10. 设置 tracking 用的 appearance feature
  const uint64_t set_feature_begin_ns = apollo::perf::NowNs();

  frame->feature_blob = yolo_blobs_.feat_blob;

  const uint64_t set_feature_end_ns = apollo::perf::NowNs();
  record_latency("set_feature_blob",
                 set_feature_begin_ns,
                 set_feature_end_ns);

  // 11. object 后处理循环
  const uint64_t object_loop_begin_ns = apollo::perf::NowNs();

  int left_boundary =
      static_cast<int>(border_ratio_ * static_cast<float>(image.cols()));
  int right_boundary =
      static_cast<int>((1.0f - border_ratio_) *
                       static_cast<float>(image.cols()));

  for (auto &obj : frame->detected_objects) {
    // recover alpha
    obj->camera_supplement.alpha /= ori_cycle_;

    // get area_id from visible_ratios
    if (model_param_.num_areas() == 0) {
      obj->camera_supplement.area_id =
          get_area_id(obj->camera_supplement.visible_ratios);
    }

    // clear cut off ratios
    auto &box = obj->camera_supplement.box;
    if (box.xmin >= left_boundary) {
      obj->camera_supplement.cut_off_ratios[2] = 0;
    }
    if (box.xmax <= right_boundary) {
      obj->camera_supplement.cut_off_ratios[3] = 0;
    }
  }

  const uint64_t object_loop_end_ns = apollo::perf::NowNs();
  record_latency("object_loop",
                 object_loop_begin_ns,
                 object_loop_end_ns);

  // 12. detector 总耗时
  const uint64_t total_end_ns = apollo::perf::NowNs();
  // record_latency("total", total_begin_ns, total_end_ns);
  apollo::perf::AppendLatency(
      "perception.camera_detection_multi_stage." + camera_name +
          ".detector.inner_total",
      trace_id, total_begin_ns, total_end_ns);

  return true;
}

REGISTER_OBSTACLE_DETECTOR(YoloObstacleDetector);

}  // namespace camera
}  // namespace perception
}  // namespace apollo
