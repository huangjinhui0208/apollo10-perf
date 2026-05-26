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

#include "modules/common/perf/latency_trace.h"
#include "modules/perception/common/camera/common/camera_gpu_profile_utils.h"

#include "modules/perception/camera_detection_multi_stage/detector/yolox3d/yolox3d_obstacle_detector.h"

#include "cyber/common/file.h"
#include "cyber/common/log.h"
#include "cyber/profiler/profiler.h"
#include "modules/perception/camera_detection_multi_stage/detector/yolox3d/postprocess.h"
#include "modules/perception/common/base/image.h"
#include "modules/perception/common/inference/inference_factory.h"
#include "modules/perception/common/util.h"

namespace apollo {
namespace perception {
namespace camera {

using cyber::common::GetAbsolutePath;
#if GPU_PLATFORM == NVIDIA
bool Yolox3DObstacleDetector::InitGpuProfileResources() {
  if (!gpu_profile_enabled_) {
    gpu_event_ready_ = false;
    return false;
  }

  auto create_event = [](cudaEvent_t* event, const char* name) -> bool {
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
  ok = create_event(&infer2d_event_start_, "infer2d_event_start") && ok;
  ok = create_event(&infer2d_event_stop_, "infer2d_event_stop") && ok;
  ok = create_event(&infer3d_event_start_, "infer3d_event_start") && ok;
  ok = create_event(&infer3d_event_stop_, "infer3d_event_stop") && ok;

  gpu_event_ready_ = ok;
  return ok;
}

void Yolox3DObstacleDetector::ReleaseGpuProfileResources() {
  if (infer2d_event_start_ != nullptr) {
    cudaEventDestroy(infer2d_event_start_);
    infer2d_event_start_ = nullptr;
  }
  if (infer2d_event_stop_ != nullptr) {
    cudaEventDestroy(infer2d_event_stop_);
    infer2d_event_stop_ = nullptr;
  }
  if (infer3d_event_start_ != nullptr) {
    cudaEventDestroy(infer3d_event_start_);
    infer3d_event_start_ = nullptr;
  }
  if (infer3d_event_stop_ != nullptr) {
    cudaEventDestroy(infer3d_event_stop_);
    infer3d_event_stop_ = nullptr;
  }

  gpu_event_ready_ = false;
}
#endif

void Yolox3DObstacleDetector::LoadInputShape(
    const yolox3d::ModelParam &model_param) {
  // loading config params
  width_ = model_param.resize().width();
  height_ = model_param.resize().height();

  image_width_ = options_.image_width;
  image_height_ = options_.image_height;

  AINFO << " image_height =" << image_height_
        << ", image_width=" << image_width_ << ", resize height=" << height_
        << ", width=" << width_;
}

void Yolox3DObstacleDetector::LoadParam(
    const yolox3d::ModelParam &model_param) {
  confidence_threshold_ = model_param.confidence_threshold();
  border_ratio_ = model_param.border_ratio();

  // Init NMS proto param by config file
  const auto &nms_param = model_param.nms_param();
  nms_.set_sigma(nms_param.sigma());
  nms_.set_type(nms_param.type());
  nms_.set_threshold(nms_param.threshold());
}

void AddShape3DYolox(
    std::map<std::string, std::vector<int>> *shape_map,
    const google::protobuf::RepeatedPtrField<common::ModelBlob> &model_blobs) {
  for (const auto &blob : model_blobs) {
    std::vector<int> shape(blob.shape().begin(), blob.shape().end());
    shape_map->insert(std::make_pair(blob.name(), shape));
  }
}

std::vector<std::string> GetBlobNames3DYolox(
    const google::protobuf::RepeatedPtrField<common::ModelBlob> &model_blobs) {
  std::vector<std::string> blob_names;
  for (const auto &blob : model_blobs) {
    blob_names.push_back(blob.name());
  }
  return blob_names;
}

bool Yolox3DObstacleDetector::Init3DNetwork(const common::ModelInfo &model_info,
                                            const std::string &model_path) {
  // Network files
  std::string proto_file =
      GetModelFile(model_path, model_info.proto_file().file());
  std::string weight_file =
      GetModelFile(model_path, model_info.weight_file().file());

  // Network input and output names
  std::vector<std::string> input_names =
      GetBlobNames3DYolox(model_info.inputs());
  std::vector<std::string> output_names =
      GetBlobNames3DYolox(model_info.outputs());

  // Network type
  const auto &framework = model_info.framework();
  std::string plugin_name = model_info.infer_plugin();
  static const std::string class_namespace = "apollo::perception::inference::";
  if (model_info.has_infer_plugin() && !plugin_name.empty()) {
    plugin_name = class_namespace + plugin_name;
    net_3D_ = apollo::cyber::plugin_manager::PluginManager::Instance()
                  ->CreateInstance<inference::Inference>(plugin_name);
    net_3D_->set_model_info(proto_file, input_names, output_names);
    AINFO << "net_3D load plugin success: " << plugin_name;
  } else {
    net_3D_.reset(inference::CreateInferenceByName(framework, proto_file,
                                                   weight_file, output_names,
                                                   input_names, model_path));
  }

  ACHECK(net_3D_ != nullptr);
  net_3D_->set_gpu_id(gpu_id_);

  std::map<std::string, std::vector<int>> shape_map;
  AddShape3DYolox(&shape_map, model_info.inputs());
  AddShape3DYolox(&shape_map, model_info.outputs());

  if (!net_3D_->Init(shape_map)) {
    AERROR << model_info.name() << "init failed!";
    return false;
  }
  return true;
}

cv::Mat resizeKeepAspectRatioYolox(const cv::Mat &input,
                                   const cv::Size &dstSize,
                                   const cv::Scalar &bgcolor) {
  cv::Mat output;

  double h1 = dstSize.width * (input.rows / static_cast<double>(input.cols));
  double w2 = dstSize.height * (input.cols / static_cast<double>(input.rows));
  if (h1 <= dstSize.height) {
    cv::resize(input, output, cv::Size(dstSize.width, h1));
  } else {
    cv::resize(input, output, cv::Size(w2, dstSize.height));
  }

  int top = (dstSize.height - output.rows) / 2;
  int down = (dstSize.height - output.rows + 1) / 2;
  int left = (dstSize.width - output.cols) / 2;
  int right = (dstSize.width - output.cols + 1) / 2;

  cv::copyMakeBorder(output, output, top, down, left, right,
                     cv::BORDER_CONSTANT, bgcolor);

  return output;
}

void Yolox3DObstacleDetector::Yolo3DInference(const base::Image8U *image,
                                              base::ObjectPtr obj) {
  ACHECK(image != nullptr);

  // get left top and right bottom point
  float left_x = Yoloxclamp(obj->camera_supplement.box.xmin, 0.f,
                            static_cast<float>(image->cols()));
  float left_y = Yoloxclamp(obj->camera_supplement.box.ymin, 0.f,
                            static_cast<float>(image->rows()));
  float right_x = Yoloxclamp(obj->camera_supplement.box.xmax, 0.f,
                             static_cast<float>(image->cols()));
  float right_y = Yoloxclamp(obj->camera_supplement.box.ymax, 0.f,
                             static_cast<float>(image->rows()));

  // get width and height of object
  float width = right_x - left_x;
  float height = right_y - left_y;

  // crop object
  cv::Rect object_roi(left_x, left_y, width, height);
  cv::Mat cropped_obj = img_(object_roi);

  // resize croped image to 224*224, no padding
  // cropped_obj =
  //     resizeKeepAspectRatioYolox(cropped_obj, cv::Size(224, 224), {0, 0, 0});
  cv::resize(cropped_obj, cropped_obj, cv::Size(224, 224));

  // normallize channel value from 0～255 to 0~1 and change it to float type
  cropped_obj.convertTo(cropped_obj, CV_32F, 1.0f / 255.0);

  // BGR
  std::vector<float> mean_values{0.406, 0.456, 0.485};
  std::vector<float> std_values{0.225, 0.224, 0.229};

  std::vector<cv::Mat> bgrChannels(3);
  cv::split(cropped_obj, bgrChannels);
  for (int i = 0; i < 3; ++i) {
    bgrChannels[i].convertTo(bgrChannels[i], CV_32FC1, 1 / std_values[i],
                             (0.0 - mean_values[i]) / std_values[i]);
  }
  cv::Mat dst;
  cv::merge(bgrChannels, dst);
  cropped_obj = dst;

  auto model_inputs = model_param_.info_3d().info().inputs();

  auto input_blob_3d = net_3D_->get_blob(model_inputs[0].name());

  ACHECK(input_blob_3d != nullptr);

  int model_input_chs = cropped_obj.channels();
  int model_input_rows = cropped_obj.rows;
  int model_input_cols = cropped_obj.cols;
  float *input_data = input_blob_3d->mutable_cpu_data();

  // copy to blob
  input_blob_3d->Reshape(
      {1, model_input_chs, model_input_rows, model_input_cols});
  for (int i = 0; i < model_input_chs; ++i) {
    cv::extractChannel(
        cropped_obj,
        cv::Mat(model_input_rows, model_input_cols, CV_32FC1,
                input_data + i * model_input_rows * model_input_cols),
        i);
  }

  // 3D model infer
  net_3D_->Infer();

  // get results
  auto model_outputs = model_param_.info_3d().info().outputs();
  auto blob_orient = net_3D_->get_blob(model_outputs[0].name());
  auto blob_conf = net_3D_->get_blob(model_outputs[1].name());
  auto blob_dim = net_3D_->get_blob(model_outputs[2].name());

  ACHECK(blob_orient != nullptr);
  ACHECK(blob_conf != nullptr);
  ACHECK(blob_dim != nullptr);

  const float *orient_data = blob_orient->cpu_data();
  const float *conf_data = blob_conf->cpu_data();
  const float *dim_data = blob_dim->cpu_data();

  // recover to alpha according to bin number
  const int bin_number = model_param_.info_3d().bin_num();
  int max_index = 0;
  int max_conf = conf_data[max_index];
  for (int i = 1; i < bin_number; ++i) {
    if (conf_data[i] > max_conf) {
      max_conf = conf_data[i];
      max_index = i;
    }
  }

  int orient_index = 2 * max_index;
  float alpha = 0.f;
  float cos_result = orient_data[orient_index];
  float sin_result = orient_data[orient_index + 1];
  float angle_offset = std::atan2(sin_result, cos_result);

  float wedge = 2 * M_PI / bin_number;
  alpha = angle_offset + max_index * wedge;
  alpha = alpha + wedge / 2 - M_PI;
  if (alpha > M_PI) {
    alpha -= 2 * M_PI;
  }
  obj->camera_supplement.alpha = alpha;

  // add dim to object template
  obj->size[0] += dim_data[2];
  obj->size[1] += dim_data[1];
  obj->size[2] += dim_data[0];
  return;
}

bool Yolox3DObstacleDetector::Init(const ObstacleDetectorInitOptions &options) {
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

  LoadInputShape(model_param_);
  LoadParam(model_param_);

  const auto &model_info = model_param_.info();
  std::string model_path = GetModelPath(model_info.name());
  if (!InitNetwork(model_info, model_path)) {
    AERROR << "Init network failed!";
    return false;
  }
  net_->Infer();
  AINFO << "yolox3d 2D model init success";
  std::cout << "yolox3d 2D model init success " << std::endl;

  const auto &model_info_3d = model_param_.info_3d().info();
  std::string model_path_3d = GetModelPath(model_info_3d.name());
  if (!Init3DNetwork(model_info_3d, model_path_3d)) {
    AERROR << "Init network failed!";
    return false;
  }
  net_3D_->Infer();
  AINFO << "yolox3d 3D model init success";
  std::cout << "yolox3d 3D model init success " << std::endl;

  #if GPU_PLATFORM == NVIDIA
    InitGpuProfileResources();
  #endif

  return true;
}

bool Yolox3DObstacleDetector::Preprocess(const base::Image8U *image,
                                         base::BlobPtr<float> input_blob) {
  ACHECK(image != nullptr);
  ACHECK(input_blob != nullptr);

  // init cv img containter, same to image row and col
  img_ = cv::Mat(image->rows(), image->cols(), CV_8UC3);
  memcpy(img_.data, image->cpu_data(),
         image->rows() * image->cols() * image->channels() * sizeof(uint8_t));

  // generate new pure black bg as same size as ratio
  float ratio = std::max(image_width_, image_height_);
  cv::Mat out(ratio, ratio, CV_8UC3, {114, 114, 114});  // 114 sames to yolox3d

  img_.copyTo(out(cv::Rect(0, 0, img_.cols,
                           img_.rows)));  // pad to bottom, sames to yolox3d

  cv::resize(out, out, cv::Size(width_, height_),
             cv::INTER_LINEAR);  // use INTER_LINEAR, sames to yolox3d

  out.convertTo(out, CV_32F, 1.0);  // without normalize, sames to yolox3d
  // cv::imwrite("pics/yolox3d_pad_image.png", out);

  int model_input_chs = out.channels();
  int model_input_rows = out.rows;
  int model_input_cols = out.cols;

  // fill input_blob -> tensor_image will be used by model
  input_blob->Reshape({1, model_input_chs, model_input_rows, model_input_cols});
  float *input_data = input_blob->mutable_cpu_data();

  for (int i = 0; i < model_input_chs; ++i) {
    cv::extractChannel(
        out,
        cv::Mat(model_input_rows, model_input_cols, CV_32FC1,
                input_data + i * model_input_rows * model_input_cols),
        i);
  }
  return true;
}

bool Yolox3DObstacleDetector::Detect(onboard::CameraFrame *frame) {
  // if (frame == nullptr) {
  //   return false;
  // }

  // if (cudaSetDevice(gpu_id_) != cudaSuccess) {
  //   AERROR << "Failed to set device to " << gpu_id_;
  //   return false;
  // }
  if (frame == nullptr) {
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

  const uint64_t detector_total_begin_ns = apollo::perf::NowNs();

  if (cudaSetDevice(gpu_id_) != cudaSuccess) {
    AERROR << "Failed to set device to " << gpu_id_;
    return false;
  }

  auto model_inputs = model_param_.info().inputs();
  auto input_blob_2d = net_->get_blob(model_inputs[0].name());

  // use bgr to infer
  DataProvider::ImageOptions image_options;
  image_options.target_color = base::Color::BGR;

  base::Image8U image;
  frame->data_provider->GetImage(image_options, &image);

  // cpu preprocess
  // PERF_BLOCK("2d image preprocess time: ")
  // Preprocess(&image, input_blob_2d);
  // PERF_BLOCK_END
  const std::string preprocess_nvtx_name = gpu_profile::MakeNvtxName(
    "detector.preprocess_cpu", camera_name, trace_id);
{
  gpu_profile::NvtxRangeGuard preprocess_nvtx(preprocess_nvtx_name);

  const uint64_t preprocess_begin_ns = apollo::perf::NowNs();
  PERF_BLOCK("2d image preprocess time: ")
  Preprocess(&image, input_blob_2d);
  PERF_BLOCK_END
  const uint64_t preprocess_end_ns = apollo::perf::NowNs();

  if (trace_id != 0) {
    apollo::perf::AppendLatency(
        "perception.camera_detection_multi_stage." + camera_name +
            ".detector.preprocess_cpu",
        trace_id, preprocess_begin_ns, preprocess_end_ns);
  }

  AINFO << std::fixed << std::setprecision(3)
        << "[CAM_GPU_PROFILE] "
        << "trace_id=" << trace_id
        << " camera=" << camera_name
        << " stage=detector.preprocess_cpu"
        << " warmup=" << (warmup ? 1 : 0)
        << " wall_ms="
        << static_cast<double>(preprocess_end_ns - preprocess_begin_ns) / 1.0e6;
}

  // model infer, save to output blob
  // PERF_BLOCK("2d infer time: ")
  // net_->Infer();
  // PERF_BLOCK_END
#if GPU_PLATFORM == NVIDIA
cudaStream_t infer2d_stream = static_cast<cudaStream_t>(0);
// 当前 Yolox3DObstacleDetector 没有直接暴露 net_ 内部真实 stream。
// 先按 default stream 标记，后续结合 Nsight Systems 判断是否存在默认流同步或 GPU 排队。
const bool infer2d_stream_is_default =
    gpu_profile::IsDefaultStream(infer2d_stream);
const char* infer2d_stream_scope =
    gpu_profile::StreamScope(infer2d_stream);
const char* infer2d_gpu_event_confidence =
    gpu_profile::GpuEventConfidence(infer2d_stream, true);

auto infer2d_mem_before = gpu_profile::ReadCudaMemSnapshot();
float infer2d_gpu_ms = -1.0f;
bool infer2d_event_ok = gpu_event_ready_;

const std::string infer2d_nvtx_name = gpu_profile::MakeNvtxName(
    "detector.infer2d", camera_name, trace_id);
{
  gpu_profile::NvtxRangeGuard infer2d_nvtx(infer2d_nvtx_name);

  if (infer2d_event_ok) {
    if (cudaEventRecord(infer2d_event_start_, infer2d_stream) != cudaSuccess) {
      infer2d_event_ok = false;
      AWARN << "[CAM_GPU_PROFILE] cudaEventRecord(infer2d start) failed.";
    }
  }

  const uint64_t infer2d_wall_begin_ns = apollo::perf::NowNs();
  PERF_BLOCK("2d infer time: ")
  net_->Infer();
  PERF_BLOCK_END
  const uint64_t infer2d_wall_end_ns = apollo::perf::NowNs();

  if (infer2d_event_ok) {
    if (cudaEventRecord(infer2d_event_stop_, infer2d_stream) != cudaSuccess) {
      infer2d_event_ok = false;
      AWARN << "[CAM_GPU_PROFILE] cudaEventRecord(infer2d stop) failed.";
    }
  }

  if (infer2d_event_ok) {
    if (cudaEventSynchronize(infer2d_event_stop_) != cudaSuccess) {
      infer2d_event_ok = false;
      AWARN << "[CAM_GPU_PROFILE] cudaEventSynchronize(infer2d stop) failed.";
    }
  }

  if (infer2d_event_ok) {
    if (cudaEventElapsedTime(&infer2d_gpu_ms,
                             infer2d_event_start_,
                             infer2d_event_stop_) != cudaSuccess) {
      infer2d_event_ok = false;
      AWARN << "[CAM_GPU_PROFILE] cudaEventElapsedTime(infer2d) failed.";
    }
  }

  auto infer2d_mem_after = gpu_profile::ReadCudaMemSnapshot();

  const double infer2d_wall_ms =
      static_cast<double>(infer2d_wall_end_ns - infer2d_wall_begin_ns) / 1.0e6;
  const double mem_before_used_mib = infer2d_mem_before.used_mib();
  const double mem_after_used_mib = infer2d_mem_after.used_mib();
  const double mem_delta_used_mib =
      (infer2d_mem_before.ok && infer2d_mem_after.ok)
          ? (mem_after_used_mib - mem_before_used_mib)
          : -1.0;

  if (trace_id != 0) {
    apollo::perf::AppendLatency(
        "perception.camera_detection_multi_stage." + camera_name +
            ".detector.infer2d",
        trace_id, infer2d_wall_begin_ns, infer2d_wall_end_ns);
  }

  AINFO << std::fixed << std::setprecision(3)
        << "[CAM_GPU_PROFILE] "
        << "trace_id=" << trace_id
        << " camera=" << camera_name
        << " stage=detector.infer2d"
        << " warmup=" << (warmup ? 1 : 0)
        << " stream_addr=" << reinterpret_cast<uintptr_t>(infer2d_stream)
        << " stream_is_default=" << (infer2d_stream_is_default ? 1 : 0)
        << " stream_scope=" << infer2d_stream_scope
        << " stream_source=default_fallback"
        << " gpu_event_confidence=" << infer2d_gpu_event_confidence
        << " wall_ms=" << infer2d_wall_ms
        << " gpu_event_ok=" << (infer2d_event_ok ? 1 : 0)
        << " gpu_event_ms=" << infer2d_gpu_ms
        << " wall_minus_gpu_ms="
        << (infer2d_event_ok ? infer2d_wall_ms - infer2d_gpu_ms : -1.0)
        << " mem_before_used_mib=" << mem_before_used_mib
        << " mem_after_used_mib=" << mem_after_used_mib
        << " mem_delta_used_mib=" << mem_delta_used_mib
        << " cuda_last_error=" << cudaGetErrorString(cudaGetLastError());
}
#else
PERF_BLOCK("2d infer time: ")
net_->Infer();
PERF_BLOCK_END
#endif

  // get objects from network inference result
  auto model_outputs = model_param_.info().outputs();

  // Attention(lordon): onnx predict blob is 1, pth is 0
  auto predict_blob = net_->get_blob(model_outputs[1].name());
  frame->feature_blob = net_->get_blob(model_outputs[0].name());

  // PERF_BLOCK("2d postprocess stage: ")
  // YoloxGetObjectsCpu(predict_blob, model_param_, nms_, width_, height_,
  //                    image_width_, image_height_, &frame->detected_objects);
  // PERF_BLOCK_END;
  const std::string postprocess_nvtx_name = gpu_profile::MakeNvtxName(
    "detector.postprocess_cpu", camera_name, trace_id);
{
  gpu_profile::NvtxRangeGuard postprocess_nvtx(postprocess_nvtx_name);

  const uint64_t postprocess_begin_ns = apollo::perf::NowNs();
  PERF_BLOCK("2d postprocess stage: ")
  YoloxGetObjectsCpu(predict_blob, model_param_, nms_, width_, height_,
                     image_width_, image_height_, &frame->detected_objects);
  PERF_BLOCK_END;
  const uint64_t postprocess_end_ns = apollo::perf::NowNs();

  if (trace_id != 0) {
    apollo::perf::AppendLatency(
        "perception.camera_detection_multi_stage." + camera_name +
            ".detector.postprocess_cpu",
        trace_id, postprocess_begin_ns, postprocess_end_ns);
  }

  AINFO << std::fixed << std::setprecision(3)
        << "[CAM_GPU_PROFILE] "
        << "trace_id=" << trace_id
        << " camera=" << camera_name
        << " stage=detector.postprocess_cpu"
        << " warmup=" << (warmup ? 1 : 0)
        << " wall_ms="
        << static_cast<double>(postprocess_end_ns - postprocess_begin_ns) / 1.0e6
        << " detected_object_count=" << frame->detected_objects.size()
        << " note=cpu_postprocess_no_cuda_event";
}

  // post processing
  float border_ratio = 0.01;
  int left_boundary =
      static_cast<int>(border_ratio * static_cast<float>(image.cols()));
  int right_boundary = static_cast<int>((1.0f - border_ratio) *
                                        static_cast<float>(image.cols()));
  // for (auto &obj : frame->detected_objects) {
  //   obj->camera_supplement.area_id = 1;

  //   PERF_BLOCK("3d process of per object: ")
  //   Yolo3DInference(&image, obj);
  //   PERF_BLOCK_END

  //   // clear cut off ratios
  //   auto &box = obj->camera_supplement.box;
  //   if (box.xmin >= left_boundary) {
  //     obj->camera_supplement.cut_off_ratios[2] = 0;
  //   }
  //   if (box.xmax <= right_boundary) {
  //     obj->camera_supplement.cut_off_ratios[3] = 0;
  //   }
  // }
  const uint64_t infer3d_total_begin_ns = apollo::perf::NowNs();
size_t object_idx = 0;

for (auto &obj : frame->detected_objects) {
  obj->camera_supplement.area_id = 1;

#if GPU_PLATFORM == NVIDIA
  cudaStream_t infer3d_stream = static_cast<cudaStream_t>(0);
  const bool infer3d_stream_is_default =
      gpu_profile::IsDefaultStream(infer3d_stream);
  const char* infer3d_stream_scope =
      gpu_profile::StreamScope(infer3d_stream);
  const char* infer3d_gpu_event_confidence =
      gpu_profile::GpuEventConfidence(infer3d_stream, true);

  auto infer3d_mem_before = gpu_profile::ReadCudaMemSnapshot();
  float infer3d_gpu_ms = -1.0f;
  bool infer3d_event_ok = gpu_event_ready_;

  std::ostringstream infer3d_stage_name;
  infer3d_stage_name << "detector.infer3d_per_object.obj_" << object_idx;

  const std::string infer3d_nvtx_name = gpu_profile::MakeNvtxName(
      infer3d_stage_name.str(), camera_name, trace_id);

  {
    gpu_profile::NvtxRangeGuard infer3d_nvtx(infer3d_nvtx_name);

    if (infer3d_event_ok) {
      if (cudaEventRecord(infer3d_event_start_, infer3d_stream) != cudaSuccess) {
        infer3d_event_ok = false;
        AWARN << "[CAM_GPU_PROFILE] cudaEventRecord(infer3d start) failed.";
      }
    }

    const uint64_t infer3d_wall_begin_ns = apollo::perf::NowNs();
    PERF_BLOCK("3d process of per object: ");
    Yolo3DInference(&image, obj);
    PERF_BLOCK_END
    const uint64_t infer3d_wall_end_ns = apollo::perf::NowNs();

    if (infer3d_event_ok) {
      if (cudaEventRecord(infer3d_event_stop_, infer3d_stream) != cudaSuccess) {
        infer3d_event_ok = false;
        AWARN << "[CAM_GPU_PROFILE] cudaEventRecord(infer3d stop) failed.";
      }
    }

    if (infer3d_event_ok) {
      if (cudaEventSynchronize(infer3d_event_stop_) != cudaSuccess) {
        infer3d_event_ok = false;
        AWARN << "[CAM_GPU_PROFILE] cudaEventSynchronize(infer3d stop) failed.";
      }
    }

    if (infer3d_event_ok) {
      if (cudaEventElapsedTime(&infer3d_gpu_ms,
                               infer3d_event_start_,
                               infer3d_event_stop_) != cudaSuccess) {
        infer3d_event_ok = false;
        AWARN << "[CAM_GPU_PROFILE] cudaEventElapsedTime(infer3d) failed.";
      }
    }

    auto infer3d_mem_after = gpu_profile::ReadCudaMemSnapshot();

    const double infer3d_wall_ms =
        static_cast<double>(infer3d_wall_end_ns - infer3d_wall_begin_ns) / 1.0e6;
    const double mem_before_used_mib = infer3d_mem_before.used_mib();
    const double mem_after_used_mib = infer3d_mem_after.used_mib();
    const double mem_delta_used_mib =
        (infer3d_mem_before.ok && infer3d_mem_after.ok)
            ? (mem_after_used_mib - mem_before_used_mib)
            : -1.0;

    // 注意：per-object 不建议 AppendLatency，避免同一个 trace_id 下产生大量重复事件。
    // 这里先通过 AINFO + NVTX 看每个 object 的耗时。
    AINFO << std::fixed << std::setprecision(3)
          << "[CAM_GPU_PROFILE] "
          << "trace_id=" << trace_id
          << " camera=" << camera_name
          << " stage=detector.infer3d_per_object"
          << " object_idx=" << object_idx
          << " warmup=" << (warmup ? 1 : 0)
          << " stream_addr=" << reinterpret_cast<uintptr_t>(infer3d_stream)
          << " stream_is_default=" << (infer3d_stream_is_default ? 1 : 0)
          << " stream_scope=" << infer3d_stream_scope
          << " stream_source=default_fallback"
          << " gpu_event_confidence=" << infer3d_gpu_event_confidence
          << " wall_ms=" << infer3d_wall_ms
          << " gpu_event_ok=" << (infer3d_event_ok ? 1 : 0)
          << " gpu_event_ms=" << infer3d_gpu_ms
          << " wall_minus_gpu_ms="
          << (infer3d_event_ok ? infer3d_wall_ms - infer3d_gpu_ms : -1.0)
          << " mem_before_used_mib=" << mem_before_used_mib
          << " mem_after_used_mib=" << mem_after_used_mib
          << " mem_delta_used_mib=" << mem_delta_used_mib
          << " cuda_last_error=" << cudaGetErrorString(cudaGetLastError());
  }
#else
  PERF_BLOCK("3d process of per object: ")
  Yolo3DInference(&image, obj);
  PERF_BLOCK_END
#endif

  // clear cut off ratios
  auto &box = obj->camera_supplement.box;
  if (box.xmin >= left_boundary) {
    obj->camera_supplement.cut_off_ratios[2] = 0;
  }
  if (box.xmax <= right_boundary) {
    obj->camera_supplement.cut_off_ratios[3] = 0;
  }

  ++object_idx;
}

const uint64_t infer3d_total_end_ns = apollo::perf::NowNs();
if (trace_id != 0) {
  apollo::perf::AppendLatency(
      "perception.camera_detection_multi_stage." + camera_name +
          ".detector.infer3d_total",
      trace_id, infer3d_total_begin_ns, infer3d_total_end_ns);
}

AINFO << std::fixed << std::setprecision(3)
      << "[CAM_GPU_PROFILE] "
      << "trace_id=" << trace_id
      << " camera=" << camera_name
      << " stage=detector.infer3d_total"
      << " warmup=" << (warmup ? 1 : 0)
      << " object_count=" << frame->detected_objects.size()
      << " wall_ms="
      << static_cast<double>(infer3d_total_end_ns - infer3d_total_begin_ns) / 1.0e6;

      const uint64_t detector_total_end_ns = apollo::perf::NowNs();
if (trace_id != 0) {
  apollo::perf::AppendLatency(
      "perception.camera_detection_multi_stage." + camera_name +
          ".detector.inner_total",
      trace_id, detector_total_begin_ns, detector_total_end_ns);
}

AINFO << std::fixed << std::setprecision(3)
      << "[CAM_GPU_PROFILE] "
      << "trace_id=" << trace_id
      << " camera=" << camera_name
      << " stage=detector.inner_total"
      << " warmup=" << (warmup ? 1 : 0)
      << " detected_object_count=" << frame->detected_objects.size()
      << " wall_ms="
      << static_cast<double>(detector_total_end_ns - detector_total_begin_ns) / 1.0e6;

return true;
  return true;
}
REGISTER_OBSTACLE_DETECTOR(Yolox3DObstacleDetector);

}  // namespace camera
}  // namespace perception
}  // namespace apollo
