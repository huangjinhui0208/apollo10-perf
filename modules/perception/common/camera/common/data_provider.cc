/******************************************************************************
 * Copyright 2018 The Apollo Authors. All Rights Reserved.
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

#include "modules/perception/common/camera/common/data_provider.h"
#include "modules/common/perf/latency_trace.h"
#include "modules/perception/common/camera/common/camera_gpu_profile_utils.h"
#include "cyber/common/log.h"
#include "modules/perception/common/camera/common/image_data_operations.h"

namespace apollo {
namespace perception {
namespace camera {

// 增加资源初始化和释放函数
DataProvider::~DataProvider() {
#if GPU_PLATFORM == NVIDIA
  ReleaseGpuProfileResources();
#endif
}

#if GPU_PLATFORM == NVIDIA
bool DataProvider::InitGpuProfileResources() {
  if (!gpu_profile_enabled_) {
    gpu_event_ready_ = false;
    return false;
  }

  if (copy_event_start_ == nullptr) {
    if (cudaEventCreate(&copy_event_start_) != cudaSuccess) {
      copy_event_start_ = nullptr;
      gpu_event_ready_ = false;
      AWARN << "[CAM_GPU_PROFILE] cudaEventCreate(copy_event_start_) failed.";
      return false;
    }
  }

  if (copy_event_stop_ == nullptr) {
    if (cudaEventCreate(&copy_event_stop_) != cudaSuccess) {
      if (copy_event_start_ != nullptr) {
        cudaEventDestroy(copy_event_start_);
        copy_event_start_ = nullptr;
      }
      copy_event_stop_ = nullptr;
      gpu_event_ready_ = false;
      AWARN << "[CAM_GPU_PROFILE] cudaEventCreate(copy_event_stop_) failed.";
      return false;
    }
  }

  gpu_event_ready_ = true;
  return true;
}

void DataProvider::ReleaseGpuProfileResources() {
  if (copy_event_start_ != nullptr) {
    cudaEventDestroy(copy_event_start_);
    copy_event_start_ = nullptr;
  }

  if (copy_event_stop_ != nullptr) {
    cudaEventDestroy(copy_event_stop_);
    copy_event_stop_ = nullptr;
  }

  gpu_event_ready_ = false;
}
#endif

bool DataProvider::Init(const DataProvider::InitOptions &options) {
  src_height_ = options.image_height;
  src_width_ = options.image_width;
  sensor_name_ = options.sensor_name;
  device_id_ = options.device_id;

  if (cudaSetDevice(device_id_) != cudaSuccess) {
    AERROR << "Failed to set device to: " << device_id_;
    return false;
  }

  // Initialize uint8 blobs
  gray_.reset(new base::Image8U(src_height_, src_width_, base::Color::GRAY));
  rgb_.reset(new base::Image8U(src_height_, src_width_, base::Color::RGB));
  bgr_.reset(new base::Image8U(src_height_, src_width_, base::Color::BGR));

  // Allocate CPU memory for uint8 blobs
  gray_->cpu_data();
  rgb_->cpu_data();
  bgr_->cpu_data();

  // Allocate GPU memory for uint8 blobs
  gray_->gpu_data();
  rgb_->gpu_data();
  bgr_->gpu_data();

  if (options.do_undistortion) {
    handler_.reset(new UndistortionHandler());
    if (!handler_->Init(options.sensor_name, device_id_)) {
      return false;
    }
    // Initialize uint8 blobs
    ori_gray_.reset(
        new base::Image8U(src_height_, src_width_, base::Color::GRAY));
    ori_rgb_.reset(
        new base::Image8U(src_height_, src_width_, base::Color::RGB));
    ori_bgr_.reset(
        new base::Image8U(src_height_, src_width_, base::Color::BGR));

    // Allocate CPU memory for uint8 blobs
    ori_gray_->cpu_data();
    ori_rgb_->cpu_data();
    ori_bgr_->cpu_data();

    // Allocate GPU memory for uint8 blobs
    ori_gray_->gpu_data();
    ori_rgb_->gpu_data();
    ori_bgr_->gpu_data();
  }

  // Warm up nppi functions
  {
    bgr_ready_ = false;
    rgb_ready_ = true;
    gray_ready_ = false;
    to_bgr_image();
    bgr_ready_ = false;
    rgb_ready_ = false;
    gray_ready_ = true;
    to_bgr_image();
  }
  {
    bgr_ready_ = false;
    rgb_ready_ = false;
    gray_ready_ = true;
    to_rgb_image();
    bgr_ready_ = true;
    rgb_ready_ = false;
    gray_ready_ = false;
    to_rgb_image();
  }
  {
    bgr_ready_ = false;
    rgb_ready_ = true;
    gray_ready_ = false;
    to_gray_image();
    bgr_ready_ = true;
    rgb_ready_ = false;
    gray_ready_ = false;
    to_gray_image();
  }
  bgr_ready_ = false;
  rgb_ready_ = false;
  gray_ready_ = false;

  // 初始化event
  #if GPU_PLATFORM == NVIDIA
  InitGpuProfileResources();
  #endif
  return true;
}

// bool DataProvider::FillImageData(int rows, int cols, const uint8_t *data,
//                                  const std::string &encoding) {
//   if (cudaSetDevice(device_id_) != cudaSuccess) {
//     AERROR << "Failed to set device to: " << device_id_;
//     return false;
//   }

//   gray_ready_ = false;
//   rgb_ready_ = false;
//   bgr_ready_ = false;

//   bool success = false;

// #if USE_GPU == 0  // copy to host memory
//   AINFO << "Fill in CPU mode ...";
//   if (handler_ != nullptr) {
//     AERROR << "Undistortion DO NOT support CPU mode!";
//     return false;
//   }
//   if (encoding == "rgb8") {
//     memcpy(rgb_->mutable_cpu_data(), data, rgb_->count() * sizeof(data[0]));
//     rgb_ready_ = true;
//     success = true;
//   } else if (encoding == "bgr8") {
//     memcpy(bgr_->mutable_cpu_data(), data, bgr_->count() * sizeof(data[0]));
//     bgr_ready_ = true;
//     success = true;
//   } else if (encoding == "gray" || encoding == "y") {
//     memcpy(gray_->mutable_cpu_data(), data, gray_->count() * sizeof(data[0]));
//     gray_ready_ = true;
//     success = true;
//   } else {
//     AERROR << "Unrecognized image encoding: " << encoding;
//   }
// #else  // copy to device memory directly
//   AINFO << "Fill in GPU mode ...";
//   if (encoding == "rgb8") {
//     if (handler_ != nullptr) {
//       cudaMemcpy(ori_rgb_->mutable_gpu_data(), data,
//                  ori_rgb_->rows() * ori_rgb_->width_step(), cudaMemcpyDefault);
//       success = handler_->Handle(*ori_rgb_, rgb_.get());
//     } else {
//       cudaMemcpy(rgb_->mutable_gpu_data(), data,
//                  rgb_->rows() * rgb_->width_step(), cudaMemcpyDefault);
//       success = true;
//     }
//     rgb_ready_ = true;
//   } else if (encoding == "bgr8") {
//     if (handler_ != nullptr) {
//       cudaMemcpy(ori_bgr_->mutable_gpu_data(), data,
//                  ori_bgr_->rows() * ori_bgr_->width_step(), cudaMemcpyDefault);
//       success = handler_->Handle(*ori_bgr_, bgr_.get());
//     } else {
//       cudaMemcpy(bgr_->mutable_gpu_data(), data,
//                  bgr_->rows() * bgr_->width_step(), cudaMemcpyDefault);
//       success = true;
//     }
//     bgr_ready_ = true;
//   } else if (encoding == "gray" || encoding == "y") {
//     if (handler_ != nullptr) {
//       cudaMemcpy(ori_gray_->mutable_gpu_data(), data,
//                  ori_gray_->rows() * ori_gray_->width_step(),
//                  cudaMemcpyDefault);
//       success = handler_->Handle(*ori_gray_, gray_.get());
//     } else {
//       cudaMemcpy(gray_->mutable_gpu_data(), data,
//                  gray_->rows() * gray_->width_step(), cudaMemcpyDefault);
//       success = true;
//     }
//     gray_ready_ = true;
//   } else {
//     AERROR << "Unrecognized image encoding: " << encoding;
//   }
// #endif

//   AINFO << "Done! (" << success << ")";
//   return success;
// }
bool DataProvider::FillImageData(int rows, int cols,
                                 const uint8_t *data,
                                 const std::string &encoding) {
  return FillImageData(rows, cols, data, encoding, "", 0);
}
bool DataProvider::FillImageData(int rows, int cols,
                                 const uint8_t *data,
                                 const std::string &encoding,
                                 const std::string &perf_prefix,
                                 uint64_t trace_id) 
  {
  const bool warmup = fill_call_idx_ < gpu_profile_warmup_frames_;
  ++fill_call_idx_;
  const bool enable_perf = (!perf_prefix.empty() && trace_id != 0);
  const uint64_t total0 = apollo::perf::NowNs();
  
  auto record = [&](const std::string& name,
                    uint64_t begin_ns,
                    uint64_t end_ns) {
    if (enable_perf) {
      apollo::perf::AppendLatency(perf_prefix + "." + name,
                                  trace_id, begin_ns, end_ns);
    }
  };

  const uint64_t set0 = apollo::perf::NowNs();
  if (cudaSetDevice(device_id_) != cudaSuccess) {
    const uint64_t set1 = apollo::perf::NowNs();
    record("fill_image.set_device", set0, set1);
    AERROR << "Failed to set device to: " << device_id_;
    return false;
  }
  const uint64_t set1 = apollo::perf::NowNs();
  record("fill_image.set_device", set0, set1);

  const uint64_t reset0 = apollo::perf::NowNs();
  gray_ready_ = false;
  rgb_ready_ = false;
  bgr_ready_ = false;
  const uint64_t reset1 = apollo::perf::NowNs();
  record("fill_image.reset_flags", reset0, reset1);

  bool success = false;

#if USE_GPU == 0
  const uint64_t copy0 = apollo::perf::NowNs();

  if (handler_ != nullptr) {
    AERROR << "Undistortion DO NOT support CPU mode!";
    return false;
  }

  if (encoding == "rgb8") {
    memcpy(rgb_->mutable_cpu_data(), data, rgb_->count() * sizeof(data[0]));
    rgb_ready_ = true;
    success = true;
  } else if (encoding == "bgr8") {
    memcpy(bgr_->mutable_cpu_data(), data, bgr_->count() * sizeof(data[0]));
    bgr_ready_ = true;
    success = true;
  } else if (encoding == "gray" || encoding == "y") {
    memcpy(gray_->mutable_cpu_data(), data, gray_->count() * sizeof(data[0]));
    gray_ready_ = true;
    success = true;
  } else {
    AERROR << "Unrecognized image encoding: " << encoding;
  }

  const uint64_t copy1 = apollo::perf::NowNs();
  record("fill_image.copy_cpu." + encoding, copy0, copy1);

#else
  if (encoding == "rgb8") {
    if (handler_ != nullptr) {
      // 如果你当前 enable_undistortion=false，这个分支一般不会走。
      cudaMemcpy(ori_rgb_->mutable_gpu_data(), data,
                 ori_rgb_->rows() * ori_rgb_->width_step(),
                 cudaMemcpyDefault);
      success = handler_->Handle(*ori_rgb_, rgb_.get());
      rgb_ready_ = true;
    } else {
      #if GPU_PLATFORM == NVIDIA
      cudaStream_t copy_stream = static_cast<cudaStream_t>(0);
      const bool stream_is_default =
          gpu_profile::IsDefaultStream(copy_stream);
      const char* stream_scope = gpu_profile::StreamScope(copy_stream);
      const char* gpu_event_confidence =
          gpu_profile::GpuEventConfidence(copy_stream, true);

      const size_t copy_bytes = rgb_->rows() * rgb_->width_step();

      auto mem_before = gpu_profile::ReadCudaMemSnapshot();

      float copy_gpu_ms = -1.0f;
      bool copy_event_ok = gpu_event_ready_;

      const std::string nvtx_name = gpu_profile::MakeNvtxName(
          "fill_image.copy_gpu.rgb8", sensor_name_, trace_id);

      {
        gpu_profile::NvtxRangeGuard nvtx_range(nvtx_name);

        if (copy_event_ok) {
          if (cudaEventRecord(copy_event_start_, copy_stream) != cudaSuccess) {
            copy_event_ok = false;
            AWARN << "[CAM_GPU_PROFILE] cudaEventRecord(copy start) failed.";
          }
        }

        const uint64_t copy_wall_begin_ns = apollo::perf::NowNs();

        cudaError_t copy_ret = cudaMemcpy(
            rgb_->mutable_gpu_data(),
            data,
            copy_bytes,
            cudaMemcpyDefault);

        const uint64_t copy_wall_end_ns = apollo::perf::NowNs();

        if (copy_event_ok) {
          if (cudaEventRecord(copy_event_stop_, copy_stream) != cudaSuccess) {
            copy_event_ok = false;
            AWARN << "[CAM_GPU_PROFILE] cudaEventRecord(copy stop) failed.";
          }
        }

        if (copy_event_ok) {
          if (cudaEventSynchronize(copy_event_stop_) != cudaSuccess) {
            copy_event_ok = false;
            AWARN << "[CAM_GPU_PROFILE] cudaEventSynchronize(copy stop) failed.";
          }
        }

        if (copy_event_ok) {
          if (cudaEventElapsedTime(&copy_gpu_ms,
                                   copy_event_start_,
                                   copy_event_stop_) != cudaSuccess) {
            copy_event_ok = false;
            AWARN << "[CAM_GPU_PROFILE] cudaEventElapsedTime(copy) failed.";
          }
        }

        auto mem_after = gpu_profile::ReadCudaMemSnapshot();

        const double copy_wall_ms =
            static_cast<double>(copy_wall_end_ns - copy_wall_begin_ns) / 1.0e6;

        const double mem_before_used_mib = mem_before.used_mib();
        const double mem_after_used_mib = mem_after.used_mib();
        const double mem_delta_used_mib =
            (mem_before.ok && mem_after.ok)
                ? (mem_after_used_mib - mem_before_used_mib)
                : -1.0;

        if (trace_id != 0) {
          apollo::perf::AppendLatency(
              "perception.camera_detection_multi_stage." + sensor_name_ +
                  ".fill_image.copy_gpu.rgb8",
              trace_id, copy_wall_begin_ns, copy_wall_end_ns);
        }

        AINFO << std::fixed << std::setprecision(3)
              << "[CAM_GPU_PROFILE] "
              << "trace_id=" << trace_id
              << " camera=" << sensor_name_
              << " stage=fill_image.copy_gpu.rgb8"
              << " warmup=" << (warmup ? 1 : 0)
              << " encoding=" << encoding
              << " rows=" << rows
              << " cols=" << cols
              << " width_step=" << rgb_->width_step()
              << " copy_bytes=" << copy_bytes
              << " stream_addr=0"
              << " stream_is_default=" << (stream_is_default ? 1 : 0)
              << " stream_scope=" << stream_scope
              << " stream_source=default_cudaMemcpy"
              << " gpu_event_confidence=" << gpu_event_confidence
              << " wall_ms=" << copy_wall_ms
              << " gpu_event_ok=" << (copy_event_ok ? 1 : 0)
              << " gpu_event_ms=" << copy_gpu_ms
              << " wall_minus_gpu_ms="
              << (copy_event_ok ? copy_wall_ms - copy_gpu_ms : -1.0)
              << " mem_before_used_mib=" << mem_before_used_mib
              << " mem_after_used_mib=" << mem_after_used_mib
              << " mem_delta_used_mib=" << mem_delta_used_mib
              << " cuda_ret=" << cudaGetErrorString(copy_ret)
              << " cuda_last_error="
              << cudaGetErrorString(cudaGetLastError());

        success = (copy_ret == cudaSuccess);
      }
    #else
      cudaMemcpy(rgb_->mutable_gpu_data(), data,
                 rgb_->rows() * rgb_->width_step(), cudaMemcpyDefault);
      success = true;
    #endif
      rgb_ready_ = true;
    }
  }else if (encoding == "bgr8") {
    if (handler_ != nullptr) {
      const uint64_t copy0 = apollo::perf::NowNs();
      cudaMemcpy(ori_bgr_->mutable_gpu_data(), data,
                 ori_bgr_->rows() * ori_bgr_->width_step(),
                 cudaMemcpyDefault);
      const uint64_t copy1 = apollo::perf::NowNs();
      record("fill_image.copy_gpu.bgr8", copy0, copy1);

      const uint64_t und0 = apollo::perf::NowNs();
      success = handler_->Handle(*ori_bgr_, bgr_.get());
      const uint64_t und1 = apollo::perf::NowNs();
      record("fill_image.undistort.bgr8", und0, und1);
    } else {
      const uint64_t copy0 = apollo::perf::NowNs();
      cudaMemcpy(bgr_->mutable_gpu_data(), data,
                 bgr_->rows() * bgr_->width_step(),
                 cudaMemcpyDefault);
      const uint64_t copy1 = apollo::perf::NowNs();
      record("fill_image.copy_gpu.bgr8", copy0, copy1);
      success = true;
    }
    bgr_ready_ = true;

  } else if (encoding == "gray" || encoding == "y") {
    if (handler_ != nullptr) {
      const uint64_t copy0 = apollo::perf::NowNs();
      cudaMemcpy(ori_gray_->mutable_gpu_data(), data,
                 ori_gray_->rows() * ori_gray_->width_step(),
                 cudaMemcpyDefault);
      const uint64_t copy1 = apollo::perf::NowNs();
      record("fill_image.copy_gpu.gray", copy0, copy1);

      const uint64_t und0 = apollo::perf::NowNs();
      success = handler_->Handle(*ori_gray_, gray_.get());
      const uint64_t und1 = apollo::perf::NowNs();
      record("fill_image.undistort.gray", und0, und1);
    } else {
      const uint64_t copy0 = apollo::perf::NowNs();
      cudaMemcpy(gray_->mutable_gpu_data(), data,
                 gray_->rows() * gray_->width_step(),
                 cudaMemcpyDefault);
      const uint64_t copy1 = apollo::perf::NowNs();
      record("fill_image.copy_gpu.gray", copy0, copy1);
      success = true;
    }
    gray_ready_ = true;

  } else {
    AERROR << "Unrecognized image encoding: " << encoding;
  }
#endif

  const uint64_t total1 = apollo::perf::NowNs();
  record("fill_image.total", total0, total1);

  return success;
}


#if 0
bool DataProvider::GetImageBlob(const DataProvider::ImageOptions &options,
                                base::Blob<float> *blob) {
  bool ret = GetImageBlob(options, &temp_uint8_);
  if (!ret) {
    return false;
  }
  blob->Reshape(temp_uint8_.shape());
  NppiSize roi;
  roi.height = temp_uint8_.shape(1);
  roi.width = temp_uint8_.shape(2);
  int channels = base::kChannelsMap.at(options.target_color);
  const uint8_t *temp_ptr = temp_uint8_.gpu_data();
  float *blob_ptr = blob->mutable_gpu_data();
  int temp_step = temp_uint8_.count(2) * sizeof(uint8_t);
  int blob_step = blob->count(2) * sizeof(float);
#if GPU_PLATFORM == NVIDIA
  if (channels == 1) {
    nppiConvert_8u32f_C1R(temp_ptr, temp_step, blob_ptr, blob_step, roi);
  } else {
    nppiConvert_8u32f_C3R(temp_ptr, temp_step, blob_ptr, blob_step, roi);
  }
#elif GPU_PLATFORM == AMD
    // TODO(B1tway): Add necesssary RPP API
#endif
  return true;
}
#endif

bool DataProvider::GetImageBlob(const DataProvider::ImageOptions &options,
                                base::Blob<uint8_t> *blob) {
  base::Image8U image;
  if (!GetImage(options, &image)) {
    return false;
  }
  imageToBlob(image, blob);
  return true;
}

bool DataProvider::GetImage(const DataProvider::ImageOptions &options,
                            base::Image8U *image) {
  AINFO << "GetImage ...";
  if (image == nullptr) {
    return false;
  }
  bool success = false;
  switch (options.target_color) {
    case base::Color::RGB:
      success = to_rgb_image();
      *image = (*rgb_);
      break;
    case base::Color::BGR:
      success = to_bgr_image();
      *image = (*bgr_);
      break;
    case base::Color::GRAY:
      success = to_gray_image();
      *image = (*gray_);
      break;
    default:
      AERROR << "Unsupported Color: "
             << static_cast<uint8_t>(options.target_color);
  }
  if (!success) {
    return false;
  }

  if (options.do_crop) {
    AINFO << "\tcropping ...";
    *image = (*image)(options.crop_roi);
  }
  AINFO << "Done!";
  return true;
}

bool DataProvider::to_gray_image() {
  if (!gray_ready_) {
    if (bgr_ready_) {
      float coeffs[] = {0.114f, 0.587f, 0.299f};
      imageToGray(bgr_, gray_, src_width_, src_height_, coeffs);
      gray_ready_ = true;
    } else if (rgb_ready_) {
      float coeffs[] = {0.299f, 0.587f, 0.114f};
      imageToGray(rgb_, gray_, src_width_, src_height_, coeffs);
      gray_ready_ = true;
    } else {
      AWARN << "No image data filled yet, return uninitialized blob!";
      return false;
    }
  }
  return true;
}

bool DataProvider::to_rgb_image() {
  if (!rgb_ready_) {
    if (bgr_ready_) {
      // BGR2RGB takes less than 0.010ms on K2200
      const int order[] = {2, 1, 0};
      swapImageChannels(bgr_, rgb_, src_width_, src_height_, order);
      rgb_ready_ = true;
    } else if (gray_ready_) {
      dupImageChannels(gray_, rgb_, src_width_, src_height_);
      rgb_ready_ = true;
    } else {
      AWARN << "No image data filled yet, return uninitialized blob!";
      return false;
    }
  }
  return true;
}

bool DataProvider::to_bgr_image() {
  if (!bgr_ready_) {
    if (rgb_ready_) {
      const int order[] = {2, 1, 0};
      swapImageChannels(rgb_, bgr_, src_width_, src_height_, order);
      bgr_ready_ = true;
    } else if (gray_ready_) {
      dupImageChannels(gray_, bgr_, src_width_, src_height_);
      bgr_ready_ = true;
    } else {
      AWARN << "No image data filled yet, return uninitialized blob!";
      return false;
    }
  }
  return true;
}

}  // namespace camera
}  // namespace perception
}  // namespace apollo
