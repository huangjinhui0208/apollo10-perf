#pragma once

#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>

#if GPU_PLATFORM == NVIDIA
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#endif

namespace apollo {
namespace perception {
namespace camera {
namespace gpu_profile {

inline double BytesToMiB(size_t bytes) {
  return static_cast<double>(bytes) / 1024.0 / 1024.0;
}

struct CudaMemSnapshot {
  bool ok = false;
  size_t free_bytes = 0;
  size_t total_bytes = 0;

  double used_mib() const {
    if (!ok || total_bytes < free_bytes) {
      return -1.0;
    }
    return BytesToMiB(total_bytes - free_bytes);
  }
};

inline CudaMemSnapshot ReadCudaMemSnapshot() {
  CudaMemSnapshot snap;
#if GPU_PLATFORM == NVIDIA
  if (cudaMemGetInfo(&snap.free_bytes, &snap.total_bytes) == cudaSuccess) {
    snap.ok = true;
  }
#endif
  return snap;
}

#if GPU_PLATFORM == NVIDIA
class NvtxRangeGuard {
 public:
  explicit NvtxRangeGuard(const std::string& msg) {
    nvtxRangePushA(msg.c_str());
  }

  ~NvtxRangeGuard() {
    nvtxRangePop();
  }
};
#else
class NvtxRangeGuard {
 public:
  explicit NvtxRangeGuard(const std::string& msg) {}
};
#endif

#if GPU_PLATFORM == NVIDIA
inline bool IsDefaultStream(cudaStream_t stream) {
  return stream == static_cast<cudaStream_t>(0);
}

inline const char* StreamScope(cudaStream_t stream) {
  return IsDefaultStream(stream) ? "default_stream" : "explicit_stream";
}

inline const char* GpuEventConfidence(cudaStream_t stream,
                                      bool stream_known = true) {
  if (!stream_known) {
    return "low";
  }
  return IsDefaultStream(stream) ? "medium" : "high";
}
#endif

inline std::string MakeNvtxName(const std::string& stage,
                                const std::string& camera_name,
                                uint64_t trace_id) {
  std::ostringstream oss;
  oss << "camera." << stage << "." << camera_name
      << ".trace_" << trace_id;
  return oss.str();
}

}  // namespace gpu_profile
}  // namespace camera
}  // namespace perception
}  // namespace apollo
