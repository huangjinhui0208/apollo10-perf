#pragma once

#if GPU_PLATFORM == NVIDIA
#include <cupti.h>
#endif

#include <atomic>
#include <fstream>
#include <mutex>
#include <string>

namespace apollo {
namespace perception {
namespace profiler {

class CuptiActivityCollector {
 public:
  static CuptiActivityCollector& Instance();

  bool Init(const std::string& out_path);
  void Shutdown();

  bool Enable();
  void Disable();

  bool enabled() const { return enabled_.load(); }

 private:
  CuptiActivityCollector() = default;
  ~CuptiActivityCollector() = default;

#if GPU_PLATFORM == NVIDIA
  static void CUPTIAPI BufferRequested(uint8_t** buffer,
                                       size_t* size,
                                       size_t* maxNumRecords);
  static void CUPTIAPI BufferCompleted(CUcontext ctx,
                                       uint32_t streamId,
                                       uint8_t* buffer,
                                       size_t size,
                                       size_t validSize);
#endif

  void HandleActivityBuffer(uint8_t* buffer, size_t validSize);

 private:
  std::atomic<bool> enabled_{false};
  std::atomic<bool> initialized_{false};
  std::mutex mu_;
  std::ofstream ofs_;
};

}  // namespace profiler
}  // namespace perception
}  // namespace apollo