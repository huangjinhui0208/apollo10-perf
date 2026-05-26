#include "modules/perception/common/profiler/cupti_activity_collector.h"

#if GPU_PLATFORM == NVIDIA
#include <cuda.h>
#include <cupti.h>
#endif

#include <cstdlib>
#include <cstring>
#include <iostream>

namespace apollo {
namespace perception {
namespace profiler {

namespace {
constexpr size_t kBufSize = 4 * 1024 * 1024;
constexpr size_t kAlignSize = 8;

inline uint8_t* AlignedMalloc(size_t size, size_t align) {
  void* ptr = nullptr;
  if (posix_memalign(&ptr, align, size) != 0) {
    return nullptr;
  }
  return reinterpret_cast<uint8_t*>(ptr);
}
}  // namespace

CuptiActivityCollector& CuptiActivityCollector::Instance() {
  static CuptiActivityCollector inst;
  return inst;
}

bool CuptiActivityCollector::Init(const std::string& out_path) {
#if GPU_PLATFORM != NVIDIA
  return false;
#else
  std::lock_guard<std::mutex> lk(mu_);
  if (initialized_.load()) {
    return true;
  }
  ofs_.open(out_path, std::ios::out | std::ios::trunc);
  if (!ofs_.is_open()) {
    return false;
  }

  ofs_ << "kind,start_ns,end_ns,device_id,context_id,stream_id,bytes,name,extra\n";

  CUptiResult r = cuptiActivityRegisterCallbacks(BufferRequested,
                                                 BufferCompleted);
  if (r != CUPTI_SUCCESS) {
    return false;
  }
  initialized_.store(true);
  return true;
#endif
}

bool CuptiActivityCollector::Enable() {
#if GPU_PLATFORM != NVIDIA
  return false;
#else
  if (!initialized_.load()) {
    return false;
  }
  cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL);
  cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
  cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY);
  cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMORY2);
  cuptiActivityEnableAllocationSource(1);
  enabled_.store(true);
  return true;
#endif
}

void CuptiActivityCollector::Disable() {
#if GPU_PLATFORM == NVIDIA
  enabled_.store(false);
  cuptiActivityDisable(CUPTI_ACTIVITY_KIND_KERNEL);
  cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
  cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY);
  cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMORY2);
  cuptiActivityFlushAll(0);
#endif
}

void CuptiActivityCollector::Shutdown() {
#if GPU_PLATFORM == NVIDIA
  Disable();
#endif
  std::lock_guard<std::mutex> lk(mu_);
  if (ofs_.is_open()) {
    ofs_.flush();
    ofs_.close();
  }
  initialized_.store(false);
}

#if GPU_PLATFORM == NVIDIA
void CUPTIAPI CuptiActivityCollector::BufferRequested(
    uint8_t** buffer, size_t* size, size_t* maxNumRecords) {
  *size = kBufSize;
  *buffer = AlignedMalloc(*size, kAlignSize);
  *maxNumRecords = 0;
}

void CUPTIAPI CuptiActivityCollector::BufferCompleted(
    CUcontext ctx, uint32_t streamId, uint8_t* buffer, size_t size,
    size_t validSize) {
  CuptiActivityCollector::Instance().HandleActivityBuffer(buffer, validSize);
  free(buffer);
}
#endif

void CuptiActivityCollector::HandleActivityBuffer(uint8_t* buffer,
                                                  size_t validSize) {
#if GPU_PLATFORM != NVIDIA
  return;
#else
  CUpti_Activity* record = nullptr;
  do {
    CUptiResult status = cuptiActivityGetNextRecord(buffer, validSize, &record);
    if (status == CUPTI_SUCCESS) {
      std::lock_guard<std::mutex> lk(mu_);
      switch (record->kind) {
        case CUPTI_ACTIVITY_KIND_KERNEL:
        case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
          auto* k = reinterpret_cast<CUpti_ActivityKernel9*>(record);
          ofs_ << "kernel," << k->start << "," << k->end << ","
               << k->deviceId << "," << k->contextId << ","
               << k->streamId << ",0,\""
               << (k->name ? k->name : "") << "\",\"grid="
               << k->gridX << "x" << k->gridY << "x" << k->gridZ
               << ";block=" << k->blockX << "x" << k->blockY << "x" << k->blockZ
               << "\"\n";
          break;
        }
        case CUPTI_ACTIVITY_KIND_MEMCPY: {
          auto* m = reinterpret_cast<CUpti_ActivityMemcpy*>(record);
          ofs_ << "memcpy," << m->start << "," << m->end << ","
               << m->deviceId << "," << m->contextId << ","
               << m->streamId << "," << m->bytes << ",\"\",\"copyKind="
               << m->copyKind << "\"\n";
          break;
        }
        case CUPTI_ACTIVITY_KIND_MEMORY2: {
          auto* mm = reinterpret_cast<CUpti_ActivityMemory2*>(record);
          ofs_ << "memory2," << mm->start << "," << mm->end << ","
               << mm->deviceId << "," << mm->contextId << ","
               << mm->streamId << "," << mm->bytes << ",\"\",\"memoryKind="
               << mm->memoryKind << ";address=" << mm->address << "\"\n";
          break;
        }
        default:
          break;
      }
    } else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) {
      break;
    } else {
      break;
    }
  } while (true);
#endif
}

}  // namespace profiler
}  // namespace perception
}  // namespace apollo