/******************************************************************************
 * Copyright 2024 The Apollo Authors. All Rights Reserved.
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

#include "modules/perception/barrier_recognition/barrier_recognition_component.h"

#include "cyber/time/clock.h"
#include "modules/perception/common/lidar/common/lidar_frame_pool.h"
#include "modules/common/perf/latency_trace.h"

namespace apollo {
namespace perception {
namespace lidar {

using apollo::cyber::common::GetAbsolutePath;

const std::string kBarrierId2Type[] = {
    "UNKOWN", "ROD", "FENCE", "ADVERTISING", "TELESCOPIC", "OTHER"
};

bool BarrierRecognitionComponent::Init() {
    BarrierRecognitionComponentConfig barrier_comp_config;
    if (!GetProtoConfig(&barrier_comp_config)) {
        AERROR << "Get BarrierRecognitionComponent file failed";
        return false;
    }
    AINFO << "BarrierRecognitionComponent Component Configs: " << barrier_comp_config.DebugString();

    tf2_frame_id_ = barrier_comp_config.frame_id();
    tf2_child_frame_id_ = barrier_comp_config.child_frame_id();

    auto recognizer_param = barrier_comp_config.recognizer_param();
    BarrierRecognizerInitOptions barrier_bank_init_options;
    barrier_bank_init_options.config_path = recognizer_param.config_path();
    barrier_bank_init_options.config_file = recognizer_param.config_file();
    ACHECK(barrier_bank_.Init(barrier_bank_init_options));

    auto tracker_param = barrier_comp_config.tracker_param();
    StatusTrackerInitOptions status_tracker_init_options;
    status_tracker_init_options.config_path = tracker_param.config_path();
    status_tracker_init_options.config_file = tracker_param.config_file();
    ACHECK(status_tracker_.Init(status_tracker_init_options));
    
    // init hdmaps
    hd_map_ = map::HDMapInput::Instance();
    if (hd_map_ == nullptr) {
        AERROR << "BarrierRecognitionComponent get hdmap failed.";
        return false;
    }

    if (!hd_map_->Init()) {
        AERROR << "BarrierRecognitionComponent init hd-map failed.";
        return false;
    }
    AINFO << "Successfully init barrier recognition component.";

    writer_ = node_->CreateWriter<PerceptionBarrierGate>(
      barrier_comp_config.output_channel_name());

    return true;
}

bool BarrierRecognitionComponent::Proc(const std::shared_ptr<LidarFrameMessage>& message) {
    PERF_FUNCTION()

    // 提取 trace_id
    uint64_t trace_id = 0;
    if (message != nullptr && message->timestamp_ > 0.0) {
        trace_id = static_cast<uint64_t>(message->timestamp_ * 1e9);
    }

    // tick - 帧到达时间点
    const uint64_t t0 = apollo::perf::NowNs();
    apollo::perf::AppendPoint("perception.barrier_recognition.tick",
                              trace_id, t0);

    // internal proc - 内部处理
    const uint64_t i0 = apollo::perf::NowNs();
    std::vector<BarrieGate> gates;
    bool status = InternalProc(message, gates);
    const uint64_t i1 = apollo::perf::NowNs();
    apollo::perf::AppendLatency("perception.barrier_recognition.internal_proc",
                                trace_id, i0, i1);

    if (status) {
        // MakeProtobufMsg - 生成输出消息
        const uint64_t m0 = apollo::perf::NowNs();
        std::shared_ptr<PerceptionBarrierGate> out_message(new (std::nothrow)
                                                           PerceptionBarrierGate);
        if (!MakeProtobufMsg(message->timestamp_, seq_num_, gates,
                            out_message.get())) {
            AERROR << "MakeProtobufMsg failed ts: " << message->timestamp_;
            return false;
        }
        const uint64_t m1 = apollo::perf::NowNs();
        apollo::perf::AppendLatency("perception.barrier_recognition.make_msg",
                                    trace_id, m0, m1);

        // Write - 发送消息
        const uint64_t w0 = apollo::perf::NowNs();
        seq_num_++;
        writer_->Write(out_message);
        const uint64_t w1 = apollo::perf::NowNs();
        apollo::perf::AppendLatency("perception.barrier_recognition.write",
                                    trace_id, w0, w1);
        apollo::perf::AppendLatency("perception.barrier_recognition.total",
                                    trace_id, t0, w1);

        AINFO << "Send peception barrier gate message.";
    }
    return status;
}

bool BarrierRecognitionComponent::InternalProc(
                const std::shared_ptr<LidarFrameMessage>& message, 
                std::vector<BarrieGate>& gates) {
    // 提取 trace_id
    uint64_t trace_id = 0;
    if (message != nullptr && message->timestamp_ > 0.0) {
        trace_id = static_cast<uint64_t>(message->timestamp_ * 1e9);
    }

    const double timestamp = message->timestamp_;
    auto frame = message->lidar_frame_.get();

    // 1. QueryPoseAndGates - 查询位置和屏障门
    const uint64_t q0 = apollo::perf::NowNs();
    std::vector<apollo::hdmap::BarrierGate> hdmap_gates;
    QueryPoseAndGates(timestamp, frame->lidar2world_pose, &hdmap_gates);
    const uint64_t q1 = apollo::perf::NowNs();
    apollo::perf::AppendLatency("perception.barrier_recognition.query_gates",
                                trace_id, q0, q1);

    // 2. BarrierRecognition - 屏障识别
    const uint64_t b0 = apollo::perf::NowNs();
    PERF_BLOCK("barrier_recognition")
    for (auto& hdmap_gate : hdmap_gates) {
        // 2.1 GetGatesPolygon - 获取门的多边形
        const uint64_t p0 = apollo::perf::NowNs();
        std::vector<double> world_polygon;
        GetGatesPolygon(hdmap_gate, world_polygon);
        const uint64_t p1 = apollo::perf::NowNs();
        apollo::perf::AppendLatency("perception.barrier_recognition.get_polygon",
                                    trace_id, p0, p1);

        // 2.2 Recognize - 识别
        const uint64_t r0 = apollo::perf::NowNs();
        BarrierRecognizerOptions detector_options;
        detector_options.name = "StraightPoleRecognizer";
        detector_options.world_roi_polygon = world_polygon;

        float open_percent = 0.;
        if (!barrier_bank_.Recognize(detector_options, frame, open_percent)) {
            AINFO << "barrier recognize error.";
            return false;
        }
        const uint64_t r1 = apollo::perf::NowNs();
        apollo::perf::AppendLatency("perception.barrier_recognition.recognize",
                                    trace_id, r0, r1);

        // 2.3 Track - 跟踪状态
        const uint64_t t0 = apollo::perf::NowNs();
        int status_id = 0;
        if (!status_tracker_.Track(timestamp, open_percent, status_id)) {
            AINFO << "barrier track error.";
            return false;
        }
        const uint64_t t1 = apollo::perf::NowNs();
        apollo::perf::AppendLatency("perception.barrier_recognition.track",
                                    trace_id, t0, t1);

        BarrieGate obj = {status_id, hdmap_gate.id().id(), 
                          kBarrierId2Type[static_cast<int>(hdmap_gate.type())],
                          open_percent};
        gates.emplace_back(obj);
    }
    PERF_BLOCK_END
    const uint64_t b1 = apollo::perf::NowNs();
    apollo::perf::AppendLatency("perception.barrier_recognition.recognition",
                                trace_id, b0, b1);

    return true;
}

bool BarrierRecognitionComponent::QueryPoseAndGates(
              const double ts, const Eigen::Affine3d& pose,
              std::vector<apollo::hdmap::BarrierGate>* gates) {
    if (!hd_map_) {
        AERROR << "hd_map_ not init.";
        return false;
    }

    // get gates
    Eigen::Vector3d global_tranlation = pose.translation();
    if (!hd_map_->GetBarrierGates(global_tranlation, 
                                  forward_distance_to_queries_,
                                  gates)) {
        if (ts - last_gates_ts_ < valid_hdmap_interval_) {
            *gates = last_gates_;
            AWARN << "query pose and gates failed to get gates info. "
                  << "Now use last info. ts:" << ts 
                  << " pose:" << global_tranlation
                  << " gates.size(): " << gates->size();
        } else {
            AERROR << "query pose and gates failed to get gates info. "
                  << "ts:" << ts << " pose:" << global_tranlation;
            return true;
        }
    } else {
        AINFO << "query pose and gates succeeded, gates.size(): "
              << gates->size();
        last_gates_ts_ = ts;
        last_gates_ = *gates;
    }
    return true;
}

void BarrierRecognitionComponent::GetGatesPolygon(
    const apollo::hdmap::BarrierGate& barrier_gate,
    std::vector<double>& world_polygon) {
    world_polygon.clear();
    for (int i = 0; i < barrier_gate.polygon().point_size(); i+=2) {
        world_polygon.push_back(barrier_gate.polygon().point(i).x());
        world_polygon.push_back(barrier_gate.polygon().point(i).y());
    }
}

bool BarrierRecognitionComponent::MakeProtobufMsg(
    double msg_timestamp, int seq_num,
    std::vector<BarrieGate> gates,
    PerceptionBarrierGate *msg) {
    double publish_time = apollo::cyber::Clock::NowInSeconds();
    auto header = msg->mutable_header();
    header->set_timestamp_sec(publish_time);
    header->set_module_name("perception_lidar");
    header->set_sequence_num(seq_num);
    // in nanosecond
    // PnC would use lidar timestamp to predict
    header->set_lidar_timestamp(static_cast<uint64_t>(msg_timestamp * 1e9));
    header->set_camera_timestamp(static_cast<uint64_t>(msg_timestamp * 1e9));

    for (const auto &obj : gates) {
        BarrierGate *gate = msg->add_barrier_gates();
        gate->set_status(static_cast<BarrierGate::Status>(obj.status_id));
        gate->set_id(obj.id);
        gate->set_type(obj.type);
        gate->set_is_useable(true);
        gate->set_open_percent(obj.open_percent);
    }

    return true;
}


}  // namespace lidar
}  // namespace perception
}  // namespace apollo
