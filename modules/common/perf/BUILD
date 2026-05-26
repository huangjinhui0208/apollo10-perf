load("//tools:apollo_package.bzl", "apollo_cc_library", "apollo_package")

package(default_visibility = ["//visibility:public"])

apollo_cc_library(
    name = "perf_gflags",
    srcs = ["perf_gflags.cc"],
    hdrs = ["perf_gflags.h"],
    deps = [
        "@com_github_gflags_gflags//:gflags",
    ],
)

apollo_cc_library(
    name = "latency_trace",
    srcs = ["latency_trace.cc"],
    hdrs = ["latency_trace.h"],
    deps = [
        "//cyber",
        "//modules/common/latency_recorder",
        "//modules/common/perf:perf_gflags",
        "//modules/common/adapters:adapter_gflags",
        "//modules/common_msgs/basic_msgs:header_cc_proto",
    ],
)

apollo_package()
