load("//tools:cpplint.bzl", "cpplint")
load("//tools:apollo_package.bzl", "apollo_cc_library", "apollo_package")

apollo_cc_library(
    name = "latency_recorder",
    srcs = ["latency_recorder.cc"],
    hdrs = ["latency_recorder.h"],
    deps = [
        "//cyber",
        "//modules/common/perf:perf_gflags",
    ],
)

apollo_package()

cpplint()
