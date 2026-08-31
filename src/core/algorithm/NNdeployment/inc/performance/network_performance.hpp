#pragma once

#include <cstdint>
#include <string>

namespace MPT
{
// OpenVINO 官方 benchmark_app 的运行参数，与部署库端到端计时相互独立。
struct OfficialBenchmarkConfig
{
    std::string benchmark_app_path = "benchmark_app"; // benchmark_app 可执行文件路径。
    std::string perf_hint = "latency";                 // OpenVINO 的 latency/throughput 提示。
    int time_seconds = 10;                             // iterations=0 时按持续时间测试。
    int iterations = 0;                               // 大于 0 时按迭代次数测试。
    bool inference_only = true;                        // 是否只统计推理阶段。
    bool no_warmup = false;                            // 是否关闭 benchmark_app 自带预热。
};

// 累积部署库各阶段耗时，所有输入与内部统计单位均为微秒。
struct SpeedStats
{
    // 累加一次各阶段耗时。
    void update(double preprocess_us, double inference_us, double postprocess_us);
    // 输出当前累计耗时统计。
    void printCurrentStats() const;

private:
    double m_total_preprocess_us = 0.0;
    double m_total_inference_us = 0.0;
    double m_total_postprocess_us = 0.0;
    double m_max_inference_us = 0.0;
    double m_min_inference_us = 0.0;
    std::uint64_t m_call_count = 0;
};

// 运行 OpenVINO benchmark_app 并记录统一格式的性能报告。
void runOfficialBenchmark(const std::string &model_path,
                          const std::string &device,
                          const std::string &infer_mode,
                          const OfficialBenchmarkConfig &config = {});
} // namespace MPT
