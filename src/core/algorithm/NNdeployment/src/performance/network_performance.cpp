#include "network_performance.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <unistd.h>

// 写给 rp 网络组的同学：performance 相关内容（MPT部分）不需要阅读，该部分代码大部分是统计换算和文本解析且非常繁琐，该部分是26赛季全 AI 生成的，后续如果需要改动该部分内容，可以直接 AI 重写，不要花精力在具体代码上

// 性能支持模块：部署库端到端分阶段统计，以及独立的 benchmark_app 官方基准。
// 端到端推理更符合上车实际性能，benchmark 统计的是在GPU被不断送入图片，保持最高并发状态下，单 GPU 推理的耗时性能，实际运行无法达到该速度
// 两类结果的统计范围不同，调用方应分别记录，不直接横向比较。

namespace MPT
{
namespace detail
{
struct EndToEndResult
{
    std::uint64_t calls = 0;
    double average_preprocess_us = 0.0;
    double average_inference_us = 0.0;
    double average_postprocess_us = 0.0;
    double average_total_us = 0.0;
    double minimum_inference_us = 0.0;
    double maximum_inference_us = 0.0;
    double average_fps = 0.0;
};

std::mutex end_to_end_mutex;
EndToEndResult latest_end_to_end_result;
} // namespace detail

// 累加一次预处理、推理和后处理耗时。
void SpeedStats::update(double preprocess_us, double inference_us, double postprocess_us)
{
    m_total_preprocess_us += preprocess_us;
    m_total_inference_us += inference_us;
    m_total_postprocess_us += postprocess_us;
    m_max_inference_us = std::max(m_max_inference_us, inference_us);
    m_min_inference_us = m_call_count == 0 ? inference_us : std::min(m_min_inference_us, inference_us);
    ++m_call_count;
}

// 计算并输出当前累计耗时统计。
void SpeedStats::printCurrentStats() const
{
    const double count = static_cast<double>(m_call_count);
    const double average_preprocess_us = m_call_count > 0 ? m_total_preprocess_us / count : 0.0;
    const double average_inference_us = m_call_count > 0 ? m_total_inference_us / count : 0.0;
    const double average_postprocess_us = m_call_count > 0 ? m_total_postprocess_us / count : 0.0;
    const double average_total_us = average_preprocess_us + average_inference_us + average_postprocess_us;
    const double average_fps = average_total_us > 0.0 ? 1'000'000.0 / average_total_us : 0.0;

    {
        std::lock_guard<std::mutex> lock(detail::end_to_end_mutex);
        detail::latest_end_to_end_result = {m_call_count,
                                            average_preprocess_us,
                                            average_inference_us,
                                            average_postprocess_us,
                                            average_total_us,
                                            m_min_inference_us,
                                            m_max_inference_us,
                                            average_fps};
    }

    std::cout << std::fixed << std::setprecision(2)
              << "次数:" << m_call_count
              << " 平均预处理时间:" << average_preprocess_us << " μs"
              << " | 平均推理时间:" << average_inference_us << " μs"
              << " | 平均后处理时间:" << average_postprocess_us << " μs\n"
              << "平均总时间:" << average_total_us << " μs"
              << " | 最大推理时间: " << m_max_inference_us << " μs"
              << " | 最小推理时间:" << m_min_inference_us << " μs"
              << " | 平均FPS:" << average_fps << '\n';
    std::cout.flush();
}

// ==================== MPT: Model Performance Testing ====================
// 这一段负责测速日志写入、benchmark_app 命令构造和官方输出解析。
namespace detail
{
struct OfficialBenchmarkResult
{
    int iterations = 0;
    double duration_ms = 0.0;
    double latency_median_ms = 0.0;
    double latency_average_ms = 0.0;
    double latency_min_ms = 0.0;
    double latency_max_ms = 0.0;
    double throughput_fps = 0.0;
};

// 将浮点数格式化为保留两位小数的字符串。
std::string fixedValue(double value)
{
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(2) << value;
    return stream.str();
}

// 向性能报告追加一行指标数据。
void appendMetricRow(std::ostringstream &stream,
                     const std::string &scope,
                     const std::string &metric,
                     const std::string &value,
                     const std::string &unit)
{
    stream << "| " << scope
           << " | " << metric
           << " | " << value
           << " | " << unit << " |\n";
}

// 汇总端到端统计和 benchmark_app 结果并生成报告。
std::string formatPerformanceReport(const std::string &model_path,
                                    const std::string &device,
                                    const std::string &infer_mode,
                                    const OfficialBenchmarkConfig &config,
                                    const EndToEndResult &end_to_end,
                                    const OfficialBenchmarkResult *benchmark,
                                    const std::string &benchmark_error = {})
{
    const std::filesystem::path model(model_path);
    std::ostringstream stream;
    stream << "\n## Model Performance Report\n\n"
           << "- Model: " << model.filename().string() << '\n'
           << "- Model path: " << model_path << '\n'
           << "- Device: " << device << '\n'
           << "- Inference mode: " << infer_mode << '\n'
           << "- Benchmark performance hint: " << config.perf_hint << "\n\n"
           << "| Test scope | Metric | Value | Unit |\n"
           << "| --- | --- | ---: | --- |\n";

    if (end_to_end.calls > 0)
    {
        appendMetricRow(stream, "End-to-end", "Calls",
                        std::to_string(end_to_end.calls), "calls");
        appendMetricRow(stream, "End-to-end", "Average preprocess time",
                        fixedValue(end_to_end.average_preprocess_us), "us");
        appendMetricRow(stream, "End-to-end", "Average inference time",
                        fixedValue(end_to_end.average_inference_us), "us");
        appendMetricRow(stream, "End-to-end", "Average postprocess time",
                        fixedValue(end_to_end.average_postprocess_us), "us");
        appendMetricRow(stream, "End-to-end", "Average total time",
                        fixedValue(end_to_end.average_total_us), "us");
        appendMetricRow(stream, "End-to-end", "Minimum inference time",
                        fixedValue(end_to_end.minimum_inference_us), "us");
        appendMetricRow(stream, "End-to-end", "Maximum inference time",
                        fixedValue(end_to_end.maximum_inference_us), "us");
        appendMetricRow(stream, "End-to-end", "Average FPS",
                        fixedValue(end_to_end.average_fps), "FPS");
    }
    else
    {
        appendMetricRow(stream, "End-to-end", "Status",
                        "No timing data", "-");
    }

    if (benchmark != nullptr)
    {
        appendMetricRow(stream, "BenchmarkApp", "Iterations",
                        std::to_string(benchmark->iterations), "iterations");
        appendMetricRow(stream, "BenchmarkApp", "Duration",
                        fixedValue(benchmark->duration_ms), "ms");
        appendMetricRow(stream, "BenchmarkApp", "Median latency",
                        fixedValue(benchmark->latency_median_ms), "ms");
        appendMetricRow(stream, "BenchmarkApp", "Average latency",
                        fixedValue(benchmark->latency_average_ms), "ms");
        appendMetricRow(stream, "BenchmarkApp", "Minimum latency",
                        fixedValue(benchmark->latency_min_ms), "ms");
        appendMetricRow(stream, "BenchmarkApp", "Maximum latency",
                        fixedValue(benchmark->latency_max_ms), "ms");
        appendMetricRow(stream, "BenchmarkApp", "Throughput",
                        fixedValue(benchmark->throughput_fps), "FPS");
    }
    else
    {
        appendMetricRow(stream, "BenchmarkApp", "Status", "Failed", "-");
        if (!benchmark_error.empty())
            stream << "\nBenchmark error:\n" << benchmark_error << '\n';
    }

    return stream.str();
}

// 获取与当前可执行文件同目录的性能日志路径。
std::string resolveMptLogPath()
{
    char exe_path[4096] = {0};
    const ssize_t path_len = ::readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
    if (path_len <= 0)
        return "network_mpt.log";

    exe_path[path_len] = '\0';
    return (std::filesystem::path(exe_path).parent_path() / "network_mpt.log").string();
}

// 将性能报告追加写入日志文件。
void appendMptLog(const std::string &message)
{
    static std::mutex mutex;
    static std::ofstream stream(resolveMptLogPath(), std::ios::out | std::ios::app);
    std::lock_guard<std::mutex> lock(mutex);
    if (!stream.is_open())
        return;

    stream << message;
    if (message.empty() || message.back() != '\n')
        stream << '\n';
    stream.flush();
}

// 将外部命令参数转换为安全的 shell 单引号字符串。
std::string shellQuote(const std::string &value)
{
    std::string quoted = "'";
    for (char ch : value)
    {
        if (ch == '\'')
            quoted += "'\\''";
        else
            quoted += ch;
    }
    quoted += "'";
    return quoted;
}

// 删除字符串首尾的 ASCII 空白字符。
std::string trimAscii(std::string value)
{
    auto is_space = [](unsigned char ch)
    { return std::isspace(ch) != 0; };

    while (!value.empty() && is_space(static_cast<unsigned char>(value.front())))
        value.erase(value.begin());
    while (!value.empty() && is_space(static_cast<unsigned char>(value.back())))
        value.pop_back();

    return value;
}

// 提取指定标签与后缀之间的文本。
bool extractTokenBetween(const std::string &text, const std::string &label, const std::string &suffix, std::string &token)
{
    const std::size_t label_pos = text.find(label);
    if (label_pos == std::string::npos)
        return false;

    const std::size_t value_begin = label_pos + label.size();
    const std::size_t value_end = text.find(suffix, value_begin);
    if (value_end == std::string::npos || value_end <= value_begin)
        return false;

    token = trimAscii(text.substr(value_begin, value_end - value_begin));
    return !token.empty();
}

// 从 benchmark_app 输出中提取浮点指标。
bool extractBenchmarkValue(const std::string &text, const std::string &label, const std::string &suffix, double &value)
{
    std::string token;
    if (!extractTokenBetween(text, label, suffix, token))
        return false;

    char *end = nullptr;
    value = std::strtod(token.c_str(), &end);
    return end != token.c_str();
}

// 从 benchmark_app 输出中提取整数指标。
bool extractBenchmarkValue(const std::string &text, const std::string &label, const std::string &suffix, int &value)
{
    std::string token;
    if (!extractTokenBetween(text, label, suffix, token))
        return false;

    char *end = nullptr;
    value = static_cast<int>(std::strtol(token.c_str(), &end, 10));
    return end != token.c_str();
}

} // namespace detail

// 运行 benchmark_app，解析结果并输出统一的性能报告。
void runOfficialBenchmark(const std::string &model_path,
                          const std::string &device,
                          const std::string &infer_mode,
                          const OfficialBenchmarkConfig &config)
{
    detail::EndToEndResult end_to_end;
    {
        std::lock_guard<std::mutex> lock(detail::end_to_end_mutex);
        end_to_end = detail::latest_end_to_end_result;
        detail::latest_end_to_end_result = {};
    }
    const auto emit_report = [&](const detail::OfficialBenchmarkResult *benchmark,
                                 const std::string &benchmark_error)
    {
        const std::string report = detail::formatPerformanceReport(model_path,
                                                                   device,
                                                                   infer_mode,
                                                                   config,
                                                                   end_to_end,
                                                                   benchmark,
                                                                   benchmark_error);
        std::cout << report << std::endl;
        detail::appendMptLog(report);
    };

    // benchmark_app 只区分 sync/async API，多请求模式统一映射为 async。
    const std::string benchmark_api = infer_mode == "sync" ? "sync" : "async";
    std::ostringstream command;
    command << detail::shellQuote(config.benchmark_app_path)
            << " -m " << detail::shellQuote(model_path)
            << " -d " << detail::shellQuote(device)
            << " -hint " << detail::shellQuote(config.perf_hint)
            << " -api " << detail::shellQuote(benchmark_api);

    if (config.iterations > 0)
        command << " -niter " << config.iterations;
    else
        command << " -t " << config.time_seconds;
    if (config.inference_only)
        command << " -inference_only true";
    if (config.no_warmup)
        command << " -no_warmup";
    command << " 2>&1";

    std::unique_ptr<FILE, int (*)(FILE *)> pipe(popen(command.str().c_str(), "r"), pclose);
    if (!pipe)
    {
        emit_report(nullptr, "failed to launch benchmark_app");
        throw std::runtime_error("failed to launch benchmark_app");
    }

    std::string output;
    char buffer[4096] = {0};
    while (std::fgets(buffer, sizeof(buffer), pipe.get()) != nullptr)
        output += buffer;

    // 先读取完整输出，再检查子进程退出码，失败时保留原始诊断信息。
    const int exit_code = pclose(pipe.release());
    if (exit_code != 0)
    {
        emit_report(nullptr, "benchmark_app failed:\n" + output);
        throw std::runtime_error("benchmark_app failed:\n" + output);
    }

    // 解析字段和单位来自 benchmark_app 的标准文本输出。
    detail::OfficialBenchmarkResult result;
    const bool parsed_ok =
        detail::extractBenchmarkValue(output, "Count:", "iterations", result.iterations) &&
        detail::extractBenchmarkValue(output, "Duration:", "ms", result.duration_ms) &&
        detail::extractBenchmarkValue(output, "Median:", "ms", result.latency_median_ms) &&
        detail::extractBenchmarkValue(output, "Average:", "ms", result.latency_average_ms) &&
        detail::extractBenchmarkValue(output, "Min:", "ms", result.latency_min_ms) &&
        detail::extractBenchmarkValue(output, "Max:", "ms", result.latency_max_ms) &&
        detail::extractBenchmarkValue(output, "Throughput:", "FPS", result.throughput_fps);
    if (!parsed_ok)
    {
        emit_report(nullptr, "failed to parse benchmark_app output:\n" + output);
        throw std::runtime_error("failed to parse benchmark_app output:\n" + output);
    }

    emit_report(&result, {});
}

} // namespace MPT
