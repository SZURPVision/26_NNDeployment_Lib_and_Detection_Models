#include "network_performance.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <unistd.h>

// 性能支持模块：部署库端到端分阶段统计，以及独立的 benchmark_app 官方基准。
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

std::mutex &endToEndResultMutex()
{
    static std::mutex mutex;
    return mutex;
}

EndToEndResult &latestEndToEndResult()
{
    static EndToEndResult result;
    return result;
}

void recordEndToEndResult(const EndToEndResult &result)
{
    std::lock_guard<std::mutex> lock(endToEndResultMutex());
    latestEndToEndResult() = result;
}

EndToEndResult takeEndToEndResult()
{
    std::lock_guard<std::mutex> lock(endToEndResultMutex());
    const EndToEndResult result = latestEndToEndResult();
    latestEndToEndResult() = {};
    return result;
}
} // namespace detail

void SpeedStats::update(double preprocess_us, double inference_us, double postprocess_us)
{
    m_total_preprocess_us += preprocess_us;
    m_total_inference_us += inference_us;
    m_total_postprocess_us += postprocess_us;
    m_max_inference_us = std::max(m_max_inference_us, inference_us);
    m_min_inference_us = m_call_count == 0 ? inference_us : std::min(m_min_inference_us, inference_us);
    ++m_call_count;
}

void SpeedStats::printCurrentStats() const
{
    const double count = static_cast<double>(m_call_count);
    const double average_preprocess_us = m_call_count > 0 ? m_total_preprocess_us / count : 0.0;
    const double average_inference_us = m_call_count > 0 ? m_total_inference_us / count : 0.0;
    const double average_postprocess_us = m_call_count > 0 ? m_total_postprocess_us / count : 0.0;
    const double average_total_us = average_preprocess_us + average_inference_us + average_postprocess_us;
    const double average_fps = average_total_us > 0.0 ? 1'000'000.0 / average_total_us : 0.0;

    detail::recordEndToEndResult({m_call_count,
                                  average_preprocess_us,
                                  average_inference_us,
                                  average_postprocess_us,
                                  average_total_us,
                                  m_min_inference_us,
                                  m_max_inference_us,
                                  average_fps});

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

std::string fixedValue(double value)
{
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(2) << value;
    return stream.str();
}

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

std::string resolveMptLogPath()
{
    char exe_path[4096] = {0};
    const ssize_t path_len = ::readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
    if (path_len <= 0)
        return "network_mpt.log";

    exe_path[path_len] = '\0';
    std::string exe_full_path(exe_path);
    const std::size_t split_pos = exe_full_path.find_last_of('/');
    if (split_pos == std::string::npos)
        return "network_mpt.log";

    return exe_full_path.substr(0, split_pos + 1) + "network_mpt.log";
}

std::mutex &mptLogMutex()
{
    static std::mutex mutex;
    return mutex;
}

std::ofstream &mptLogStream()
{
    static std::ofstream stream(resolveMptLogPath(), std::ios::out | std::ios::app);
    return stream;
}

void appendMptLog(const std::string &message)
{
    std::lock_guard<std::mutex> lock(mptLogMutex());
    std::ofstream &stream = mptLogStream();
    if (!stream.is_open())
        return;

    stream << message;
    if (message.empty() || message.back() != '\n')
        stream << '\n';
    stream.flush();
}

std::string shellQuote(const std::string &value)
{
    // 所有外部参数均使用单引号转义，避免路径和设备字符串被 shell 拆分。
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
    if (token.empty())
        return false;

    return true;
}

bool extractBenchmarkValue(const std::string &text, const std::string &label, const std::string &suffix, double &value)
{
    std::string token;
    if (!extractTokenBetween(text, label, suffix, token))
        return false;

    char *end = nullptr;
    value = std::strtod(token.c_str(), &end);
    if (end == token.c_str())
        return false;
    return true;
}

bool extractBenchmarkValue(const std::string &text, const std::string &label, const std::string &suffix, int &value)
{
    std::string token;
    if (!extractTokenBetween(text, label, suffix, token))
        return false;

    char *end = nullptr;
    value = static_cast<int>(std::strtol(token.c_str(), &end, 10));
    if (end == token.c_str())
        return false;
    return true;
}

} // namespace detail

void runOfficialBenchmark(const std::string &model_path,
                          const std::string &device,
                          const std::string &infer_mode,
                          const OfficialBenchmarkConfig &config)
{
    const detail::EndToEndResult end_to_end = detail::takeEndToEndResult();
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
