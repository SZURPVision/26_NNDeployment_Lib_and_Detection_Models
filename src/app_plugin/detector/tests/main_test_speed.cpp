#include "DetectorRuntime.hpp"
#include "NNDetector.hpp"
#include "network_performance.hpp"

#include <opencv2/videoio.hpp>

#include <filesystem>
#include <iostream>
#include <optional>

int main(int argc, char **argv)
{
    try
    {
        const std::optional<std::filesystem::path> project_root =
            argc > 1 ? std::optional<std::filesystem::path>(argv[1]) : std::nullopt;
        const DetectorProjectPaths paths = resolveProjectPaths(project_root);
        const JsonConfig model_config{paths.json_path.string(),
                                      "rune_detect",
                                      paths.model_folder.string()};

        cv::VideoCapture video = openVideo(paths.rune_video, "性能测试视频");
        cv::Mat frame;
        if (!video.read(frame) || frame.empty())
            throw std::runtime_error("性能测试视频没有可读取的首帧: " +
                                     paths.armor_video.string());

        NNDetector detector(NNDetector::Task::Rune,
                            model_config,
                            DebugConfig(false, true));

        // 先固定输入调用 200 次，统计部署库预处理、推理和后处理的端到端耗时。
        for (int index = 0; index < 2000; ++index)
            detector.detectRune(frame);

        // 官方 benchmark 与端到端统计相互独立，参数继续读取同一个 JSON 节点。
        const JsonModelRuntime runtime = readJsonModelRuntime(model_config);
        MPT::OfficialBenchmarkConfig config;
        config.benchmark_app_path = findExecutableOnPath("benchmark_app").string();
        config.perf_hint = "throughput";
        config.time_seconds = 20;
        config.inference_only = false;
        MPT::runOfficialBenchmark(runtime.model_path.string(),
                                  runtime.device,
                                  runtime.infer_mode,
                                  config);
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "main_test_speed failed: " << error.what() << std::endl;
        return 1;
    }
}
