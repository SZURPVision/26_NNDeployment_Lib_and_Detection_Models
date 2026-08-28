#include "DetectorRuntime.hpp"
#include "NNDetector.hpp"
#include "network_performance.hpp"

#include <opencv2/videoio.hpp>

#include <array>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

namespace
{
constexpr int kTestIterations = 2000;
constexpr int kArmorColorAll = 2;

struct ModelTestCase
{
    std::string name;
    std::string config_key;
    NNDetector::Task task;
    std::filesystem::path video_path;
};

std::filesystem::path findProjectRoot(
    const std::optional<std::filesystem::path> &requested_root,
    const char *executable)
{
    if (requested_root.has_value())
        return std::filesystem::absolute(*requested_root).lexically_normal();

    const std::array<std::filesystem::path, 2> search_starts = {
        std::filesystem::current_path(),
        std::filesystem::absolute(executable).parent_path()};

    for (const std::filesystem::path &start : search_starts)
    {
        std::filesystem::path candidate = start.lexically_normal();
        while (!candidate.empty())
        {
            const std::filesystem::path config_path =
                candidate / "src/app_plugin/detector/config/detect.json";
            if (std::filesystem::is_regular_file(config_path))
                return candidate;

            const std::filesystem::path parent = candidate.parent_path();
            if (parent == candidate)
                break;
            candidate = parent;
        }
    }

    throw std::runtime_error(
        "Cannot locate the project root. Run from the repository or pass it as argv[1].");
}

void runModelTest(const ModelTestCase &test_case,
                  const std::filesystem::path &json_path,
                  const std::filesystem::path &model_folder,
                  const std::filesystem::path &benchmark_app)
{
    std::cout << "\n============================================================\n"
              << "Testing model: " << test_case.name << '\n'
              << "Config key: " << test_case.config_key << '\n'
              << "============================================================"
              << std::endl;

    cv::VideoCapture video = openVideo(test_case.video_path,
                                       test_case.name + " test video");
    cv::Mat frame;
    if (!video.read(frame) || frame.empty())
        throw std::runtime_error("No readable frame in test video: " +
                                 test_case.video_path.string());

    const JsonConfig model_config{json_path.string(),
                                  test_case.config_key,
                                  model_folder.string()};
    NNDetector detector(test_case.task,
                        model_config,
                        DebugConfig(false, true));

    for (int index = 0; index < kTestIterations; ++index)
    {
        if (test_case.task == NNDetector::Task::Armor)
            detector.detectArmor(frame, kArmorColorAll);
        else
            detector.detectRune(frame);
    }

    const JsonModelRuntime runtime = readJsonModelRuntime(model_config);
    MPT::OfficialBenchmarkConfig config;
    config.benchmark_app_path = benchmark_app.string();
    config.perf_hint = "throughput";
    config.time_seconds = 20;
    config.inference_only = false;
    MPT::runOfficialBenchmark(runtime.model_path.string(),
                              runtime.device,
                              runtime.infer_mode,
                              config);

    std::cout << "Completed model: " << test_case.name << std::endl;
}
} // namespace

int main(int argc, char **argv)
{
    try
    {
        const std::optional<std::filesystem::path> requested_root =
            argc > 1 ? std::optional<std::filesystem::path>(argv[1]) : std::nullopt;
        const std::filesystem::path project_root =
            findProjectRoot(requested_root, argv[0]);
        const std::filesystem::path json_path =
            project_root / "src/app_plugin/detector/config/detect.json";
        const std::filesystem::path model_folder =
            project_root / "所有模型/openvino";
        const std::filesystem::path armor_video =
            project_root / "测试视频/中距离陀螺.avi";
        const std::filesystem::path rune_video =
            project_root / "测试视频/符.avi";
        const std::filesystem::path benchmark_app =
            findExecutableOnPath("benchmark_app");

        const std::array<ModelTestCase, 3> test_cases = {{
            {"Armor V8", "armor_v8", NNDetector::Task::Armor, armor_video},
            {"Armor V5", "armor_v5", NNDetector::Task::Armor, armor_video},
            {"Rune V8", "rune_detect", NNDetector::Task::Rune, rune_video},
        }};

        int failed_tests = 0;
        for (const ModelTestCase &test_case : test_cases)
        {
            try
            {
                runModelTest(test_case,
                             json_path,
                             model_folder,
                             benchmark_app);
            }
            catch (const std::exception &error)
            {
                ++failed_tests;
                std::cerr << "Model test failed [" << test_case.name
                          << "]: " << error.what() << std::endl;
            }
        }

        if (failed_tests != 0)
        {
            std::cerr << failed_tests << " of " << test_cases.size()
                      << " model tests failed." << std::endl;
            return 1;
        }

        std::cout << "\nAll three OpenVINO model tests completed successfully."
                  << std::endl;
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "main_test_speed failed: " << error.what() << std::endl;
        return 1;
    }
}
