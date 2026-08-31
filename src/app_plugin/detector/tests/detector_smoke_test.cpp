#include "NNDetector.hpp"

#include <opencv2/videoio.hpp>

#include <cstddef>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#ifndef NNDEPLOYMENT_PROJECT_ROOT
#define NNDEPLOYMENT_PROJECT_ROOT "."
#endif

namespace fs = std::filesystem;

namespace
{
constexpr std::size_t kSyncPipelineDelay = 0;

void require(bool condition, const std::string &message)
{
    if (!condition)
        throw std::runtime_error(message);
}

cv::Mat readFrame(const fs::path &path)
{
    cv::VideoCapture video(path.string());
    cv::Mat frame;
    require(video.isOpened() && video.read(frame) && !frame.empty(),
            "无法读取测试帧: " + path.string());
    return frame;
}

JsonConfig config(const fs::path &root, const std::string &key)
{
    return JsonConfig{
        (root / "src/app_plugin/detector/tests/detector_smoke.json").string(),
        key,
        (root / "所有模型/openvino").string()};
}

void testArmor(const fs::path &root, const std::string &key,
               const cv::Mat &frame)
{
    ArmorDetector detector(config(root, key), kSyncPipelineDelay);
    auto detected = detector.process(frame, 2);
    require(detected.has_value() && !detected->image.empty(),
            key + " 首帧推理未返回图像");

    const cv::Mat blank = cv::Mat::zeros(frame.size(), frame.type());
    auto empty_result = detector.process(blank, 2);
    require(empty_result.has_value() && empty_result->results.empty(),
            key + " 空白图像应返回空结果");

    bool rejected_empty_input = false;
    try
    {
        detector.process(cv::Mat{}, 2);
    }
    catch (const std::invalid_argument &)
    {
        rejected_empty_input = true;
    }
    require(rejected_empty_input, key + " 未拒绝空输入");
    std::cout << key << ": first-frame, blank-result and empty-input checks passed\n";
}

void testRune(const fs::path &root, const cv::Mat &frame)
{
    RuneDetector detector(config(root, "rune_detect"), kSyncPipelineDelay);
    auto detected = detector.process(frame);
    require(detected.has_value() && !detected->image.empty(),
            "rune_detect 首帧推理未返回图像");

    const cv::Mat blank = cv::Mat::zeros(frame.size(), frame.type());
    auto empty_result = detector.process(blank);
    require(empty_result.has_value() && empty_result->results.empty(),
            "rune_detect 空白图像应返回空结果");

    bool rejected_empty_input = false;
    try
    {
        detector.process(cv::Mat{});
    }
    catch (const std::invalid_argument &)
    {
        rejected_empty_input = true;
    }
    require(rejected_empty_input, "rune_detect 未拒绝空输入");
    std::cout << "rune_detect: first-frame, blank-result and empty-input checks passed\n";
}

void testInvalidConfig(const fs::path &root)
{
    bool rejected = false;
    try
    {
        ArmorDetector detector(config(root, "missing_model"),
                               kSyncPipelineDelay);
    }
    catch (const std::exception &)
    {
        rejected = true;
    }
    require(rejected, "错误配置未被拒绝");
    std::cout << "invalid-config check passed\n";
}

void testMismatchedTask(const fs::path &root)
{
    bool armor_rejected = false;
    try
    {
        ArmorDetector detector(config(root, "rune_detect"),
                               kSyncPipelineDelay);
    }
    catch (const std::invalid_argument &)
    {
        armor_rejected = true;
    }
    require(armor_rejected, "ArmorDetector 未拒绝神符模型");

    bool rune_rejected = false;
    try
    {
        RuneDetector detector(config(root, "armor_v5"),
                              kSyncPipelineDelay);
    }
    catch (const std::invalid_argument &)
    {
        rune_rejected = true;
    }
    require(rune_rejected, "RuneDetector 未拒绝装甲板模型");
    std::cout << "mismatched-task checks passed\n";
}
} // namespace

int main()
{
    try
    {
        const fs::path root = fs::path(NNDEPLOYMENT_PROJECT_ROOT);
        const cv::Mat armor_frame = readFrame(root / "测试视频/装甲板.mp4");
        const cv::Mat rune_frame = readFrame(root / "测试视频/符.avi");

        testArmor(root, "armor_v5", armor_frame);
        testArmor(root, "armor_v8", armor_frame);
        testRune(root, rune_frame);
        testInvalidConfig(root);
        testMismatchedTask(root);
        std::cout << "detector smoke test passed" << std::endl;
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "detector smoke test failed: " << error.what() << std::endl;
        return 1;
    }
}
