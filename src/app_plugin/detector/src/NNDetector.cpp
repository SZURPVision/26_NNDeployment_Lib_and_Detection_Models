#include "NNDetector.hpp"

#include <opencv2/core/persistence.hpp>

#include <stdexcept>
#include <string>
#include <utility>

namespace
{
std::size_t pipelineDelay(const JsonConfig &config)
{
    cv::FileStorage file(config.json_path, cv::FileStorage::READ);
    if (!file.isOpened())
        throw std::runtime_error("无法打开检测配置: " + config.json_path);

    const cv::FileNode node = file[config.model_key];
    if (node.empty())
        throw std::runtime_error("检测配置中不存在节点: " + config.model_key);

    const std::string mode = static_cast<std::string>(node["infer_mode"]);
    if (mode == "sync")
        return 0;
    if (mode == "async")
        return 1;
    if (mode == "async4")
        return 3;
    throw std::runtime_error("不支持的推理模式: " + mode);
}

cv::Mat takeMatchedFrame(std::deque<cv::Mat> &frames, std::size_t delay)
{
    if (frames.size() <= delay)
        return {};
    cv::Mat matched = std::move(frames.front());
    frames.pop_front();
    return matched;
}
} // namespace

ArmorDetector::ArmorDetector(const JsonConfig &config, const DebugConfig &debug)
    : m_pipeline_delay(pipelineDelay(config)), m_model(config, debug)
{}

std::optional<ArmorDetectionFrame> ArmorDetector::process(const cv::Mat &image,
                                                          int my_color)
{
    if (image.empty())
        throw std::invalid_argument("装甲板检测不能输入空图像");

    m_submitted_frames.push_back(image.clone());
    std::vector<NetArmorResult> results = m_model.netProcess(image, my_color);
    cv::Mat matched = takeMatchedFrame(m_submitted_frames, m_pipeline_delay);
    if (matched.empty())
        return std::nullopt;
    return ArmorDetectionFrame{std::move(matched), std::move(results)};
}

RuneDetector::RuneDetector(const JsonConfig &config, const DebugConfig &debug)
    : m_pipeline_delay(pipelineDelay(config)), m_model(config, debug)
{}

std::optional<RuneDetectionFrame> RuneDetector::process(const cv::Mat &image)
{
    if (image.empty())
        throw std::invalid_argument("能量机关检测不能输入空图像");

    m_submitted_frames.push_back(image.clone());
    std::vector<NetRuneResult> results = m_model.netProcess(image);
    cv::Mat matched = takeMatchedFrame(m_submitted_frames, m_pipeline_delay);
    if (matched.empty())
        return std::nullopt;
    return RuneDetectionFrame{std::move(matched), std::move(results)};
}
