#include "NNDetector.hpp"

#include <stdexcept>
#include <utility>

namespace
{
cv::Mat takeMatchedFrame(std::deque<cv::Mat> &frames, std::size_t delay)
{
    if (frames.size() <= delay)
        return {};
    cv::Mat matched = std::move(frames.front());
    frames.pop_front();
    return matched;
}
} // namespace

ArmorDetector::ArmorDetector(const JsonConfig &config,
                             std::size_t pipeline_delay,
                             const DebugConfig &debug)
    : m_model(config, debug), m_pipeline_delay(pipeline_delay)
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

RuneDetector::RuneDetector(const JsonConfig &config,
                           std::size_t pipeline_delay,
                           const DebugConfig &debug)
    : m_model(config, debug), m_pipeline_delay(pipeline_delay)
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
