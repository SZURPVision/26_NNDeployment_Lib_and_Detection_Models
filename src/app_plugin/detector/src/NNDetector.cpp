#include "NNDetector.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace
{
cv::Point toPoint(const cv::Point2d &point)
{
    return {cvRound(point.x), cvRound(point.y)};
}

void drawLabel(cv::Mat &image, const std::string &text, const cv::Point &origin)
{
    int baseline = 0;
    const cv::Size size = cv::getTextSize(
        text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
    const cv::Point position(
        std::max(0, origin.x),
        std::max(size.height + 3, origin.y));

    cv::rectangle(image,
                  cv::Rect(position.x,
                           position.y - size.height - 3,
                           size.width + 6,
                           size.height + baseline + 6),
                  cv::Scalar(0, 0, 0),
                  cv::FILLED);
    cv::putText(image,
                text,
                position + cv::Point(3, 0),
                cv::FONT_HERSHEY_SIMPLEX,
                0.5,
                cv::Scalar(255, 255, 255),
                1,
                cv::LINE_AA);
}
} // namespace

class NNDetector::Impl
{
public:
    Impl(Task task,
         const JsonConfig &json_config,
         const DebugConfig &debug_config)
        : m_task(task)
    {
        if (m_task == Task::Armor)
            m_armor_model = std::make_unique<ArmorModel>(json_config, debug_config);
        else
            m_rune_model = std::make_unique<RuneModel>(json_config, debug_config);
    }

    std::vector<NetArmorResult> detectArmor(const cv::Mat &image, int my_color)
    {
        if (image.empty())
            throw std::invalid_argument("装甲板检测不能输入空图像");
        if (!m_armor_model)
            throw std::logic_error("当前 NNDetector 未配置为装甲板任务");
        return m_armor_model->netProcess(image, my_color);
    }

    std::vector<NetRuneResult> detectRune(const cv::Mat &image)
    {
        if (image.empty())
            throw std::invalid_argument("神符检测不能输入空图像");
        if (!m_rune_model)
            throw std::logic_error("当前 NNDetector 未配置为神符任务");
        return m_rune_model->netProcess(image);
    }

private:
    Task m_task;
    std::unique_ptr<ArmorModel> m_armor_model;
    std::unique_ptr<RuneModel> m_rune_model;
};

NNDetector::NNDetector(Task task,
                       const JsonConfig &json_config,
                       const DebugConfig &debug_config)
    : m_impl(std::make_unique<Impl>(task, json_config, debug_config))
{}

NNDetector::~NNDetector() = default;
NNDetector::NNDetector(NNDetector &&) noexcept = default;
NNDetector &NNDetector::operator=(NNDetector &&) noexcept = default;

std::vector<NetArmorResult> NNDetector::detectArmor(const cv::Mat &image,
                                                    int my_color)
{
    return m_impl->detectArmor(image, my_color);
}

std::vector<NetRuneResult> NNDetector::detectRune(const cv::Mat &image)
{
    return m_impl->detectRune(image);
}

void NNDetector::drawArmorResults(
    cv::Mat &image,
    const std::vector<NetArmorResult> &results)
{
    for (const NetArmorResult &result : results)
    {
        if (result.points.size() != 4)
            throw std::runtime_error("装甲板结果必须包含 4 个关键点");

        for (std::size_t index = 0; index < result.points.size(); ++index)
        {
            cv::line(image,
                     toPoint(result.points[index]),
                     toPoint(result.points[(index + 1) % result.points.size()]),
                     cv::Scalar(0, 255, 255),
                     2,
                     cv::LINE_AA);
            cv::circle(image,
                       toPoint(result.points[index]),
                       4,
                       cv::Scalar(0, 255, 0),
                       cv::FILLED,
                       cv::LINE_AA);
        }

        std::ostringstream label;
        label << "class_id:" << result.armor_id
              << " color_id:" << result.color_id
              << " score:" << std::fixed << std::setprecision(2) << result.score;
        drawLabel(image, label.str(), toPoint(result.points.front()) + cv::Point(0, -12));
    }
}

void NNDetector::drawRuneResults(
    cv::Mat &image,
    const std::vector<NetRuneResult> &results)
{
    constexpr int outline_indices[] = {0, 1, 4, 3};

    for (const NetRuneResult &result : results)
    {
        if (result.points.size() != 5)
            throw std::runtime_error("神符结果必须包含 5 个关键点");

        for (std::size_t index = 0; index < std::size(outline_indices); ++index)
        {
            const int current = outline_indices[index];
            const int next = outline_indices[(index + 1) % std::size(outline_indices)];
            cv::line(image,
                     toPoint(result.points[current]),
                     toPoint(result.points[next]),
                     cv::Scalar(255, 255, 0),
                     2,
                     cv::LINE_AA);
            cv::circle(image,
                       toPoint(result.points[current]),
                       4,
                       cv::Scalar(0, 255, 255),
                       cv::FILLED,
                       cv::LINE_AA);
        }
        cv::circle(image,
                   toPoint(result.points[2]),
                   5,
                   cv::Scalar(0, 255, 0),
                   cv::FILLED,
                   cv::LINE_AA);

        std::ostringstream label;
        label << "class_id:" << result.class_id
              << " score:" << std::fixed << std::setprecision(2) << result.score;
        drawLabel(image, label.str(), toPoint(result.points.front()) + cv::Point(0, -12));
    }
}
