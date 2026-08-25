#pragma once

#include "network_deployment_interface.hpp"

#include <memory>
#include <vector>

class NNDetector final
{
public:
    enum class Task
    {
        Armor,
        Rune
    };

    NNDetector(Task task,
               const JsonConfig &json_config,
               const DebugConfig &debug_config = DebugConfig());
    ~NNDetector();

    NNDetector(const NNDetector &) = delete;
    NNDetector &operator=(const NNDetector &) = delete;
    NNDetector(NNDetector &&) noexcept;
    NNDetector &operator=(NNDetector &&) noexcept;

    std::vector<NetArmorResult> detectArmor(const cv::Mat &image, int my_color);
    std::vector<NetRuneResult> detectRune(const cv::Mat &image);

    static void drawArmorResults(cv::Mat &image,
                                 const std::vector<NetArmorResult> &results);
    static void drawRuneResults(cv::Mat &image,
                                const std::vector<NetRuneResult> &results);

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};
