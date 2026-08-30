#pragma once

#include "network_deployment_interface.hpp"

#include <cstddef>
#include <deque>
#include <optional>
#include <vector>

struct ArmorDetectionFrame
{
    cv::Mat image;
    std::vector<NetArmorResult> results;
};

struct RuneDetectionFrame
{
    cv::Mat image;
    std::vector<NetRuneResult> results;
};

class ArmorDetector final
{
public:
    explicit ArmorDetector(const JsonConfig &config,
                           const DebugConfig &debug = DebugConfig());

    std::optional<ArmorDetectionFrame> process(const cv::Mat &image,
                                               int my_color);

private:
    std::size_t m_pipeline_delay;
    ArmorModel m_model;
    std::deque<cv::Mat> m_submitted_frames;
};

class RuneDetector final
{
public:
    explicit RuneDetector(const JsonConfig &config,
                          const DebugConfig &debug = DebugConfig());

    std::optional<RuneDetectionFrame> process(const cv::Mat &image);

private:
    std::size_t m_pipeline_delay;
    RuneModel m_model;
    std::deque<cv::Mat> m_submitted_frames;
};
