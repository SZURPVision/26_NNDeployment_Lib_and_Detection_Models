#pragma once

#include "network_deployment_interface.hpp"

#include <opencv2/videoio.hpp>

#include <filesystem>
#include <optional>
#include <string>

struct DetectorProjectPaths
{
    std::filesystem::path project_root;
    std::filesystem::path json_path;
    std::filesystem::path model_folder;
    std::filesystem::path armor_video;
    std::filesystem::path rune_video;
};

struct JsonModelRuntime
{
    std::filesystem::path model_path;
    std::string device;
    std::string infer_mode;
};

DetectorProjectPaths resolveProjectPaths(
    const std::optional<std::filesystem::path> &project_root = std::nullopt);
std::filesystem::path resultDirectory(const std::string &program_name);
cv::VideoCapture openVideo(const std::filesystem::path &path,
                           const std::string &description);
cv::VideoWriter createVideoWriter(const cv::VideoCapture &capture,
                                  const std::filesystem::path &output_path);
JsonModelRuntime readJsonModelRuntime(const JsonConfig &json_config);
std::filesystem::path findExecutableOnPath(const std::string &name);
