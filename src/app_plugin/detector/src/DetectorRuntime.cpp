#include "DetectorRuntime.hpp"

#include <opencv2/core/persistence.hpp>

#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <system_error>
#include <unistd.h>

#ifndef NETLIB_PROJECT_ROOT
#define NETLIB_PROJECT_ROOT "."
#endif

namespace
{
void requireRegularFile(const std::filesystem::path &path,
                        const std::string &description)
{
    if (!std::filesystem::is_regular_file(path))
        throw std::runtime_error(description + "不存在或不是普通文件: " + path.string());
}

std::filesystem::path executableDirectory()
{
    std::error_code error;
    const std::filesystem::path executable =
        std::filesystem::read_symlink("/proc/self/exe", error);
    if (!error && executable.has_parent_path())
        return executable.parent_path();
    return std::filesystem::current_path();
}
} // namespace

DetectorProjectPaths resolveProjectPaths(
    const std::optional<std::filesystem::path> &project_root)
{
    DetectorProjectPaths paths;
    paths.project_root = project_root.has_value()
                             ? std::filesystem::absolute(*project_root)
                             : std::filesystem::path(NETLIB_PROJECT_ROOT);
    paths.project_root = paths.project_root.lexically_normal();
    paths.json_path = paths.project_root /
                      "src/app_plugin/detector/config/detect.json";
    paths.model_folder = paths.project_root / "所有模型/openvino";
    paths.armor_video = paths.project_root / "测试视频/中距离陀螺.avi";
    paths.rune_video = paths.project_root / "测试视频/符.avi";

    requireRegularFile(paths.json_path, "检测配置");
    if (!std::filesystem::is_directory(paths.model_folder))
        throw std::runtime_error("模型目录不存在: " + paths.model_folder.string());
    requireRegularFile(paths.armor_video, "装甲板演示视频");
    requireRegularFile(paths.rune_video, "神符演示视频");

    for (const std::string &model_key :
         {"armor_v8", "armor_v5", "rune_detect"})
    {
        readJsonModelRuntime(JsonConfig{paths.json_path.string(),
                                        model_key,
                                        paths.model_folder.string()});
    }
    return paths;
}

std::filesystem::path resultDirectory(const std::string &program_name)
{
    const std::filesystem::path executable_dir = executableDirectory();
    const std::filesystem::path build_dir =
        executable_dir.filename() == "bin"
            ? executable_dir.parent_path()
            : executable_dir;
    const std::filesystem::path output = build_dir / "results" / program_name;
    std::filesystem::create_directories(output);
    return output;
}

cv::VideoCapture openVideo(const std::filesystem::path &path,
                           const std::string &description)
{
    requireRegularFile(path, description);
    cv::VideoCapture capture(path.string());
    if (!capture.isOpened())
        throw std::runtime_error("无法打开" + description + ": " + path.string());
    return capture;
}

cv::VideoWriter createVideoWriter(const cv::VideoCapture &capture,
                                  const std::filesystem::path &output_path)
{
    const int width = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
    const int height = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = capture.get(cv::CAP_PROP_FPS);
    if (width <= 0 || height <= 0)
        throw std::runtime_error("输入视频的宽高无效");
    if (!std::isfinite(fps) || fps <= 0.0)
        fps = 30.0;

    std::filesystem::create_directories(output_path.parent_path());
    cv::VideoWriter writer(output_path.string(),
                           cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                           fps,
                           cv::Size(width, height));
    if (!writer.isOpened())
        throw std::runtime_error("无法创建结果视频: " + output_path.string());
    return writer;
}

JsonModelRuntime readJsonModelRuntime(const JsonConfig &json_config)
{
    cv::FileStorage config(json_config.json_path, cv::FileStorage::READ);
    if (!config.isOpened())
        throw std::runtime_error("无法打开模型配置文件: " + json_config.json_path);

    const cv::FileNode node = config[json_config.model_key];
    if (node.empty())
        throw std::runtime_error("模型配置中不存在键: " + json_config.model_key);

    const std::string xml_name = static_cast<std::string>(node["xml"]);
    if (xml_name.empty())
        throw std::runtime_error("模型配置缺少 xml: " + json_config.model_key);

    JsonModelRuntime runtime;
    runtime.model_path =
        (std::filesystem::path(json_config.model_folder) / xml_name).lexically_normal();
    runtime.device = static_cast<std::string>(node["device"]);
    runtime.infer_mode = static_cast<std::string>(node["infer_mode"]);
    if (runtime.device.empty())
        runtime.device = "GPU";
    if (runtime.infer_mode.empty())
        runtime.infer_mode = "async";

    requireRegularFile(runtime.model_path, "性能测试模型");
    return runtime;
}

std::filesystem::path findExecutableOnPath(const std::string &name)
{
    const std::filesystem::path requested(name);
    if (requested.has_parent_path())
    {
        if (::access(requested.c_str(), X_OK) == 0)
            return std::filesystem::absolute(requested);
        throw std::runtime_error("可执行文件不可用: " + requested.string());
    }

    const char *path_value = std::getenv("PATH");
    if (!path_value)
        throw std::runtime_error("PATH 环境变量不存在，无法查找: " + name);

    const std::string path_list(path_value);
    std::size_t begin = 0;
    while (begin <= path_list.size())
    {
        const std::size_t end = path_list.find(':', begin);
        const std::filesystem::path directory =
            path_list.substr(begin, end == std::string::npos
                                        ? std::string::npos
                                        : end - begin);
        const std::filesystem::path candidate = directory / name;
        if (::access(candidate.c_str(), X_OK) == 0)
            return candidate;
        if (end == std::string::npos)
            break;
        begin = end + 1;
    }

    throw std::runtime_error("PATH 中找不到可执行文件: " + name);
}
