#include "NNDetector.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

#ifndef NETLIB_PROJECT_ROOT
#define NETLIB_PROJECT_ROOT "."
#endif

namespace fs = std::filesystem;

namespace
{
struct VideoResult
{
    std::uint64_t frames = 0;
    fs::path output_path;
};

struct VideoClip
{
    cv::VideoCapture video;
    std::uint64_t frame_limit = 0;
};

constexpr double kShowcaseSeconds = 10.0;

fs::path projectRoot(int argc, char **argv)
{
    return fs::absolute(argc > 1 ? fs::path(argv[1])
                                 : fs::path(NETLIB_PROJECT_ROOT));
}

cv::VideoCapture openVideo(const fs::path &path)
{
    cv::VideoCapture video(path.string());
    if (!video.isOpened())
        throw std::runtime_error("无法打开测试视频: " + path.string());
    return video;
}

VideoClip openMiddleClip(const fs::path &path)
{
    cv::VideoCapture video = openVideo(path);
    const double fps = video.get(cv::CAP_PROP_FPS);
    const double frame_count = video.get(cv::CAP_PROP_FRAME_COUNT);
    if (!std::isfinite(fps) || fps <= 0.0 ||
        !std::isfinite(frame_count) || frame_count <= 0.0)
        throw std::runtime_error("测试视频帧率或帧数无效: " + path.string());

    const double clip_frames = std::min(frame_count, kShowcaseSeconds * fps);
    const double start_frame = std::max(0.0, (frame_count - clip_frames) / 2.0);
    if (!video.set(cv::CAP_PROP_POS_FRAMES, std::floor(start_frame)))
        throw std::runtime_error("无法定位测试视频中段: " + path.string());
    return {std::move(video), static_cast<std::uint64_t>(std::floor(clip_frames))};
}

cv::VideoWriter createWriter(const cv::VideoCapture &video, const fs::path &path)
{
    const cv::Size size(
        static_cast<int>(video.get(cv::CAP_PROP_FRAME_WIDTH)),
        static_cast<int>(video.get(cv::CAP_PROP_FRAME_HEIGHT)));
    double fps = video.get(cv::CAP_PROP_FPS);
    if (size.width <= 0 || size.height <= 0)
        throw std::runtime_error("测试视频尺寸无效");
    if (!std::isfinite(fps) || fps <= 0.0)
        fps = 30.0;

    fs::create_directories(path.parent_path());
    cv::VideoWriter writer(path.string(),
                           cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                           fps, size);
    if (!writer.isOpened())
        throw std::runtime_error("无法创建结果视频: " + path.string());
    return writer;
}

cv::Point point(const cv::Point2d &value)
{
    return {cvRound(value.x), cvRound(value.y)};
}

void drawLabel(cv::Mat &image, const std::string &text, cv::Point origin,
               double scale = 0.45)
{
    int baseline = 0;
    const cv::Size size = cv::getTextSize(
        text, cv::FONT_HERSHEY_SIMPLEX, scale, 1, &baseline);
    origin.x = std::clamp(origin.x, 0, std::max(0, image.cols - size.width - 6));
    origin.y = std::clamp(origin.y, size.height + 6,
                          std::max(size.height + 6, image.rows - baseline - 2));
    const cv::Rect background(
        origin.x, origin.y - size.height - 5,
        size.width + 6, size.height + baseline + 7);
    cv::rectangle(image, background, cv::Scalar(24, 24, 24), cv::FILLED);
    cv::rectangle(image, background, cv::Scalar(255, 255, 255), 1);
    cv::putText(image, text, origin + cv::Point(3, -2),
                cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(255, 255, 255),
                1, cv::LINE_AA);
}

void drawIndexedPoint(cv::Mat &image, const cv::Point2d &value,
                      std::size_t index, const cv::Scalar &color)
{
    const cv::Point center = point(value);
    cv::circle(image, center, 4, color, cv::FILLED, cv::LINE_AA);
    drawLabel(image, std::to_string(index), center + cv::Point(7, -5), 0.4);
}

std::string armorColorName(int color_id)
{
    constexpr std::array<const char *, 4> names = {
        "blue", "red", "white", "purple"};
    if (color_id < 0 || color_id >= static_cast<int>(names.size()))
        return "unknown";
    return names[static_cast<std::size_t>(color_id)];
}

void drawArmor(cv::Mat &image, const std::vector<NetArmorResult> &results)
{
    for (const NetArmorResult &result : results)
    {
        if (result.points.size() != 4)
            continue;
        for (std::size_t index = 0; index < result.points.size(); ++index)
            cv::line(image, point(result.points[index]),
                     point(result.points[(index + 1) % result.points.size()]),
                     cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
        drawLabel(image,
                  "class=" + std::to_string(result.armor_id) +
                      " color=" + armorColorName(result.color_id),
                  point(result.points.front()) + cv::Point(0, -28), 0.5);
        for (std::size_t index = 0; index < result.points.size(); ++index)
            drawIndexedPoint(image, result.points[index], index,
                             cv::Scalar(0, 255, 0));
    }
}

void drawRune(cv::Mat &image, const std::vector<NetRuneResult> &results)
{
    constexpr std::array<int, 4> outline = {0, 1, 4, 3};
    for (const NetRuneResult &result : results)
    {
        if (result.points.size() != 5)
            continue;
        for (std::size_t index = 0; index < outline.size(); ++index)
            cv::line(image, point(result.points[outline[index]]),
                     point(result.points[outline[(index + 1) % outline.size()]]),
                     cv::Scalar(255, 255, 0), 2, cv::LINE_AA);
        drawLabel(image, "class=" + std::to_string(result.class_id),
                  point(result.points.front()) + cv::Point(0, -28), 0.5);
        for (std::size_t index = 0; index < result.points.size(); ++index)
            drawIndexedPoint(image, result.points[index], index,
                             cv::Scalar(0, 255, 255));
    }
}

VideoResult runArmor(const fs::path &root,
                     const fs::path &config_path,
                     const fs::path &output_dir,
                     const std::string &config_key,
                     const std::string &output_name)
{
    const fs::path models = root / "所有模型/openvino";
    ArmorDetector detector(JsonConfig{config_path.string(), config_key, models.string()});
    VideoClip clip = openMiddleClip(root / "测试视频/装甲板.mp4");
    const fs::path output_path = output_dir / output_name;
    cv::VideoWriter writer = createWriter(clip.video, output_path);

    std::uint64_t count = 0;
    std::uint64_t submitted = 0;
    cv::Mat input;
    while (submitted < clip.frame_limit && clip.video.read(input))
    {
        ++submitted;
        std::optional<ArmorDetectionFrame> output = detector.process(input, 2);
        if (!output)
            continue;
        drawArmor(output->image, output->results);
        writer.write(output->image);
        ++count;
    }
    return {count, output_path};
}

VideoResult runRune(const fs::path &root, const fs::path &config_path,
                    const fs::path &output_dir)
{
    const fs::path models = root / "所有模型/openvino";
    RuneDetector detector(JsonConfig{config_path.string(), "rune_detect", models.string()});
    VideoClip clip = openMiddleClip(root / "测试视频/符.avi");
    const fs::path output_path = output_dir / "rune_result.mp4";
    cv::VideoWriter writer = createWriter(clip.video, output_path);

    std::uint64_t count = 0;
    std::uint64_t submitted = 0;
    cv::Mat input;
    while (submitted < clip.frame_limit && clip.video.read(input))
    {
        ++submitted;
        std::optional<RuneDetectionFrame> output = detector.process(input);
        if (!output)
            continue;
        drawRune(output->image, output->results);
        writer.write(output->image);
        ++count;
    }
    return {count, output_path};
}
} // namespace

int main(int argc, char **argv)
{
    try
    {
        const fs::path root = projectRoot(argc, argv);
        const fs::path config = argc > 2
                                    ? fs::absolute(fs::path(argv[2]))
                                    : root / "src/app_plugin/detector/config/detect.json";
        const fs::path output_dir = root / "build/results/main";
        const VideoResult v5 = runArmor(root, config, output_dir, "armor_v5",
                                        "armor_v5_result.mp4");
        const VideoResult v8 = runArmor(root, config, output_dir, "armor_v8",
                                        "armor_v8_result.mp4");
        const VideoResult rune = runRune(root, config, output_dir);

        std::cout << "Armor V5 output frames: " << v5.frames << '\n'
                  << "Armor V5 video: " << v5.output_path << '\n'
                  << "Armor V8 output frames: " << v8.frames << '\n'
                  << "Armor V8 video: " << v8.output_path << '\n'
                  << "Rune output frames: " << rune.frames << '\n'
                  << "Rune video: " << rune.output_path << std::endl;
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "main failed: " << error.what() << std::endl;
        return 1;
    }
}
