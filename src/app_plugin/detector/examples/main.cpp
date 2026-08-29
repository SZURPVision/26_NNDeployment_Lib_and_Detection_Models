#include "DetectorRuntime.hpp"
#include "NNDetector.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace
{
struct VideoResult
{
    std::uint64_t frames = 0;
    std::filesystem::path output_path;
};

VideoResult runArmorDemo(const DetectorProjectPaths &paths,
                         const std::filesystem::path &output_dir,
                         const std::string &config_key,
                         const std::string &model_name,
                         const std::string &output_filename,
                         int my_color)
{
    NNDetector detector(
        NNDetector::Task::Armor,
        JsonConfig{paths.json_path.string(), config_key, paths.model_folder.string()});
    cv::VideoCapture video = openVideo(paths.armor_video, model_name + " 演示视频");
    const std::filesystem::path output_path = output_dir / output_filename;
    cv::VideoWriter writer = createVideoWriter(video, output_path);

    cv::Mat frame;
    std::uint64_t frame_count = 0;
    while (video.read(frame))
    {
        const std::vector<NetArmorResult> results =
            detector.detectArmor(frame, my_color);
        NNDetector::drawArmorResults(frame, results);
        writer.write(frame);
        ++frame_count;

        // cv::imshow(model_name + " result", frame);
        // if (cv::waitKey(1) == 27) break;
    }

    writer.release();
    video.release();
    return {frame_count, output_path};
}

VideoResult runRuneDemo(const DetectorProjectPaths &paths,
                        const std::filesystem::path &output_dir)
{
    NNDetector detector(
        NNDetector::Task::Rune,
        JsonConfig{paths.json_path.string(), "rune_detect", paths.model_folder.string()});
    cv::VideoCapture video = openVideo(paths.rune_video, "能量机关演示视频");
    const std::filesystem::path output_path = output_dir / "rune_result.mp4";
    cv::VideoWriter writer = createVideoWriter(video, output_path);

    cv::Mat frame;
    std::uint64_t frame_count = 0;
    while (video.read(frame))
    {
        const std::vector<NetRuneResult> results = detector.detectRune(frame);
        NNDetector::drawRuneResults(frame, results);
        writer.write(frame);
        ++frame_count;

        // cv::imshow("Rune result", frame);
        // if (cv::waitKey(1) == 27) break;
    }

    writer.release();
    video.release();
    return {frame_count, output_path};
}
} // namespace

int main(int argc, char **argv)
{
    try
    {
        const std::optional<std::filesystem::path> project_root =
            argc > 1 ? std::optional<std::filesystem::path>(argv[1]) : std::nullopt;
        const DetectorProjectPaths paths = resolveProjectPaths(project_root);
        const std::filesystem::path output_dir = resultDirectory("main");
        constexpr int my_color = 2;

        const VideoResult armor_v5 =
            runArmorDemo(paths, output_dir, "armor_v5", "Armor V5",
                         "armor_v5_result.mp4", my_color);
        const VideoResult armor_v8 =
            runArmorDemo(paths, output_dir, "armor_v8", "Armor V8",
                         "armor_v8_result.mp4", my_color);
        const VideoResult rune = runRuneDemo(paths, output_dir);

        std::cout << "Armor V5 frames: " << armor_v5.frames << '\n'
                  << "Armor V5 video: " << armor_v5.output_path << '\n'
                  << "Armor V8 frames: " << armor_v8.frames << '\n'
                  << "Armor V8 video: " << armor_v8.output_path << '\n'
                  << "Rune frames: " << rune.frames << '\n'
                  << "Rune video: " << rune.output_path << std::endl;
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "main failed: " << error.what() << std::endl;
        return 1;
    }
}
