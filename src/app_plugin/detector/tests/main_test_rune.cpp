#include "DetectorRuntime.hpp"
#include "NNDetector.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>

int main(int argc, char **argv)
{
    try
    {
        const std::optional<std::filesystem::path> project_root =
            argc > 1 ? std::optional<std::filesystem::path>(argv[1]) : std::nullopt;
        const DetectorProjectPaths paths = resolveProjectPaths(project_root);
        NNDetector detector(
            NNDetector::Task::Rune,
            JsonConfig{paths.json_path.string(),
                       "rune_detect",
                       paths.model_folder.string()});

        cv::VideoCapture video = openVideo(paths.rune_video, "神符测试视频");
        const std::filesystem::path output =
            resultDirectory("rune") / "rune_result.mp4";
        cv::VideoWriter writer = createVideoWriter(video, output);

        cv::Mat frame;
        std::uint64_t input_frames = 0;
        std::uint64_t output_frames = 0;
        std::uint64_t detection_count = 0;
        while (video.read(frame))
        {
            ++input_frames;
            const std::vector<NetRuneResult> results = detector.detectRune(frame);
            detection_count += results.size();
            NNDetector::drawRuneResults(frame, results);
            writer.write(frame);
            ++output_frames;

            // cv::imshow("Rune result", frame);
            // if (cv::waitKey(1) == 27) break;
        }
        writer.release();
        video.release();

        std::cout << "Input frames: " << input_frames << '\n'
                  << "Output frames: " << output_frames << '\n'
                  << "Detections: " << detection_count << '\n'
                  << "Rune video: " << output << std::endl;
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "main_test_rune failed: " << error.what() << std::endl;
        return 1;
    }
}
