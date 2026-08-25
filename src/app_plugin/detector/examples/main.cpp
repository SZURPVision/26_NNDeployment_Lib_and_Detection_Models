#include "DetectorRuntime.hpp"
#include "NNDetector.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <filesystem>
#include <iostream>
#include <optional>

int main(int argc, char **argv)
{
    try
    {
        const std::optional<std::filesystem::path> project_root =
            argc > 1 ? std::optional<std::filesystem::path>(argv[1]) : std::nullopt;
        const DetectorProjectPaths paths = resolveProjectPaths(project_root);
        const std::filesystem::path output_dir = resultDirectory("main");
        constexpr int my_color = 2;

        cv::VideoCapture armor_video = openVideo(paths.armor_video, "装甲板演示视频");
        cv::Mat armor_frame;
        if (!armor_video.read(armor_frame) || armor_frame.empty())
            throw std::runtime_error("装甲板演示视频没有可读取的首帧: " +
                                     paths.armor_video.string());

        NNDetector v5_detector(
            NNDetector::Task::Armor,
            JsonConfig{paths.json_path.string(),
                       "armor_detect_outpost",
                       paths.model_folder.string()});
        v5_detector.detectArmor(armor_frame, my_color);
        const std::vector<NetArmorResult> v5_results =
            v5_detector.detectArmor(armor_frame, my_color);
        std::cout << "V5 results: " << v5_results.size() << std::endl;
        for (const NetArmorResult &result : v5_results)
            std::cout << "class_id=" << result.armor_id
                      << " color_id=" << result.color_id
                      << " score=" << result.score << std::endl;

        NNDetector v8_detector(
            NNDetector::Task::Armor,
            JsonConfig{paths.json_path.string(),
                       "armor_detect_aim",
                       paths.model_folder.string()});
        const std::filesystem::path armor_output =
            output_dir / "armor_v8_result.mp4";
        cv::VideoWriter armor_writer = createVideoWriter(armor_video, armor_output);
        if (!armor_video.set(cv::CAP_PROP_POS_FRAMES, 0))
            throw std::runtime_error("无法将装甲板演示视频复位到首帧");

        std::uint64_t armor_frames = 0;
        while (armor_video.read(armor_frame))
        {
            const std::vector<NetArmorResult> results =
                v8_detector.detectArmor(armor_frame, my_color);
            NNDetector::drawArmorResults(armor_frame, results);
            armor_writer.write(armor_frame);
            ++armor_frames;

            // cv::imshow("Armor result", armor_frame);
            // if (cv::waitKey(1) == 27) break;
        }
        armor_writer.release();
        armor_video.release();

        NNDetector rune_detector(
            NNDetector::Task::Rune,
            JsonConfig{paths.json_path.string(),
                       "rune_detect",
                       paths.model_folder.string()});
        cv::VideoCapture rune_video = openVideo(paths.rune_video, "神符演示视频");
        const std::filesystem::path rune_output = output_dir / "rune_result.mp4";
        cv::VideoWriter rune_writer = createVideoWriter(rune_video, rune_output);

        cv::Mat rune_frame;
        std::uint64_t rune_frames = 0;
        while (rune_video.read(rune_frame))
        {
            const std::vector<NetRuneResult> results =
                rune_detector.detectRune(rune_frame);
            NNDetector::drawRuneResults(rune_frame, results);
            rune_writer.write(rune_frame);
            ++rune_frames;

            // cv::imshow("Rune result", rune_frame);
            // if (cv::waitKey(1) == 27) break;
        }
        rune_writer.release();
        rune_video.release();

        std::cout << "Armor frames: " << armor_frames << '\n'
                  << "Armor video: " << armor_output << '\n'
                  << "Rune frames: " << rune_frames << '\n'
                  << "Rune video: " << rune_output << std::endl;
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "main failed: " << error.what() << std::endl;
        return 1;
    }
}
