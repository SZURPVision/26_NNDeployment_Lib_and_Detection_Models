#pragma once

#include "network_deployment_interface.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace MPT { struct SpeedStats; }

// 后端与后处理之间的参数：输出张量形状，以及预处理生成的原图坐标还原信息。
struct InferParam
{
    int origin_width = 0;
    int origin_height = 0;
    int out_tensor_cols = 0;
    int out_tensor_rows = 0;
    float scale = 1.0;
    int pad_x = 0;
    int pad_y = 0;
};

class OpenVINOEngine;
class TensorRTEngine;
class V5InfantryPostProcessor;
class V8InfantryPostProcessor;
class V8_21InfantryPostProcessor;
class LidarPostProcessor;
class RunePostProcessor;

// ==================== 主类定义 ====================
class YOLOModel
{
public:
    // 使用默认参数加载模型：async + openvino + GPU + auto_detect。
    YOLOModel(const std::string &model_path);

    // 直接指定模型路径、推理模式和部署后端。
    YOLOModel(const std::string &model_path,
              const std::string &infer_mode,
              const std::string &deployment_way,
              const std::string &device = "GPU",
              float confidence_threshold = 0.5f,
              const std::string &postprocessmode = "auto_detect",
              const DebugConfig &debugconfig = DebugConfig());
    // 从 JSON 的指定节点读取模型配置。
    YOLOModel(const JsonConfig &json_config,
              const DebugConfig &debugconfig = DebugConfig());
    YOLOModel() = delete;
    ~YOLOModel() noexcept;

    YOLOModel(const YOLOModel &) = delete;
    YOLOModel &operator=(const YOLOModel &) = delete;
    YOLOModel(YOLOModel &&) noexcept = default;
    YOLOModel &operator=(YOLOModel &&) noexcept = default;

    std::vector<NetArmorResult> netProcess(const cv::Mat &input_image, const int &my_color);
    std::vector<NetRuneResult> netProcess(const cv::Mat &input_image);

    // 判断当前模型是否使用装甲板后处理。
    bool supportsArmor() const;
    // 判断当前模型是否使用神符后处理。
    bool supportsRune() const;

    class InferenceEngine;
    class Postprocessor;

    enum class NetDeployWay
    {
        OpenVINO = 0,
        TensorRT = 1
    };
    enum class NetInferMode
    {
        sync = 0,
        async = 1,
        async4 = 2
    };
    enum class NetPostProcessMode
    {
        auto_detect = 0,
        v5infantry_fourpoints = 1,
        v8infantry_fourpoints = 2,
        lidar_fourpoints = 3,
        rune_fivepoints = 4,
        v8infantry_fourpoints_21 = 5
    };

    struct ModelConfig
    {
        std::string model_path;
        NetPostProcessMode postprocess_mode;
        NetInferMode infer_mode;
        NetDeployWay deployment_way;
        std::string device;
        float confidence_threshold = 0.5f;

        ModelConfig() = delete;
        ModelConfig(const std::string &path,
                    const NetInferMode &infermode = NetInferMode::async,
                    const NetDeployWay &deployway = NetDeployWay::OpenVINO,
                    const std::string &device = "GPU",
                    float confidence = 0.5f,
                    const NetPostProcessMode &postprocessmode = NetPostProcessMode::auto_detect)
            : model_path(path),
              postprocess_mode(postprocessmode),
              infer_mode(infermode),
              deployment_way(deployway),
              device(device),
              confidence_threshold(confidence)
        {}
    };

private:
    YOLOModel(const ModelConfig &modelconfig, const DebugConfig &debugconfig = DebugConfig());

    static float defaultNmsThreshold(NetPostProcessMode mode);

    ModelConfig m_model_config;
    DebugConfig m_debug_config;
    std::unique_ptr<MPT::SpeedStats> m_speed_stats;

    std::unique_ptr<InferenceEngine> m_inference_engine;
    std::unique_ptr<Postprocessor> m_postprocessor;
};
