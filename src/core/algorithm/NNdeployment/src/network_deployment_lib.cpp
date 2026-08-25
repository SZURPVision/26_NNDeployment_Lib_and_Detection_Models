#include "network_deployment_lib.hpp"
#include "network_inference.hpp"
#include "network_performance.hpp"
#include "network_postprocess.hpp"

#ifndef NETLIB_WITH_TENSORRT
#define NETLIB_WITH_TENSORRT 0
#endif

#ifndef NETLIB_WITH_OPENVINO
#define NETLIB_WITH_OPENVINO 1
#endif

#if NETLIB_WITH_OPENVINO
#include "network_deployment_openvino.hpp"
#endif

#if NETLIB_WITH_TENSORRT
#include "network_deployment_tensorrt.hpp"
#endif

#include <opencv2/core/persistence.hpp>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>

// 总模块：YOLOModel 构造、部署/后处理选择以及完整推理入口。

// ==================== YOLOModel 实现 ====================

// namespace 中只包含字符串解析、枚举转换、配置读取和提示输出等局部工具，不用细看，跳过即可。
namespace
{
using NetDeployWay = YOLOModel::NetDeployWay;
using NetInferMode = YOLOModel::NetInferMode;
using NetPostProcessMode = YOLOModel::NetPostProcessMode;
using ModelConfig = YOLOModel::ModelConfig;

// 将字符串统一转换为小写。
std::string lowerString(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return value;
}

using PerformanceClock = std::chrono::steady_clock;

// 计算两个时间点之间的微秒数。
double elapsedMicroseconds(PerformanceClock::time_point start,
                           PerformanceClock::time_point stop)
{
    return std::chrono::duration<double, std::micro>(stop - start).count();
}

// 将字符串解析为推理模式。
NetInferMode parseInferMode(const std::string &value)
{
    const std::string mode = lowerString(value);
    if (mode.empty() || mode == "async")
        return NetInferMode::async;
    if (mode == "sync")
        return NetInferMode::sync;
    if (mode == "async4")
        return NetInferMode::async4;
    throw std::runtime_error("不支持的infer_mode: " + value);
}

// 将字符串解析为部署后端。
NetDeployWay parseDeployWay(const std::string &value)
{
    const std::string way = lowerString(value);
    if (way.empty() || way == "openvino")
        return NetDeployWay::OpenVINO;
    if (way == "tensorrt")
        return NetDeployWay::TensorRT;
    throw std::runtime_error("不支持的deploy_way: " + value);
}

// 将字符串解析为后处理模式。
NetPostProcessMode parsePostProcessMode(const std::string &value)
{
    const std::string mode = lowerString(value);
    if (mode.empty() || mode == "auto" || mode == "auto_detect")
        return NetPostProcessMode::auto_detect;
    if (mode == "v5infantry" || mode == "v5infantry_fourpoints")
        return NetPostProcessMode::v5infantry_fourpoints;
    if (mode == "v8infantry" || mode == "v8infantry_fourpoints")
        return NetPostProcessMode::v8infantry_fourpoints;
    if (mode == "v8infantry_21" || mode == "v8infantry_fourpoints_21")
        return NetPostProcessMode::v8infantry_fourpoints_21;
    if (mode == "lidar" || mode == "lidar_fourpoints")
        return NetPostProcessMode::lidar_fourpoints;
    if (mode == "rune" || mode == "rune_fivepoints")
        return NetPostProcessMode::rune_fivepoints;
    throw std::runtime_error("不支持的postprocess_mode: " + value);
}

// 将部署后端转换为可打印字符串。
std::string toString(NetDeployWay way)
{
    switch (way)
    {
    case NetDeployWay::OpenVINO:
        return "openvino";
    case NetDeployWay::TensorRT:
        return "tensorrt";
    default:
        return "error type";
    }
}

// 将推理模式转换为可打印字符串。
std::string toString(NetInferMode way)
{
    switch (way)
    {
    case NetInferMode::sync:
        return "sync";
    case NetInferMode::async:
        return "async";
    case NetInferMode::async4:
        return "async4";
    default:
        return "error type";
    }
}

// 将后处理模式转换为可打印字符串。
std::string toString(NetPostProcessMode way)
{
    switch (way)
    {
    case NetPostProcessMode::auto_detect:
        return "auto";
    case NetPostProcessMode::v5infantry_fourpoints:
        return "v5infantry";
    case NetPostProcessMode::v8infantry_fourpoints:
        return "v8infantry";
    case NetPostProcessMode::v8infantry_fourpoints_21:
        return "v8infantry_21";
    case NetPostProcessMode::lidar_fourpoints:
        return "lidar";
    case NetPostProcessMode::rune_fivepoints:
        return "rune";
    default:
        return "error type";
    }
}

// 打印当前模型配置及非标准组合提示。
void printCurrentModelInfo(const ModelConfig &model_config)
{
    std::cout << "兵种：" << toString(model_config.postprocess_mode) << " "
              << "推理模式：" << toString(model_config.infer_mode) << " "
              << "部署方式：" << toString(model_config.deployment_way) << std::endl;

    switch (model_config.postprocess_mode)
    {
    case NetPostProcessMode::auto_detect:
        break;
    case NetPostProcessMode::v5infantry_fourpoints:
        if ((model_config.infer_mode != NetInferMode::async && model_config.infer_mode != NetInferMode::async4) ||
            model_config.deployment_way != NetDeployWay::OpenVINO)
            std::cout << "注：标准情况下v5步兵应该使用" << toString(NetDeployWay::OpenVINO) << "部署，" << toString(NetInferMode::async) << "推理" << std::endl;
        break;
    case NetPostProcessMode::v8infantry_fourpoints:
    case NetPostProcessMode::v8infantry_fourpoints_21:
        if ((model_config.infer_mode != NetInferMode::async && model_config.infer_mode != NetInferMode::async4) ||
            model_config.deployment_way != NetDeployWay::OpenVINO)
            std::cout << "注：标准情况下v8步兵应该使用" << toString(NetDeployWay::OpenVINO) << "部署，" << toString(NetInferMode::async) << "推理" << std::endl;
        break;
    case NetPostProcessMode::lidar_fourpoints:
        if (model_config.infer_mode != NetInferMode::sync || model_config.deployment_way != NetDeployWay::TensorRT)
            std::cout << "注：标准情况下雷达应该使用" << toString(NetDeployWay::TensorRT) << "部署，" << toString(NetInferMode::sync) << "推理" << std::endl;
        break;
    case NetPostProcessMode::rune_fivepoints:
        if (model_config.infer_mode != NetInferMode::sync || model_config.deployment_way != NetDeployWay::OpenVINO)
            std::cout << "注：标准情况下打符应该使用" << toString(NetDeployWay::OpenVINO) << "部署，" << toString(NetInferMode::sync) << "推理" << std::endl;
        break;
    }
    std::cout << std::endl;
}

// 从 JSON 的指定节点读取内部模型配置。
ModelConfig loadModelConfigFromJson(const JsonConfig &json_config)
{
    cv::FileStorage config(json_config.json_path, cv::FileStorage::READ);
    if (!config.isOpened())
        throw std::runtime_error("无法打开模型配置文件: " + json_config.json_path);

    const cv::FileNode node = config[json_config.model_key];
    if (node.empty())
        throw std::runtime_error("模型配置中不存在键: " + json_config.model_key);

    if (json_config.model_folder.empty())
        throw std::runtime_error("传入的模型文件夹路径为空: " + json_config.model_key);

    const std::string xml_name = (std::string)node["xml"];
    if (xml_name.empty())
        throw std::runtime_error("模型配置缺少xml文件名: " + json_config.model_key);

    const std::string model_path =
        (std::filesystem::path(json_config.model_folder) / xml_name).lexically_normal().string();

    const NetInferMode infermode = parseInferMode((std::string)node["infer_mode"]);
    const NetDeployWay deployway = parseDeployWay((std::string)node["deploy_way"]);
    const NetPostProcessMode postprocessmode = parsePostProcessMode((std::string)node["postprocess_mode"]);

    std::string device = (std::string)node["device"];
    if (device.empty())
        device = "GPU";

    float confidence_threshold = 0.5f;
    if (!node["score_threshold"].empty())
        confidence_threshold = (float)node["score_threshold"];
    else if (!node["config_thresh"].empty())                // 兼容旧版本json配置文件
        confidence_threshold = (float)node["config_thresh"];

    return ModelConfig(model_path, infermode, deployway, device, confidence_threshold, postprocessmode);
}
} // namespace

YOLOModel::InferenceEngine::InferenceEngine(const ModelConfig &modelconfig, const DebugConfig &debugconfig) : m_model_config(modelconfig), m_debug_config(debugconfig) {}

const float *YOLOModel::InferenceEngine::infer(const cv::Mat &pre_processed_image)
{
    switch (m_model_config.infer_mode)
    {
    case NetInferMode::sync:
        return syncInfer(pre_processed_image);
    case NetInferMode::async:
        return asyncInfer(pre_processed_image);
    case NetInferMode::async4:
        return asyncInfer4(pre_processed_image);
    }
    throw std::runtime_error("不支持的推理模式");
}

float YOLOModel::defaultNmsThreshold(NetPostProcessMode mode)
{
    switch (mode)
    {
    case NetPostProcessMode::rune_fivepoints:
        return 100.f;
    case NetPostProcessMode::v5infantry_fourpoints:
    case NetPostProcessMode::v8infantry_fourpoints:
    case NetPostProcessMode::v8infantry_fourpoints_21:
    case NetPostProcessMode::lidar_fourpoints:
    case NetPostProcessMode::auto_detect:
    default:
        return 0.2f;
    }
}

YOLOModel::YOLOModel(const std::string &model_path)
    : YOLOModel(model_path,
                "async",
                "openvino",
                "GPU",
                0.5f,
                "auto_detect",
                DebugConfig())
{}

YOLOModel::YOLOModel(const std::string &model_path,
                     const std::string &infer_mode,
                     const std::string &deployment_way,
                     const std::string &device,
                     float confidence_threshold,
                     const std::string &postprocessmode,
                     const DebugConfig &debugconfig)
    : YOLOModel(ModelConfig(model_path,
                            parseInferMode(infer_mode),
                            parseDeployWay(deployment_way),
                            device,
                            confidence_threshold,
                            parsePostProcessMode(postprocessmode)),
                debugconfig)
{}

YOLOModel::YOLOModel(const JsonConfig &json_config,
                     const DebugConfig &debugconfig)
    : YOLOModel(loadModelConfigFromJson(json_config), debugconfig)
{}

YOLOModel::~YOLOModel() noexcept = default;

YOLOModel::YOLOModel(const ModelConfig &modelconfig, const DebugConfig &debugconfig) : m_model_config(modelconfig), m_debug_config(debugconfig)
{
    if (m_debug_config.calculate_speed_info)
        m_speed_stats = std::make_unique<MPT::SpeedStats>();

    // 根据部署方式选择openvino/tensorrt，目前仅雷达会使用tensorrt
    switch (m_model_config.deployment_way)
    {
    case NetDeployWay::OpenVINO:
#if NETLIB_WITH_OPENVINO
        m_inference_engine = std::make_unique<OpenVINOEngine>(m_model_config, m_debug_config);
#else
        throw std::runtime_error("OpenVINO support is disabled at build time");
#endif
        break;
    case NetDeployWay::TensorRT:
#if NETLIB_WITH_TENSORRT
        m_inference_engine = std::make_unique<TensorRTEngine>(m_model_config, m_debug_config);
#else
        throw std::runtime_error("TensorRT support is disabled at build time");
#endif
        break;
    }

    // 根据后处理模式决定后处理模块，大部分情况下使用auto即可，除非模型结构改了
    switch (m_model_config.postprocess_mode)
    {
    case NetPostProcessMode::auto_detect:
        {
            const int rows = m_inference_engine->m_infer_param.out_tensor_rows;
            const int cols = m_inference_engine->m_infer_param.out_tensor_cols;
            if (rows == 25200 && cols == 22)
            {
                m_model_config.postprocess_mode = NetPostProcessMode::v5infantry_fourpoints;
                m_postprocessor = std::make_unique<V5InfantryPostProcessor>(
                    m_model_config,
                    defaultNmsThreshold(m_model_config.postprocess_mode));
            }
            else if (rows == 25 && cols == 6300)
            {
                m_model_config.postprocess_mode = NetPostProcessMode::v8infantry_fourpoints;
                m_postprocessor = std::make_unique<V8InfantryPostProcessor>(
                    m_model_config,
                    defaultNmsThreshold(m_model_config.postprocess_mode));
            }
            else if (rows == 21 && cols == 6300)
            {
                m_model_config.postprocess_mode = NetPostProcessMode::v8infantry_fourpoints_21;
                m_postprocessor = std::make_unique<V8_21InfantryPostProcessor>(
                    m_model_config,
                    defaultNmsThreshold(m_model_config.postprocess_mode));
            }
            else if (rows == 18 && cols == 6300)
            {
                m_model_config.postprocess_mode = NetPostProcessMode::rune_fivepoints;
                m_postprocessor = std::make_unique<RunePostProcessor>(
                    m_model_config,
                    defaultNmsThreshold(m_model_config.postprocess_mode));
            }
            else
            {
                throw std::runtime_error("无法自动识别模型输出格式: [" + std::to_string(rows) + ", " + std::to_string(cols) + "]");
            }
        }
        break;
    case NetPostProcessMode::v5infantry_fourpoints:
        m_postprocessor = std::make_unique<V5InfantryPostProcessor>(
            m_model_config,
            defaultNmsThreshold(m_model_config.postprocess_mode));
        break;
    case NetPostProcessMode::v8infantry_fourpoints:
        m_postprocessor = std::make_unique<V8InfantryPostProcessor>(
            m_model_config,
            defaultNmsThreshold(m_model_config.postprocess_mode));
        break;
    case NetPostProcessMode::v8infantry_fourpoints_21:
        m_postprocessor = std::make_unique<V8_21InfantryPostProcessor>(
            m_model_config,
            defaultNmsThreshold(m_model_config.postprocess_mode));
        break;
    case NetPostProcessMode::lidar_fourpoints:
        m_postprocessor = std::make_unique<LidarPostProcessor>(
            m_model_config,
            defaultNmsThreshold(m_model_config.postprocess_mode));
        break;
    case NetPostProcessMode::rune_fivepoints:
        m_postprocessor = std::make_unique<RunePostProcessor>(
            m_model_config,
            defaultNmsThreshold(m_model_config.postprocess_mode));
        break;
    }

    // 调用namespace中的函数打印使用的模型信息，这部分不需要debug开启也打印
    printCurrentModelInfo(m_model_config);
}

//  完整的网络识别框架——主要流程：传入图片->预处理->推理->后处理->输出结果
std::vector<NetArmorResult> YOLOModel::netProcess(const cv::Mat &input_image, const int &my_color)
{
    if (!(m_model_config.postprocess_mode == NetPostProcessMode::auto_detect ||
          m_model_config.postprocess_mode == NetPostProcessMode::v5infantry_fourpoints ||
          m_model_config.postprocess_mode == NetPostProcessMode::v8infantry_fourpoints ||
          m_model_config.postprocess_mode == NetPostProcessMode::v8infantry_fourpoints_21 ||
          m_model_config.postprocess_mode == NetPostProcessMode::lidar_fourpoints))
    {
        std::cerr << "该模型不支持四点装甲板推理" << std::endl;
        return {};
    }

    // debug 的测速选项开启时进行各部分的测速。
    const bool measure = m_speed_stats != nullptr;
    PerformanceClock::time_point phase_start{};
    double preprocess_us = 0.0;
    double inference_us = 0.0;
    double postprocess_us = 0.0;

    // 预处理
    if (measure)
        phase_start = PerformanceClock::now();
    cv::Mat pre_processed_mat = m_inference_engine->preProcessImage(input_image);
    if (measure)
        preprocess_us = elapsedMicroseconds(phase_start, PerformanceClock::now());
    if (pre_processed_mat.empty())
        return {};

    // 推理
    if (measure)
        phase_start = PerformanceClock::now();
    const float *infer_output = m_inference_engine->infer(pre_processed_mat);
    if (measure)
        inference_us = elapsedMicroseconds(phase_start, PerformanceClock::now());

    // 后处理
    if (measure)
        phase_start = PerformanceClock::now();
    std::vector<NetArmorResult> results = m_postprocessor->postProcessArmorMat(
        infer_output, m_inference_engine->m_infer_param, my_color);
    if (measure)
        postprocess_us = elapsedMicroseconds(phase_start, PerformanceClock::now());

    // 输出计时结果
    if (measure)
    {
        m_speed_stats->update(preprocess_us, inference_us, postprocess_us);
        m_speed_stats->printCurrentStats();
    }
    return results;
}

std::vector<NetRuneResult> YOLOModel::netProcess(const cv::Mat &input_image)
{
    if (m_model_config.postprocess_mode != NetPostProcessMode::rune_fivepoints)
    {
        std::cerr << "该模型不支持五点符叶推理" << std::endl;
        return {};
    }

    const bool measure = m_speed_stats != nullptr;
    PerformanceClock::time_point phase_start{};
    double preprocess_us = 0.0;
    double inference_us = 0.0;
    double postprocess_us = 0.0;

    // 预处理
    if (measure)
        phase_start = PerformanceClock::now();
    cv::Mat pre_processed_mat = m_inference_engine->preProcessImage(input_image);
    if (measure)
        preprocess_us = elapsedMicroseconds(phase_start, PerformanceClock::now());
    if (pre_processed_mat.empty())
        return {};

    // 推理
    if (measure)
        phase_start = PerformanceClock::now();
    const float *infer_output = m_inference_engine->infer(pre_processed_mat);
    if (measure)
        inference_us = elapsedMicroseconds(phase_start, PerformanceClock::now());

    // 后处理
    if (measure)
        phase_start = PerformanceClock::now();
    std::vector<NetRuneResult> results = m_postprocessor->postProcessRuneMat(
        infer_output, m_inference_engine->m_infer_param);
    if (measure)
        postprocess_us = elapsedMicroseconds(phase_start, PerformanceClock::now());

    // 输出计时结果
    if (measure)
    {
        m_speed_stats->update(preprocess_us, inference_us, postprocess_us);
        m_speed_stats->printCurrentStats();
    }
    return results;
}

bool YOLOModel::supportsArmor() const
{
    return m_model_config.postprocess_mode == NetPostProcessMode::v5infantry_fourpoints ||
           m_model_config.postprocess_mode == NetPostProcessMode::v8infantry_fourpoints ||
           m_model_config.postprocess_mode == NetPostProcessMode::v8infantry_fourpoints_21 ||
           m_model_config.postprocess_mode == NetPostProcessMode::lidar_fourpoints;
}

bool YOLOModel::supportsRune() const
{
    return m_model_config.postprocess_mode == NetPostProcessMode::rune_fivepoints;
}
