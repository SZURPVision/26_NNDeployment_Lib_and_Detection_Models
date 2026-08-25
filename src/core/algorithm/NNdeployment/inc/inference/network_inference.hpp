#pragma once

#include "network_deployment_lib.hpp"

#include <opencv2/core.hpp>

class YOLOModel::InferenceEngine
{
public:
    InferenceEngine(const ModelConfig &modelconfig, const DebugConfig &debugconfig);
    virtual ~InferenceEngine() = default;

    virtual int inputWidth() const = 0;
    virtual int inputHeight() const = 0;
    // 返回供当前推理请求写入的预处理缓冲区。
    virtual cv::Mat acquirePreprocessBuffer() { return {}; }
    // 使用当前后端的输入尺寸、坐标参数和请求缓冲区预处理图像。
    cv::Mat preProcessImage(const cv::Mat &origin_image);
    const float *infer(const cv::Mat &pre_processed_image);
    virtual const float *syncInfer(const cv::Mat &pre_processed_image) = 0;
    virtual const float *asyncInfer(const cv::Mat &pre_processed_image) = 0;
    virtual const float *asyncInfer4(const cv::Mat &pre_processed_image) { throw std::logic_error("async4 infer not supported"); }

    InferParam m_infer_param;

protected:
    ModelConfig m_model_config;
    DebugConfig m_debug_config;
};
