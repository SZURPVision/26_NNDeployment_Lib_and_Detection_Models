#pragma once

#include "network_inference.hpp"

#include <openvino/openvino.hpp>

#include <array>
#include <cstddef>

class OpenVINOEngine : public YOLOModel::InferenceEngine
{
public:
    OpenVINOEngine(const YOLOModel::ModelConfig &model_config, const DebugConfig &debug_config);
    ~OpenVINOEngine() = default;

    int inputWidth() const override;
    int inputHeight() const override;
    // 根据推理模式返回对应 OpenVINO 请求的预处理缓冲区。
    cv::Mat acquirePreprocessBuffer() override;
    const float *syncInfer(const cv::Mat &pre_processed_image) override;
    const float *asyncInfer(const cv::Mat &pre_processed_image) override;
    const float *asyncInfer4(const cv::Mat &pre_processed_image) override;

private:
    ov::CompiledModel m_compiled_model;
    ov::Shape m_input_shape;
    ov::Shape m_output_shape;
    ov::InferRequest m_sync_infer_request;
    std::array<ov::InferRequest, 2> m_async_infer_requests;
    std::array<cv::Mat, 2> m_async_preprocess_buffer;
    std::size_t m_async_submit_index = 0;
    std::size_t m_async_ready_index = 0;
    std::array<ov::InferRequest, 4> m_async4_infer_requests;
    std::array<cv::Mat, 4> m_async4_preprocess_buffer;
    std::size_t m_async4_submit_index = 0;
    std::size_t m_async4_ready_index = 0;

    void asyncStartup();
    void async4Startup();
};
