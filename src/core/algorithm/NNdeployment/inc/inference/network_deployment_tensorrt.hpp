#pragma once

#ifndef NETLIB_WITH_TENSORRT
#define NETLIB_WITH_TENSORRT 0
#endif

#if NETLIB_WITH_TENSORRT

#include "network_inference.hpp"

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvInferRuntimeCommon.h>
#include <cuda_runtime_api.h>

#include <fstream>
#include <iostream>

// logger是TensorRT必要的类，创建运行时环境的报错信息只能通过logger传递
class Logger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char *msg) noexcept override
    {
        if (severity != Severity::kINFO) // 屏蔽了所有的INFO等级信息，只显示警告和错误信息
        {
            std::cout << msg << std::endl;
        }
    }
};

class TensorRTEngine : public YOLOModel::InferenceEngine
{
public:
    TensorRTEngine(const YOLOModel::ModelConfig &modelconfig, const DebugConfig &debugconfig);
    ~TensorRTEngine();
    int inputWidth() const override;
    int inputHeight() const override;
    const float *syncInfer(const cv::Mat &pre_processed_image) override;
    const float *asyncInfer(const cv::Mat &pre_processed_image) override;

private:
    // TensorRT相关成员
    Logger m_logger;                                                  // 日志记录器
    std::unique_ptr<nvinfer1::IRuntime> m_runtime = nullptr;          // TensorRT运行时对象
    std::unique_ptr<nvinfer1::ICudaEngine> m_engine = nullptr;        // TensorRT推理引擎
    std::unique_ptr<nvinfer1::IExecutionContext> m_context = nullptr; // 执行上下文
    void *m_buffers[2] = {nullptr, nullptr};                          // GPU显存指针数组[输入, 输出]
    const int m_input_index = 0;                                      // 输入缓冲区索引
    const int m_output_index = 1;                                     // 输出缓冲区索引
    cudaStream_t m_cuda_stream = nullptr;                             // CUDA流用于异步操作

    // 数据处理成员
    cv::Mat m_blob;                // 预处理后的输入张量
    void *m_blob_pinned = nullptr; // 页锁定内存中的输入数据
    float *m_rst = nullptr;        // 页锁定内存中的原始推理结果
    int m_target_width = 0;        // 目标输入图像宽度
    int m_target_height = 0;       // 模型输入高度
    int m_output_anchors = 0;      // 输出张量的锚框数量
    size_t m_input_volume = 0;     // 输入数据量(NCHW形式)
    size_t m_output_volume = 0;    // 输出数据量(NHW)
};

#endif
