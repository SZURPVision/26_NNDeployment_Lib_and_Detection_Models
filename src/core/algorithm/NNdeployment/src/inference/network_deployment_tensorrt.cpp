#include "network_deployment_tensorrt.hpp"

#if NNDEPLOYMENT_WITH_TENSORRT

#include <cmath>
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

namespace
{
void printTensorInfo(const nvinfer1::ICudaEngine *engine)
{
    if (!engine)
    {
        std::cerr << "错误: 引擎未初始化" << std::endl;
        return;
    }

    std::cout << "=== 模型张量信息 ===" << '\n';
    for (int i = 0; i < 2; ++i)
    {
        const char *name = engine->getIOTensorName(i);
        nvinfer1::TensorIOMode ioMode = engine->getTensorIOMode(name);
        nvinfer1::Dims dims = engine->getTensorShape(name);
        nvinfer1::DataType dtype = engine->getTensorDataType(name);

        std::cout << "张量 " << i << ":" << '\n';
        std::cout << "  名称: " << name << '\n';
        std::cout << "  类型: " << (ioMode == nvinfer1::TensorIOMode::kINPUT ? "输入" : "输出") << '\n';
        std::cout << "  形状: [";
        for (int j = 0; j < dims.nbDims; ++j)
        {
            std::cout << dims.d[j];
            if (j < dims.nbDims - 1)
                std::cout << ", ";
        }
        std::cout << "]" << '\n';

        std::cout << "  数据类型: ";
        switch (dtype)
        {
        case nvinfer1::DataType::kFLOAT:
            std::cout << "float32";
            break;
        case nvinfer1::DataType::kHALF:
            std::cout << "float16";
            break;
        case nvinfer1::DataType::kINT8:
            std::cout << "int8";
            break;
        case nvinfer1::DataType::kINT32:
            std::cout << "int32";
            break;
        case nvinfer1::DataType::kBOOL:
            std::cout << "bool";
            break;
        default:
            std::cout << "unknown";
        }
        std::cout << '\n' << '\n';
    }
    std::cout.flush();
}
} // namespace

TensorRTEngine::TensorRTEngine(const YOLOModel::ModelConfig &model_config, const DebugConfig &debug_config) : InferenceEngine(model_config, debug_config)
{
    // ==================== 初始化推理引擎 ====================
    // 读取引擎文件
    std::ifstream engine_file(m_model_config.model_path, std::ios::binary);
    if (!engine_file)
        throw std::runtime_error("找不到 TensorRT 模型: " + m_model_config.model_path);

    // 获取文件大小
    engine_file.seekg(0, std::ios::end);
    const size_t file_size = engine_file.tellg();
    engine_file.seekg(0, std::ios::beg);

    // 以二进制文件形式加载模型内容
    std::vector<char> engine_data(file_size);
    if (!engine_file.read(engine_data.data(), file_size))
        throw std::runtime_error("读取 TensorRT 模型失败: " + m_model_config.model_path);

    engine_file.close();

    // 创建TensorRT运行时对象
    m_runtime.reset(nvinfer1::createInferRuntime(m_logger));
    if (!m_runtime)
        throw std::runtime_error("无法创建 TensorRT 运行时");

    // 反序列化引擎数据,生成推理引擎
    m_engine.reset(m_runtime->deserializeCudaEngine(engine_data.data(), file_size));
    if (!m_engine)
        throw std::runtime_error("无法反序列化 TensorRT 模型: " + m_model_config.model_path);

    // 创建执行上下文
    m_context.reset(m_engine->createExecutionContext());
    if (!m_context)
        throw std::runtime_error("无法创建 TensorRT 执行上下文");

    // 写入输入输出张量大小
    m_target_width = m_engine->getTensorShape(m_engine->getIOTensorName(m_input_index)).d[3];
    m_target_height = m_engine->getTensorShape(m_engine->getIOTensorName(m_input_index)).d[2];

    m_output_anchors = m_engine->getTensorShape(m_engine->getIOTensorName(m_output_index)).d[2];

    m_infer_param.out_tensor_rows = m_engine->getTensorShape(m_engine->getIOTensorName(m_output_index)).d[1];
    m_infer_param.out_tensor_cols = m_engine->getTensorShape(m_engine->getIOTensorName(m_output_index)).d[2];

    m_input_volume = 1 * sizeof(float);
    for (int i = 0; i < m_engine->getTensorShape(m_engine->getIOTensorName(m_input_index)).nbDims; i++)
    {
        m_input_volume *= m_engine->getTensorShape(m_engine->getIOTensorName(m_input_index)).d[i];
    }
    m_output_volume = 1 * sizeof(float);
    for (int i = 0; i < m_engine->getTensorShape(m_engine->getIOTensorName(m_output_index)).nbDims; i++)
    {
        m_output_volume *= m_engine->getTensorShape(m_engine->getIOTensorName(m_output_index)).d[i];
    }

    // ==================== 分配推理缓冲区 ====================
    // 获取输入张量的形状并设置
    nvinfer1::Dims inputDims = m_engine->getTensorShape("images");
    m_context->setInputShape("images", inputDims);

    // 分配GPU显存用于输入输出
    cudaMalloc(&m_buffers[m_input_index], m_input_volume);
    cudaMalloc(&m_buffers[m_output_index], m_output_volume);

    // 绑定显存地址到TensorRT上下文
    m_context->setTensorAddress("images", m_buffers[m_input_index]);
    m_context->setTensorAddress("output0", m_buffers[m_output_index]);

    // 分配页锁定内存用于存储输入输出数据（提高传输效率）
    cudaMallocHost(reinterpret_cast<void **>(&m_rst), m_output_volume);
    cudaMallocHost(&m_blob_pinned, m_input_volume);

    // 创建CUDA流用于异步操作
    cudaStreamCreate(&m_cuda_stream);

    if (m_debug_config.print_debug_info)
        printTensorInfo(m_engine.get());
}

int TensorRTEngine::inputWidth() const
{
    return m_target_width;
}

int TensorRTEngine::inputHeight() const
{
    return m_target_height;
}

const float *TensorRTEngine::syncInfer(const cv::Mat &pre_processed_image)
{
    cv::dnn::blobFromImage(pre_processed_image, m_blob, 1 / 255.0,
                           cv::Size(m_target_width, m_target_height),
                           cv::Scalar(0, 0, 0), true, false, CV_32F);

    memcpy(m_blob_pinned, m_blob.data, m_blob.total() * m_blob.elemSize());

    cudaMemcpyAsync(m_buffers[m_input_index], m_blob_pinned,
                    m_input_volume, cudaMemcpyHostToDevice, m_cuda_stream);

    m_context->enqueueV3(m_cuda_stream);

    cudaMemcpyAsync(m_rst, m_buffers[m_output_index],
                    m_output_volume, cudaMemcpyDeviceToHost, m_cuda_stream);

    cudaStreamSynchronize(m_cuda_stream);

    return m_rst;
}

const float *TensorRTEngine::asyncInfer(const cv::Mat &pre_processed_image)
{
    cv::dnn::blobFromImage(pre_processed_image, m_blob, 1 / 255.0,
                           cv::Size(m_target_width, m_target_height),
                           cv::Scalar(0, 0, 0), true, false, CV_32F);

    memcpy(m_blob_pinned, m_blob.data, m_blob.total() * m_blob.elemSize());

    cudaMemcpyAsync(m_buffers[m_input_index], m_blob_pinned,
                    m_input_volume, cudaMemcpyHostToDevice, m_cuda_stream);

    m_context->enqueueV3(m_cuda_stream);

    cudaMemcpyAsync(m_rst, m_buffers[m_output_index],
                    m_output_volume, cudaMemcpyDeviceToHost, m_cuda_stream);

    cudaStreamSynchronize(m_cuda_stream);

    return m_rst;
}

TensorRTEngine::~TensorRTEngine()
{
    if (m_cuda_stream)
        cudaStreamDestroy(m_cuda_stream);
    if (m_blob_pinned)
        cudaFreeHost(m_blob_pinned);
    if (m_rst)
        cudaFreeHost(m_rst);
    if (m_buffers[m_input_index])
        cudaFree(m_buffers[m_input_index]);
    if (m_buffers[m_output_index])
        cudaFree(m_buffers[m_output_index]);
}

#endif
