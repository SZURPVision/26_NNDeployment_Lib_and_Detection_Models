#include "network_deployment_openvino.hpp"

#include <opencv2/imgproc.hpp>

#include <cmath>
#include <iostream>

// ==================== 推理引擎模块 ====================
// OpenVINO 部署模块：模型读取、预处理配置、设备编译和请求池创建。
// TensorRT 部署实现独立保存在 network_deployment_tensorrt.cpp/.hpp，并由编译开关控制。

namespace
{
// 检查 OpenVINO 可用设备中是否存在指定设备或其子设备。
bool hasOpenVINODevice(const std::vector<std::string> &devices, const std::string &target)
{
    return std::any_of(devices.begin(), devices.end(), [&](const std::string &device)
    {
        return device.compare(0, target.size(), target) == 0;
    });
}

std::string toString(YOLOModel::NetPostProcessMode way)
{
    switch (way)
    {
    case YOLOModel::NetPostProcessMode::auto_detect:
        return "auto";
    case YOLOModel::NetPostProcessMode::v5infantry_fourpoints:
        return "v5infantry";
    case YOLOModel::NetPostProcessMode::v8infantry_fourpoints:
        return "v8infantry";
    case YOLOModel::NetPostProcessMode::v8infantry_fourpoints_21:
        return "v8infantry_21";
    case YOLOModel::NetPostProcessMode::lidar_fourpoints:
        return "lidar";
    case YOLOModel::NetPostProcessMode::rune_fivepoints:
        return "rune";
    default:
        return "error type";
    }
}

void printTensorInfo(const ov::Tensor &tensor, const std::string &name = "Tensor")
{
    std::cout << name << " 张量信息:" << "\n";
    std::cout << "  形状: " << tensor.get_shape() << "\n";
    std::cout << "  类别: " << tensor.get_element_type() << "\n";
    std::cout << "  元素量: " << tensor.get_size() << " elements" << std::endl;
}
} // namespace

// ==================== OpenVINO实现 ====================

OpenVINOEngine::OpenVINOEngine(const YOLOModel::ModelConfig &modelconfig, const DebugConfig &debugconfig) : InferenceEngine(modelconfig, debugconfig)
{
    // 1. 读取模型
    ov::Core core;                                                                 // OpenVINO核心对象
    std::shared_ptr<ov::Model> model = core.read_model(m_model_config.model_path); // 加载模型源文件
    ov::preprocess::PrePostProcessor ppp(model);                                   // ppp用于自动化部分预处理和后处理流程

    // 2. 设置自动化的参数
    // 设置数据类型为uchar255，输入张量为格式为“批次（N）高度（H）宽度（W）通道（C）”，输入颜色为BGR输入
    ppp.input().tensor().set_element_type(ov::element::u8).set_layout("NHWC").set_color_format(ov::preprocess::ColorFormat::BGR);
    // 设置预处理操作为 uchar->float32，BGR->RGB，归一化（所有像素÷255）
    ppp.input().preprocess().convert_element_type(ov::element::f32).convert_color(ov::preprocess::ColorFormat::RGB).scale(255.f);
    // 设置模型预期格式为NCHW，接收到输入张量时将NHWC->NCHW（可选,用于告知——在配置文件中未规定输入格式或与配置文件不匹配时生效）
    ppp.input().model().set_layout("NCHW");
    // 设置输出张量的数据类型为float32（可选，用于告知）
    ppp.output().tensor().set_element_type(ov::element::f32);

    // 保存输入输出张量信息
    if (model->inputs().empty() || model->outputs().empty())
    {
        std::cerr << "模型没有正确的输入或输出信息，请检查模型是否正确" << std::endl;
        return;
    }
    m_input_shape = model->inputs()[0].get_shape();
    m_output_shape = model->outputs()[0].get_shape();
    //NHWC，1是H高度rows，2是W宽度cols
    m_infer_param.out_tensor_rows = model->outputs()[0].get_shape()[1];
    m_infer_param.out_tensor_cols = model->outputs()[0].get_shape()[2];

    // 3. 输出模型信息，检查输入输出张量是否有误（这里的信息不是张量来的，是模型来的，用不了printTensorInfo）
    if (m_debug_config.print_debug_info)
    {
        // 获取模型输入信息
        ov::OutputVector inputs = model->inputs();
        const std::vector<std::string> available_devices = core.get_available_devices();

        ov::element::Type input_type = inputs[0].get_element_type();
        std::cout << toString(modelconfig.postprocess_mode) << "网络输入张量形状：" << m_input_shape << '\n';
        std::cout << toString(modelconfig.postprocess_mode) << "网络输入张量类别" << input_type << '\n';
        std::cout << toString(modelconfig.postprocess_mode) << "网络输出张量形状" << m_output_shape << '\n';
        std::cout << "OpenVINO可用设备:";

        bool has_available_target = false;
        for (const char *target : {"CPU", "GPU", "NPU"})
        {
            if (!hasOpenVINODevice(available_devices, target))
                continue;

            if (has_available_target)
                std::cout << ", ";
            std::cout << target;
            has_available_target = true;
        }

        if (!has_available_target)
            std::cout << "无";
        std::cout << std::endl;
    }

    // 4. 配置编译选项并编译模型——同步模式加强单帧延时，异步模式加强吞吐
    ov::AnyMap compile_config = {};
    const ov::hint::PerformanceMode performance_mode =
        (m_model_config.infer_mode == YOLOModel::NetInferMode::sync)
            ? ov::hint::PerformanceMode::LATENCY
            : ov::hint::PerformanceMode::THROUGHPUT;
    compile_config[ov::hint::performance_mode.name()] = performance_mode;
    if (m_model_config.device == "GPU")
    {
        compile_config["GPU_ENABLE_LOOP_UNROLLING"] = "NO"; // GPU优化配置
        compile_config["GPU_HOST_TASK_PRIORITY"] = "HIGH";  // 优先处理gpu相关请求
    }

    // 编译模型
    m_compiled_model = core.compile_model(ppp.build(), m_model_config.device, compile_config);
    if (m_debug_config.print_debug_info)
        std::cout << "已完成模型编译" << '\n'
                  << "使用设备" << m_model_config.device << std::endl;

    // 5. 根据推理模式创建推理请求
    if (m_model_config.infer_mode == YOLOModel::NetInferMode::sync)
    {
        m_sync_infer_request = m_compiled_model.create_infer_request();
    }
    else if (m_model_config.infer_mode == YOLOModel::NetInferMode::async)
    {
        // 异步推理速度较快时，request 可能仍在读取输入，而局部预处理结果已经析构。
        // 因此 async 模式使用 2 个固定槽位：每个槽位绑定自己的 request 和成员级预处理缓冲区。
        for (std::size_t i = 0; i < m_async_infer_requests.size(); ++i)
        {
            m_async_infer_requests[i] = m_compiled_model.create_infer_request();
            m_async_preprocess_buffer[i] = cv::Mat(m_input_shape[2], m_input_shape[3], CV_8UC3, cv::Scalar(124, 124, 124));
        }
        m_async_submit_index = 1;
        m_async_ready_index = 0;

        // 6. 创建符合input_tensor要求的灰色图像，启动异步推理的第一步，后续的第一张图片舍弃
        asyncStartup();
    }
    else if (m_model_config.infer_mode == YOLOModel::NetInferMode::async4)
    {
        // 与 2 路异步相同，4 路异步也需要让每个 request 绑定独立的成员级预处理缓冲区，避免推理尚未完成读取时，局部预处理图像提前释放。
        for (std::size_t i = 0; i < m_async4_infer_requests.size(); ++i)
        {
            m_async4_infer_requests[i] = m_compiled_model.create_infer_request();
            m_async4_preprocess_buffer[i] = cv::Mat(m_input_shape[2], m_input_shape[3], CV_8UC3, cv::Scalar(124, 124, 124));
        }

        // 4 路异步流水线预热 3 个请求，之后进入稳定的 4 request 并发。
        async4Startup();
    }

    if (m_debug_config.print_debug_info)
        std::cout << "推理请求已创建" << std::endl;
}

// ==================== OpenVINO预处理与推理实现 ====================
void OpenVINOEngine::asyncStartup()
{
    cv::Mat &startup_img = m_async_preprocess_buffer[m_async_ready_index];
    startup_img.setTo(cv::Scalar(128, 128, 128)); // BGR格式的灰色(128,128,128)
    ov::Tensor input_tensor = ov::Tensor(ov::element::u8, {1, (size_t)startup_img.rows, (size_t)startup_img.cols, 3}, startup_img.data);
    // 设置输入并启动异步推理
    m_async_infer_requests[m_async_ready_index].set_input_tensor(input_tensor);
    m_async_infer_requests[m_async_ready_index].start_async();
}

void OpenVINOEngine::async4Startup()
{
    // 4 路异步流水线需要预热 3 个请求，之后第 4 个请求开始送入真实帧。
    for (std::size_t i = 0; i < 3; ++i)
    {
        cv::Mat &startup_img = m_async4_preprocess_buffer[i];
        startup_img.setTo(cv::Scalar(128, 128, 128)); // BGR格式的灰色(128,128,128)
        ov::Tensor input_tensor = ov::Tensor(ov::element::u8, {1, (size_t)startup_img.rows, (size_t)startup_img.cols, 3}, startup_img.data);
        m_async4_infer_requests[i].set_input_tensor(input_tensor);
        m_async4_infer_requests[i].start_async();
    }

    m_async4_submit_index = 3;
    m_async4_ready_index = 0;
}

int OpenVINOEngine::inputWidth() const
{
    return static_cast<int>(m_input_shape[3]);
}

int OpenVINOEngine::inputHeight() const
{
    return static_cast<int>(m_input_shape[2]);
}

// 返回和推理模式匹配的缓冲区（不同推理模式不共用）
cv::Mat OpenVINOEngine::acquirePreprocessBuffer()
{
    if (m_model_config.infer_mode == YOLOModel::NetInferMode::async)
        return m_async_preprocess_buffer[m_async_submit_index];
    if (m_model_config.infer_mode == YOLOModel::NetInferMode::async4)
        return m_async4_preprocess_buffer[m_async4_submit_index];
    return {};
}

// 同步推理
const float* OpenVINOEngine::syncInfer(const cv::Mat &pre_processed_image)
{
    // Tensor构造方法：先指定类型，再指定形状，形状为NHWC（预处理之前的图像）
    // 由于mat矩阵必然是uchar，所以上面初始化的时候定义了图像预处理为u8，设置tensor的时候ppp会自动把u8改为float32
    ov::Tensor input_tensor = ov::Tensor(ov::element::u8, {1, (size_t)pre_processed_image.rows, (size_t)pre_processed_image.cols, 3}, pre_processed_image.data);

    // 设置输入并启动同步推理
    m_sync_infer_request.set_input_tensor(input_tensor);
    m_sync_infer_request.infer();

    ov::Tensor output_tensor = m_sync_infer_request.get_output_tensor();

    const float *output_data_ptr = output_tensor.data<const float>();
    // 以yolov5armor输出为例：output_shape=[1,25200,22]，可以理解为返回了一个数组头
    return output_data_ptr;
}

// 异步双缓冲
const float *OpenVINOEngine::asyncInfer(const cv::Mat &pre_processed_image)
{
    // Tensor构造方法：先指定类型，再指定形状，形状为NHWC（预处理之前的图像）
    // 由于mat矩阵必然是uchar，所以上面初始化的时候定义了图像预处理为u8，设置tensor的时候ppp会自动把u8改为float32
    ov::Tensor input_tensor = ov::Tensor(ov::element::u8, {1, (size_t)pre_processed_image.rows, (size_t)pre_processed_image.cols, 3}, pre_processed_image.data);

    // 正常运行阶段：向当前提交槽位送入新帧，并启动对应异步请求。
    // 预处理结果写入的是成员级缓冲区，即使当前函数结束，request 也不会引用到已经释放的局部 Mat。
    m_async_infer_requests[m_async_submit_index].set_input_tensor(input_tensor);
    m_async_infer_requests[m_async_submit_index].start_async();

    // 等待当前取结果槽位完成（上一帧的推理），形成 2 路并行流水。
    m_async_infer_requests[m_async_ready_index].wait();

    // 获取模型输出张量
    ov::Tensor output_tensor = m_async_infer_requests[m_async_ready_index].get_output_tensor();

    const float *output_data_ptr = output_tensor.data<const float>();

    m_async_submit_index = (m_async_submit_index + 1) % m_async_infer_requests.size();
    m_async_ready_index = (m_async_ready_index + 1) % m_async_infer_requests.size();

    return output_data_ptr;
}

// 异步四缓冲
const float *OpenVINOEngine::asyncInfer4(const cv::Mat &pre_processed_image)
{
    // Tensor构造方法：先指定类型，再指定形状，形状为NHWC（预处理之前的图像）
    ov::Tensor input_tensor = ov::Tensor(ov::element::u8, {1, (size_t)pre_processed_image.rows, (size_t)pre_processed_image.cols, 3}, pre_processed_image.data);

    // 向当前提交槽位送入新帧，并启动对应异步请求。
    // 4 路异步与 2 路异步保持同样的槽位轮询结构，预处理结果同样绑定到成员级缓冲区。
    m_async4_infer_requests[m_async4_submit_index].set_input_tensor(input_tensor);
    m_async4_infer_requests[m_async4_submit_index].start_async();

    // 等待当前取结果槽位完成，形成 4 路并行流水。
    m_async4_infer_requests[m_async4_ready_index].wait();
    ov::Tensor output_tensor = m_async4_infer_requests[m_async4_ready_index].get_output_tensor();
    const float *output_data_ptr = output_tensor.data<const float>();

    // 计算下一个提交槽位和取结果槽位
    m_async4_submit_index = (m_async4_submit_index + 1) % m_async4_infer_requests.size();
    m_async4_ready_index = (m_async4_ready_index + 1) % m_async4_infer_requests.size();

    return output_data_ptr;
}
