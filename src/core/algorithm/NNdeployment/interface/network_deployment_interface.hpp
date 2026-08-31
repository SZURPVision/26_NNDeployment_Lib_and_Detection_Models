#pragma once

#include <opencv2/core.hpp>

#include <memory>
#include <string>
#include <vector>

class YOLOModel;

// JSON 配置入口所需的三个路径参数。
class JsonConfig
{
public:
    // json路径、json内部key、模型文件夹路径。
    JsonConfig(std::string json_path,
               std::string model_key,
               std::string model_folder);

    std::string json_path;
    std::string model_key;
    std::string model_folder;
};

// 装甲板检测结果，坐标均为原图像素坐标。
struct NetArmorResult
{
    std::vector<cv::Point2d> points; // 左上、左下、右下、右上。
    int armor_id = -1;
    int color_id = -1;               // 0=蓝色，1=红色，2=白色。my_color=1表示我方为蓝色，=0表示我方为红色。
    int size = 0;                    // 0=小装甲板，1=大装甲板。
    double score = 0.;
    std::string class_name;
    std::string color_name;
};

// 神符检测结果，坐标均为原图像素坐标。
struct NetRuneResult
{
    std::vector<cv::Point2d> points; // top、left、R、right、bottom。
    cv::Point2d top, left, right, bottom, point_R;
    int class_id = -1;
    double score = 0.;
    std::string class_name;
};

// 控制模型信息输出与逐阶段耗时统计，可选构造参数之一
struct DebugConfig
{
    bool print_debug_info = false;      // 开启后会输出网络的debug调试结果
    bool calculate_speed_info = false;  // 开启后会输出网络各阶段耗时统计信息

    DebugConfig(bool debug = false, bool speed = false)
        : print_debug_info(debug), calculate_speed_info(speed)
    {}
};

// 只产生装甲板结果的网络模型。对象不可复制，可以移动。
class ArmorModel
{
public:
    // model_path：模型文件完整路径。
    explicit ArmorModel(const std::string &model_path);
    // 直接指定模型路径、推理模式和部署后端，剩余参数使用默认值。
    ArmorModel(const std::string &model_path,
               const std::string &infer_mode,
               const std::string &deploy_way,
               const std::string &device = "GPU",
               float confidence_threshold = 0.5f,
               const std::string &postprocess_mode = "auto_detect",
               const DebugConfig &debug_config = DebugConfig());
    // 从 JSON 的指定节点读取模型配置。
    explicit ArmorModel(const JsonConfig &json_config,
                        const DebugConfig &debug_config = DebugConfig());
    ~ArmorModel() noexcept;

    // 禁止复制构造和赋值，允许移动构造和移动赋值。
    ArmorModel(const ArmorModel &) = delete;
    ArmorModel &operator=(const ArmorModel &) = delete;
    ArmorModel(ArmorModel &&) noexcept;
    ArmorModel &operator=(ArmorModel &&) noexcept;

    // 执行装甲板推理；my_color=1 表示我方蓝色，=0 表示我方红色，=2 表示离线演示时保留蓝红双方结果。
    // 无有效结果时返回空 vector；模型任务不匹配会在构造时抛出异常。
    std::vector<NetArmorResult> netProcess(const cv::Mat &input_image, const int &my_color);

private:
    // 具体实现由内部的 YOLOModel 完成，ArmorModel 仅作为接口类，对外暴露构造函数和armor获取函数。
    std::unique_ptr<YOLOModel> m_model;
};

// 只产生神符结果的网络模型。对象不可复制，可以移动。
class RuneModel
{
public:
    // model_path：模型文件完整路径。
    explicit RuneModel(const std::string &model_path);
    // 直接指定模型路径、推理模式和部署后端，剩余参数使用默认值。
    RuneModel(const std::string &model_path,
              const std::string &infer_mode,
              const std::string &deploy_way,
              const std::string &device = "GPU",
              float confidence_threshold = 0.5f,
              const std::string &postprocess_mode = "auto_detect",
              const DebugConfig &debug_config = DebugConfig());
    // 从 JSON 的指定节点读取模型配置。
    explicit RuneModel(const JsonConfig &json_config,
                       const DebugConfig &debug_config = DebugConfig());
    ~RuneModel() noexcept;

    // 禁止复制构造和赋值，允许移动构造和移动赋值。
    RuneModel(const RuneModel &) = delete;
    RuneModel &operator=(const RuneModel &) = delete;
    RuneModel(RuneModel &&) noexcept;
    RuneModel &operator=(RuneModel &&) noexcept;

    // 执行神符推理；无有效结果时返回空 vector，模型任务不匹配会在构造时抛出异常。
    std::vector<NetRuneResult> netProcess(const cv::Mat &input_image);

private:
    // 具体实现由内部的 YOLOModel 完成，RuneModel 仅作为接口类，对外暴露构造函数和rune获取函数。
    std::unique_ptr<YOLOModel> m_model;
};
