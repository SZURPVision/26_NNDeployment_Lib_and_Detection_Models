#include "network_deployment_interface.hpp"

#include "network_deployment_lib.hpp"

#include <stdexcept>
#include <utility>

JsonConfig::JsonConfig(std::string json_path,
                       std::string model_key,
                       std::string model_folder)
    : json_path(std::move(json_path)),
      model_key(std::move(model_key)),
      model_folder(std::move(model_folder))
{}

ArmorModel::ArmorModel(const std::string &model_path)
    : m_model(std::make_unique<YOLOModel>(model_path))
{
    if (!m_model->supportsArmor())
        throw std::invalid_argument("模型后处理类型不是装甲板，无法构造 ArmorModel");
}

ArmorModel::ArmorModel(const std::string &model_path,
                       const std::string &infer_mode,
                       const std::string &deploy_way,
                       const std::string &device,
                       float confidence_threshold,
                       const std::string &postprocess_mode,
                       const DebugConfig &debug_config)
    : m_model(std::make_unique<YOLOModel>(model_path,
                                         infer_mode,
                                         deploy_way,
                                         device,
                                         confidence_threshold,
                                         postprocess_mode,
                                         debug_config))
{
    if (!m_model->supportsArmor())
        throw std::invalid_argument("模型后处理类型不是装甲板，无法构造 ArmorModel");
}

ArmorModel::ArmorModel(const JsonConfig &json_config,
                       const DebugConfig &debug_config)
    : m_model(std::make_unique<YOLOModel>(json_config, debug_config))
{
    if (!m_model->supportsArmor())
        throw std::invalid_argument("模型后处理类型不是装甲板，无法构造 ArmorModel");
}

ArmorModel::~ArmorModel() noexcept = default;
ArmorModel::ArmorModel(ArmorModel &&) noexcept = default;
ArmorModel &ArmorModel::operator=(ArmorModel &&) noexcept = default;

std::vector<NetArmorResult> ArmorModel::netProcess(const cv::Mat &input_image, const int &my_color)
{
    if (!m_model)
        return {};
    return m_model->netProcess(input_image, my_color);
}

RuneModel::RuneModel(const std::string &model_path)
    : m_model(std::make_unique<YOLOModel>(model_path))
{
    if (!m_model->supportsRune())
        throw std::invalid_argument("模型后处理类型不是神符，无法构造 RuneModel");
}

RuneModel::RuneModel(const std::string &model_path,
                     const std::string &infer_mode,
                     const std::string &deploy_way,
                     const std::string &device,
                     float confidence_threshold,
                     const std::string &postprocess_mode,
                     const DebugConfig &debug_config)
    : m_model(std::make_unique<YOLOModel>(model_path,
                                         infer_mode,
                                         deploy_way,
                                         device,
                                         confidence_threshold,
                                         postprocess_mode,
                                         debug_config))
{
    if (!m_model->supportsRune())
        throw std::invalid_argument("模型后处理类型不是神符，无法构造 RuneModel");
}

RuneModel::RuneModel(const JsonConfig &json_config,
                     const DebugConfig &debug_config)
    : m_model(std::make_unique<YOLOModel>(json_config, debug_config))
{
    if (!m_model->supportsRune())
        throw std::invalid_argument("模型后处理类型不是神符，无法构造 RuneModel");
}

RuneModel::~RuneModel() noexcept = default;
RuneModel::RuneModel(RuneModel &&) noexcept = default;
RuneModel &RuneModel::operator=(RuneModel &&) noexcept = default;

std::vector<NetRuneResult> RuneModel::netProcess(const cv::Mat &input_image)
{
    if (!m_model)
        return {};
    return m_model->netProcess(input_image);
}
