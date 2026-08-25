#pragma once

#include "network_deployment_lib.hpp"

#include <opencv2/core.hpp>

#include <queue>
#include <string>
#include <vector>

class YOLOModel::Postprocessor
{
public:
    Postprocessor(const ModelConfig &modelconfig, float nms_threshold);
    virtual ~Postprocessor() = default;

    virtual std::vector<NetArmorResult> postProcessArmorMat(const float *input_ptr, const InferParam &infer_param, const int &my_color) { throw std::logic_error("detect postprocess not supported"); };

    virtual std::vector<NetRuneResult> postProcessRuneMat(const float *input_ptr, const InferParam &infer_param) { throw std::logic_error("detect postprocess not supported"); };

protected:
    ModelConfig m_model_config;
    float m_nms_threshold = 0.f;
};

class V5InfantryPostProcessor : public YOLOModel::Postprocessor
{
public:
    V5InfantryPostProcessor(const YOLOModel::ModelConfig &modelconfig, float nms_threshold);
    ~V5InfantryPostProcessor() = default;

    std::vector<NetArmorResult> postProcessArmorMat(const float *input_ptr, const InferParam &infer_param, const int &my_color) override;

private:
    std::vector<std::string> m_armor_names = {"哨兵", "英雄", "工程", "3号步兵", "4号步兵", "5号步兵", "前哨站", "基地（底部）", "基地（顶部）"};
    std::vector<std::string> m_color_names = {"蓝色", "红色", "白色", "紫色"};
    std::queue<std::vector<NetArmorResult>> m_last_results;
};

class V8InfantryPostProcessor : public YOLOModel::Postprocessor
{
public:
    V8InfantryPostProcessor(const YOLOModel::ModelConfig &modelconfig, float nms_threshold);
    ~V8InfantryPostProcessor() = default;

    std::vector<NetArmorResult> postProcessArmorMat(const float *input_ptr, const InferParam &infer_param, const int &my_color) override;

private:
    std::vector<std::string> m_armor_names = {"哨兵", "英雄", "工程", "3号步兵", "4号步兵", "5号步兵", "前哨站", "基地（底部）", "基地（顶部）"};
    std::vector<std::string> m_color_names = {"蓝色", "红色", "白色", "紫色"};
    std::queue<std::vector<NetArmorResult>> m_last_results;
};

class V8_21InfantryPostProcessor : public YOLOModel::Postprocessor
{
public:
    V8_21InfantryPostProcessor(const YOLOModel::ModelConfig &modelconfig, float nms_threshold);
    ~V8_21InfantryPostProcessor() = default;

    std::vector<NetArmorResult> postProcessArmorMat(const float *input_ptr, const InferParam &infer_param, const int &my_color) override;

private:
    std::vector<std::string> m_armor_names = {"哨兵", "英雄", "工程", "3号步兵", "4号步兵", "5号步兵", "前哨站", "基地（底部）", "基地（顶部）"};
    std::vector<std::string> m_color_names = {"蓝色", "红色", "白色", "紫色"};
    std::queue<std::vector<NetArmorResult>> m_last_results;
};

class LidarPostProcessor : public YOLOModel::Postprocessor
{
public:
    LidarPostProcessor(const YOLOModel::ModelConfig &modelconfig, float nms_threshold);
    ~LidarPostProcessor() = default;

    std::vector<NetArmorResult> postProcessArmorMat(const float *input_ptr, const InferParam &infer_param, const int &my_color) override;

private:
    std::vector<std::string> m_armor_names = {"蓝色1号英雄", "蓝色2号工程", "蓝色3号步兵", "蓝色4号步兵", "蓝色哨兵",
                                              "红色1号英雄", "红色2号工程", "红色3号步兵", "红色4号步兵", "红色哨兵"};
    std::vector<std::string> m_color_names = {"蓝色", "红色"};
};

class RunePostProcessor : public YOLOModel::Postprocessor
{
public:
    RunePostProcessor(const YOLOModel::ModelConfig &modelconfig, float nms_threshold);
    ~RunePostProcessor() = default;

    std::vector<NetRuneResult> postProcessRuneMat(const float *input_ptr, const InferParam &infer_param) override;

private:
    std::vector<std::string> m_class_names = {"未激活", "小符已激活", "大符已激活"};
};
