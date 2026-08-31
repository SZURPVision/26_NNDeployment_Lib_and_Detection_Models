#include "network_postprocess.hpp"

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <queue>
#include <sstream>
#include <utility>

// ==================== 后处理模块 ====================
// 后处理模块：v5 armor、v8 armor、lidar 和 rune 四类结果解析。

// ==================== 部署流程辅助区域 ====================
namespace
{
// 将每个候选项封装，便于直接进行张量计算
struct CandidateView
{
    const float *data = nullptr;
    std::size_t candidate_index = 0;
    std::size_t candidate_stride = 0;
    std::size_t feature_stride = 0;

    // 按特征索引读取当前候选项的数据。
    float operator[](std::size_t feature_index) const noexcept
    {
        return data[candidate_index * candidate_stride + feature_index * feature_stride];
    }
};

// 计算装甲板四个关键点的最小外接矩形。
cv::Rect2d getArmorBoundingRect(const NetArmorResult &result)
{
    double min_x = result.points[0].x;
    double max_x = result.points[0].x;
    double min_y = result.points[0].y;
    double max_y = result.points[0].y;
    for (int i = 1; i < 4; ++i)
    {
        min_x = std::min(min_x, result.points[i].x);
        max_x = std::max(max_x, result.points[i].x);
        min_y = std::min(min_y, result.points[i].y);
        max_y = std::max(max_y, result.points[i].y);
    }
    return cv::Rect2d(min_x, min_y, max_x - min_x, max_y - min_y);
}

// 计算两个装甲板外接矩形的交并比。
double getArmorIoU(const NetArmorResult &lhs, const NetArmorResult &rhs)
{
    const cv::Rect2d lhs_rect = getArmorBoundingRect(lhs);
    const cv::Rect2d rhs_rect = getArmorBoundingRect(rhs);
    const cv::Rect2d intersection = lhs_rect & rhs_rect;
    const double intersection_area = intersection.area();
    if (intersection_area <= 0.0)
        return 0.0;

    const double union_area = lhs_rect.area() + rhs_rect.area() - intersection_area;
    if (union_area <= 0.0)
        return 0.0;

    return intersection_area / union_area;
}

} // namespace

// 装甲板后处理流程

// 构造白色装甲板判定使用的初始历史队列。
std::queue<std::vector<NetArmorResult>> buildInitialArmorHistory()
{
    std::queue<std::vector<NetArmorResult>> last_results;
    last_results.push({});
    last_results.push({});
    last_results.push({});
    last_results.push({});
    last_results.push({});
    return last_results;
}

// 根据颜色和历史结果判断当前白色装甲板结果是否应保留。
bool shouldAppendArmorResult(const NetArmorResult &result, const int my_color, std::queue<std::vector<NetArmorResult>> last_results)
{
    // 仅白色装甲板额外判断。my_color == 2 表示测试模式，全部识别
    if ((result.color_id != 2) || (my_color ==2))
        return true;

    while (!last_results.empty())
    {
        const std::vector<NetArmorResult> &history_results = last_results.front();
        for (const NetArmorResult &history_result : history_results)
        {
            // 历史装甲板为待识别颜色，类别相同且 IOU 通过，可提供自瞄识别
            if (((my_color == 0 && history_result.color_id == 0) ||
                 (my_color == 1 && history_result.color_id == 1)) &&
                history_result.armor_id == result.armor_id &&
                getArmorIoU(history_result, result) > 0.3) 
                return true;
        }
        last_results.pop();
    }

    return false;
}

// 保存后处理所需的模型配置和非极大值抑制阈值。
YOLOModel::PostProcessor::PostProcessor(const ModelConfig &model_config, float nms_threshold)
    : m_model_config(model_config), m_nms_threshold(nms_threshold) {}

// ==================== v5步兵实现 ====================
// 初始化 V5 步兵后处理器及装甲板历史队列。
V5InfantryPostProcessor::V5InfantryPostProcessor(const YOLOModel::ModelConfig &model_config, float nms_threshold)
    : PostProcessor(model_config, nms_threshold), m_last_results(buildInitialArmorHistory()) {}

// 解析 V5 步兵模型输出并生成装甲板检测结果。
std::vector<NetArmorResult> V5InfantryPostProcessor::postProcessArmorMat(const float *input_ptr, const InferParam &infer_param, const int &my_color)
{
    const float confidence_threshold = m_model_config.confidence_threshold;
    const float nms_threshold = m_nms_threshold;

    if (!input_ptr || infer_param.out_tensor_cols != (4 * 2 + 1 + 4 + 9))
    {
        std::cerr << "模型输出形状不匹配，请检查模型路径" << std::endl;
        return {};
    }

    std::vector<int> class_ids_temp;         // 装甲板类别ID
    std::vector<float> confidences_temp;     // 检测置信度（用于NMS）——注意：为了对接自瞄代码，score是double，但是nms要求confidence必须是float，所以这里用的是float，到最后强转为double
    std::vector<int> sizes_temp;             // 装甲板尺寸标志（0=小装甲板，1=大装甲板）
    std::vector<int> colors_temp;            // 装甲板颜色类别（0=蓝，1=红，2=白，3=紫）
    std::vector<cv::Rect> boxes_temp;        // 边界框（用于OpenCV NMS算法）
    std::vector<cv::Point2d> keypoints_temp; // 装甲板四个关键点坐标
    std::vector<int> indices_temp;           // NMS后保留的检测框索引结果
    std::vector<NetArmorResult> results;     // 完整的装甲板检测结果

    // v5的置信度没做sigmoid，此处对confidence阈值做反sigmoid，等价于对所有置信度做了sigmoid之后按0-1阈值筛选
    const double raw_confidence_threshold = std::log(static_cast<double>(confidence_threshold) / (1.0 - static_cast<double>(confidence_threshold)));
    const float score_threshold = confidence_threshold;

    // V5 输出为 25200*22，每行是一组候选结果。
    for (int candidate_index = 0; candidate_index < infer_param.out_tensor_rows; ++candidate_index)
    {
        // 封装候选项，便于后续直接当作张量计算
        const CandidateView candidate{input_ptr, static_cast<std::size_t>(candidate_index),
                                      static_cast<std::size_t>(infer_param.out_tensor_cols), 1};

        const double raw_confidence = candidate[8]; // 第8列为未归一化的总体置信度
        if (raw_confidence < raw_confidence_threshold)
            continue; // 先过滤低置信度候选，避免无效的类别比较

        // 类别分支得分：[13,21]分别对应9个类别
        int class_id = 0;
        double best_class_score = candidate[13];
        for (int class_index = 1; class_index < 9; ++class_index)
        {
            const double class_score = candidate[13 + class_index];
            if (class_score > best_class_score)
            {
                best_class_score = class_score;
                class_id = class_index;
            }
        }

        // sigmoid 还原 0-1 区间置信度
        const double confidence = raw_confidence >= 0.0
                                      ? 1.0 / (1.0 + std::exp(-raw_confidence))
                                      : std::exp(raw_confidence) / (1.0 + std::exp(raw_confidence));

        // 颜色分支得分：[9,12]分别对应4种颜色
        int color_id = 0;
        double best_color_score = candidate[9];
        for (int color_index = 1; color_index < 4; ++color_index)
        {
            const double color_score = candidate[9 + color_index];
            if (color_score > best_color_score)
            {
                best_color_score = color_score;
                color_id = color_index;
            }
        }

        if (color_id == 3)
            continue; // 忽略紫色装甲板（异常装甲板）
        if (color_id == 0 && my_color == 1)
            continue; // 己方为蓝色时，忽略蓝色装甲板（友军标识）
        if (color_id == 1 && my_color == 0)
            continue; // 己方为红色时，忽略红色装甲板（友军标识）

        std::vector<cv::Point2f> current_keypoints(4);

        for (int point_index = 0; point_index < 4; ++point_index)
        {
            const float x = candidate[point_index * 2 + 0];
            const float y = candidate[point_index * 2 + 1];

            // 还原到原图尺度
            const float kpt_x = (x - infer_param.pad_x) / infer_param.scale;
            const float kpt_y = (y - infer_param.pad_y) / infer_param.scale;

            current_keypoints[point_index] = cv::Point2f(kpt_x, kpt_y);
        }

        // 根据类别ID判断装甲板尺寸：1（英雄）和7（基地）为大装甲板，其他为小装甲板
        if (class_id == 1 || class_id == 7)
            sizes_temp.emplace_back(1); // 大装甲板标记
        else
            sizes_temp.emplace_back(0); // 小装甲板标记

        for (const cv::Point2f &point : current_keypoints)
            keypoints_temp.emplace_back(point.x, point.y);

        // 保存检测结果的各个分量
        confidences_temp.emplace_back(static_cast<float>(confidence));
        colors_temp.emplace_back(color_id);
        class_ids_temp.emplace_back(class_id);
        boxes_temp.emplace_back(cv::boundingRect(current_keypoints));
    }

    // 对候选框执行非极大值抑制。
    cv::dnn::NMSBoxes(boxes_temp, confidences_temp, score_threshold, nms_threshold, indices_temp);

    // 组装非极大值抑制保留的装甲板结果。
    for (const int &index : indices_temp)
    {
        // 准备结果
        NetArmorResult result;

        // 检查并存储关键点
        bool throw_flag = false;
        for (int point_index = 0; point_index < 4; ++point_index)
        {
            // 检查坐标是否超出模型输入范围，防止后续计算出错
            if (keypoints_temp[index * 4 + point_index].x > infer_param.origin_width ||
                keypoints_temp[index * 4 + point_index].x < 0 ||
                keypoints_temp[index * 4 + point_index].y > infer_param.origin_height ||
                keypoints_temp[index * 4 + point_index].y < 0)
            {
                throw_flag = true;
                break;
            }
        }
        if (throw_flag)
            continue; // 坐标无效，跳过此检测框

        result.points = {
            cv::Point2d(keypoints_temp[index * 4 + 0].x, keypoints_temp[index * 4 + 0].y),
            cv::Point2d(keypoints_temp[index * 4 + 1].x, keypoints_temp[index * 4 + 1].y),
            cv::Point2d(keypoints_temp[index * 4 + 2].x, keypoints_temp[index * 4 + 2].y),
            cv::Point2d(keypoints_temp[index * 4 + 3].x, keypoints_temp[index * 4 + 3].y)};

        // 存储类别
        result.armor_id = class_ids_temp[index];
        result.color_id = colors_temp[index];
        result.size = sizes_temp[index];
        result.score = static_cast<double>(confidences_temp[index]);
        result.class_name = m_armor_names[result.armor_id];
        result.color_name = m_color_names[result.color_id];

        // 对白色装甲板进行的额外判断（非红蓝击打变化为白色时不输出）
        if (!shouldAppendArmorResult(result, my_color, m_last_results))
            continue;

        // 相当于push_back，但是更省空间，不用拷贝
        results.emplace_back(std::move(result));
    }

    // 更新白色装甲板判定使用的历史结果。
    m_last_results.pop();
    m_last_results.push(results);
    return results;
}

// ==================== v8步兵实现 ====================
// 初始化 V8 步兵后处理器及装甲板历史队列。
V8InfantryPostProcessor::V8InfantryPostProcessor(const YOLOModel::ModelConfig &model_config, float nms_threshold)
    : PostProcessor(model_config, nms_threshold), m_last_results(buildInitialArmorHistory()) {}

// ==================== v8 21维步兵实现 ====================
// 初始化 V8 21维步兵后处理器及装甲板历史队列。
V8_21InfantryPostProcessor::V8_21InfantryPostProcessor(const YOLOModel::ModelConfig &model_config, float nms_threshold)
    : PostProcessor(model_config, nms_threshold), m_last_results(buildInitialArmorHistory()) {}

// 解析 V8 21维步兵模型输出并生成装甲板检测结果。
std::vector<NetArmorResult> V8_21InfantryPostProcessor::postProcessArmorMat(const float *input_ptr, const InferParam &infer_param, const int &my_color)
{
    const float confidence_threshold = m_model_config.confidence_threshold;
    const float nms_threshold = m_nms_threshold;

    if (!input_ptr || infer_param.out_tensor_rows != (4 + 9 + 4 * 2))
    {
        std::cerr << "模型输出形状不匹配，请检查模型路径" << std::endl;
        return {};
    }

    std::vector<int> class_ids_temp;
    std::vector<float> confidences_temp;
    std::vector<int> sizes_temp;
    std::vector<int> colors_temp;
    std::vector<cv::Rect> boxes_temp;
    std::vector<cv::Point2d> keypoints_temp;
    std::vector<int> indices_temp;
    std::vector<NetArmorResult> results;

    const float score_threshold = confidence_threshold;

    // V8 21维输出为 21*6300，每列是一组候选结果。
    for (int candidate_index = 0; candidate_index < infer_param.out_tensor_cols; ++candidate_index)
    {
        // 封装候选项，便于后续直接当作张量计算
        const CandidateView candidate{input_ptr, static_cast<std::size_t>(candidate_index), 1,
                                      static_cast<std::size_t>(infer_param.out_tensor_cols)};

        // 类别分支得分：[4,12]分别对应9个类别
        int class_id = 0;
        double best_class_score = candidate[4];
        for (int class_index = 1; class_index < 9; ++class_index)
        {
            const double class_score = candidate[4 + class_index];
            if (class_score > best_class_score)
            {
                best_class_score = class_score;
                class_id = class_index;
            }
        }

        const double confidence = best_class_score;
        if (confidence < confidence_threshold)
            continue;

        // 颜色分支得分：[0,3]分别对应4种颜色
        int color_id = 0;
        double best_color_score = candidate[0];
        for (int color_index = 1; color_index < 4; ++color_index)
        {
            const double color_score = candidate[color_index];
            if (color_score > best_color_score)
            {
                best_color_score = color_score;
                color_id = color_index;
            }
        }

        // 过滤异常颜色和己方颜色。
        if (color_id == 3)
            continue;
        if (color_id == 0 && my_color == 1)
            continue;
        if (color_id == 1 && my_color == 0)
            continue;

        // 解码装甲板关键点并还原到原图尺度。
        std::vector<cv::Point2f> current_keypoints(4);
        for (int point_index = 0; point_index < 4; ++point_index)
        {
            const float x = candidate[13 + point_index * 2];
            const float y = candidate[13 + point_index * 2 + 1];

            const float kpt_x = (x - infer_param.pad_x) / infer_param.scale;
            const float kpt_y = (y - infer_param.pad_y) / infer_param.scale;

            current_keypoints[point_index] = cv::Point2f(kpt_x, kpt_y);
        }

        // 根据类别确定装甲板尺寸。
        if (class_id == 1 || class_id == 7)
            sizes_temp.emplace_back(1);
        else
            sizes_temp.emplace_back(0);

        for (const cv::Point2f &point : current_keypoints)
            keypoints_temp.emplace_back(point.x, point.y);

        // 保存候选结果的各个分量。
        confidences_temp.emplace_back(static_cast<float>(confidence));
        colors_temp.emplace_back(color_id);
        class_ids_temp.emplace_back(class_id);
        boxes_temp.emplace_back(cv::boundingRect(current_keypoints));
    }

    // 对候选框执行非极大值抑制。
    cv::dnn::NMSBoxes(boxes_temp, confidences_temp, score_threshold, nms_threshold, indices_temp);

    // 组装非极大值抑制保留的装甲板结果。
    for (const int &index : indices_temp)
    {
        NetArmorResult result;

        bool throw_flag = false;
        for (int point_index = 0; point_index < 4; ++point_index)
        {
            if (keypoints_temp[index * 4 + point_index].x > infer_param.origin_width ||
                keypoints_temp[index * 4 + point_index].x < 0 ||
                keypoints_temp[index * 4 + point_index].y > infer_param.origin_height ||
                keypoints_temp[index * 4 + point_index].y < 0)
            {
                throw_flag = true;
                break;
            }
        }
        if (throw_flag)
            continue;

        result.points = {
            cv::Point2d(keypoints_temp[index * 4 + 0].x, keypoints_temp[index * 4 + 0].y),
            cv::Point2d(keypoints_temp[index * 4 + 1].x, keypoints_temp[index * 4 + 1].y),
            cv::Point2d(keypoints_temp[index * 4 + 2].x, keypoints_temp[index * 4 + 2].y),
            cv::Point2d(keypoints_temp[index * 4 + 3].x, keypoints_temp[index * 4 + 3].y)};

        result.armor_id = class_ids_temp[index];
        result.color_id = colors_temp[index];
        result.size = sizes_temp[index];
        result.score = static_cast<double>(confidences_temp[index]);
        result.class_name = m_armor_names[result.armor_id];
        result.color_name = m_color_names[result.color_id];

        if (!shouldAppendArmorResult(result, my_color, m_last_results))
            continue;

        results.emplace_back(std::move(result));
    }

    // 更新白色装甲板判定使用的历史结果。
    m_last_results.pop();
    m_last_results.push(results);
    return results;
}

// ==================== v8步兵实现 ====================
// 解析 V8 步兵模型输出并生成装甲板检测结果。
std::vector<NetArmorResult> V8InfantryPostProcessor::postProcessArmorMat(const float *input_ptr, const InferParam &infer_param, const int &my_color)
{
    const float confidence_threshold = m_model_config.confidence_threshold;
    const float nms_threshold = m_nms_threshold;

    if (!input_ptr || infer_param.out_tensor_rows != (4 + 9 + 4 * 3))
    {
        std::cerr << "模型输出形状不匹配，请检查模型路径" << std::endl;
        return {};
    }
    std::vector<int> class_ids_temp;         // 装甲板类别ID
    std::vector<float> confidences_temp;     // 检测置信度（用于NMS）
    std::vector<int> sizes_temp;             // 装甲板尺寸标志（0=小装甲板，1=大装甲板）
    std::vector<int> colors_temp;            // 装甲板颜色类别（0=蓝，1=红，2=白，3=紫）
    std::vector<cv::Rect> boxes_temp;        // 边界框（用于OpenCV NMS算法）
    std::vector<cv::Point2d> keypoints_temp; // 装甲板四个关键点坐标
    std::vector<int> indices_temp;           // NMS后保留的检测框索引结果
    std::vector<NetArmorResult> results;     // 完整的装甲板检测结果
    const float score_threshold = confidence_threshold;

    // V8 25维输出为 25*6300，每列是一组候选结果。
    for (int candidate_index = 0; candidate_index < infer_param.out_tensor_cols; ++candidate_index)
    {
        // 封装候选项，便于后续直接当作张量计算
        const CandidateView candidate{input_ptr, static_cast<std::size_t>(candidate_index), 1,
                                      static_cast<std::size_t>(infer_param.out_tensor_cols)};

        // 类别分支得分：[4,12]分别对应9个类别
        int class_id = 0;
        double best_class_score = candidate[4];
        for (int class_index = 1; class_index < 9; ++class_index)
        {
            const double class_score = candidate[4 + class_index];
            if (class_score > best_class_score)
            {
                best_class_score = class_score;
                class_id = class_index;
            }
        }

        const double confidence = best_class_score;
        if (confidence < confidence_threshold)
            continue; // 置信度低于阈值的直接跳过

        // 颜色分支得分：[0,3]分别对应4种颜色
        int color_id = 0;
        double best_color_score = candidate[0];
        for (int color_index = 1; color_index < 4; ++color_index)
        {
            const double color_score = candidate[color_index];
            if (color_score > best_color_score)
            {
                best_color_score = color_score;
                color_id = color_index;
            }
        }

        if (color_id == 3)
            continue; // 忽略紫色装甲板（异常装甲板）
        if (color_id == 0 && my_color == 1)
            continue; // 己方为蓝色时，忽略蓝色装甲板（友军标识）
        if (color_id == 1 && my_color == 0)
            continue; // 己方为红色时，忽略红色装甲板（友军标识）

        std::vector<cv::Point2f> current_keypoints(4);
        for (int point_index = 0; point_index < 4; ++point_index)
        {
            const float x = candidate[13 + point_index * 3 + 0];
            const float y = candidate[13 + point_index * 3 + 1];

            // 还原到原图尺度
            const float kpt_x = (x - infer_param.pad_x) / infer_param.scale;
            const float kpt_y = (y - infer_param.pad_y) / infer_param.scale;

            current_keypoints[point_index] = cv::Point2f(kpt_x, kpt_y);
        }

        // 根据类别ID判断装甲板尺寸：1（英雄）和7（基地）为大装甲板，其他为小装甲板
        if (class_id == 1 || class_id == 7)
            sizes_temp.emplace_back(1); // 大装甲板标记
        else
            sizes_temp.emplace_back(0); // 小装甲板标记

        for (const cv::Point2f &point : current_keypoints)
            keypoints_temp.emplace_back(point.x, point.y);

        // 保存检测结果的各个分量
        confidences_temp.emplace_back(static_cast<float>(confidence));
        colors_temp.emplace_back(color_id);
        class_ids_temp.emplace_back(class_id);
        boxes_temp.emplace_back(cv::boundingRect(current_keypoints));
    }

    // 对候选框执行非极大值抑制。
    cv::dnn::NMSBoxes(boxes_temp, confidences_temp, score_threshold, nms_threshold, indices_temp);

    // 组装非极大值抑制保留的装甲板结果。
    for (const int &index : indices_temp)
    {
        // 准备结果
        NetArmorResult result;

        // 检查并存储关键点
        bool throw_flag = false;
        for (int point_index = 0; point_index < 4; ++point_index)
        {
            // 检查坐标是否超出模型输入范围，防止后续计算出错
            if (keypoints_temp[index * 4 + point_index].x > infer_param.origin_width ||
                keypoints_temp[index * 4 + point_index].x < 0 ||
                keypoints_temp[index * 4 + point_index].y > infer_param.origin_height ||
                keypoints_temp[index * 4 + point_index].y < 0)
            {
                throw_flag = true;
                break;
            }
        }
        if (throw_flag)
            continue; // 坐标无效，跳过此检测框

        result.points = {
            cv::Point2d(keypoints_temp[index * 4 + 0].x, keypoints_temp[index * 4 + 0].y),
            cv::Point2d(keypoints_temp[index * 4 + 1].x, keypoints_temp[index * 4 + 1].y),
            cv::Point2d(keypoints_temp[index * 4 + 2].x, keypoints_temp[index * 4 + 2].y),
            cv::Point2d(keypoints_temp[index * 4 + 3].x, keypoints_temp[index * 4 + 3].y)};

        // 存储类别
        result.armor_id = class_ids_temp[index];
        result.color_id = colors_temp[index];
        result.size = sizes_temp[index];
        result.score = static_cast<double>(confidences_temp[index]);
        result.class_name = m_armor_names[result.armor_id];
        result.color_name = m_color_names[result.color_id];

        if (!shouldAppendArmorResult(result, my_color, m_last_results))
            continue;

        // 相当于push_back，但是更省空间，不用拷贝
        results.emplace_back(std::move(result));
    }

    // 更新白色装甲板判定使用的历史结果。
    m_last_results.pop();
    m_last_results.push(results);
    return results;
}

// ==================== 雷达实现 ====================
// 初始化雷达后处理器。
LidarPostProcessor::LidarPostProcessor(const YOLOModel::ModelConfig &model_config, float nms_threshold)
    : PostProcessor(model_config, nms_threshold) {}

// 解析雷达模型输出并生成装甲板检测结果。
std::vector<NetArmorResult> LidarPostProcessor::postProcessArmorMat(const float *input_ptr, const InferParam &infer_param, const int &my_color)
{
    static_cast<void>(my_color);
    const float confidence_threshold = m_model_config.confidence_threshold;
    const float nms_threshold = m_nms_threshold;

    if (!input_ptr || infer_param.out_tensor_rows != (4 + 10 + 4 * 3))
    {
        std::cerr << "模型输出形状不匹配，请检查模型路径" << std::endl;
        return {};
    }
    std::vector<int> class_ids_temp;         // 装甲板类别ID
    std::vector<float> confidences_temp;     // 检测置信度（用于NMS）
    std::vector<cv::Rect> rects_temp;        // 雷达中rect由原始的xywh输出
    std::vector<cv::Point2d> keypoints_temp; // 装甲板四个关键点坐标
    std::vector<NetArmorResult> results;     // 完整的装甲板检测结果

    // 雷达网络输出形状为 26*2100，每列是一个预测结果
    for (int current_anchor_box = 0; current_anchor_box < infer_param.out_tensor_cols; current_anchor_box++)
    {
        // 封装候选项，便于后续直接当作张量计算
        const CandidateView candidate{input_ptr, static_cast<std::size_t>(current_anchor_box), 1,
                                      static_cast<std::size_t>(infer_param.out_tensor_cols)};

        // 置信度最高的类别的索引和置信度
        int best_class_idx = 0;
        double max_class_conf = candidate[4];
        for (int cls = 1; cls < 10; ++cls)
        {
            double score = candidate[4 + cls];
            if (score > max_class_conf)
            {
                max_class_conf = score;
                best_class_idx = cls;
            }
        }

        // 如果最大的置信度也低于阈值,则跳过该锚框
        if (max_class_conf < confidence_threshold)
        {
            continue;
        }

        // 前四行是锚框的位置和大小
        float cx_temp = candidate[0];
        float cy_temp = candidate[1];
        float w_temp = candidate[2];
        float h_temp = candidate[3];

        // 还原到原图尺度
        int lt_x = std::max(0.f, (((cx_temp - 0.5f * w_temp) - infer_param.pad_x) / infer_param.scale) + 0.5f); // lt_x为缩放前原图片中锚框的左上角x值
        int lt_y = std::max(0.f, (((cy_temp - 0.5f * h_temp) - infer_param.pad_y) / infer_param.scale) + 0.5f);
        int w = static_cast<int>(w_temp / infer_param.scale + 0.5); // 缩放前的锚框宽度
        int h = static_cast<int>(h_temp / infer_param.scale + 0.5);

        // 放入容器
        // best_class_idx是1*10向量中最大的点，x为1，y是最大config值
        class_ids_temp.push_back(best_class_idx);
        confidences_temp.push_back(static_cast<float>(max_class_conf));
        rects_temp.push_back(cv::Rect(lt_x, lt_y, w, h));

        // keypoints解码(4个关键点)
        for (int kpt_num = 0; kpt_num < 4; kpt_num++)
        {
            float kpt_x_temp = candidate[14 + kpt_num * 3 + 0]; // x
            float kpt_y_temp = candidate[14 + kpt_num * 3 + 1]; // y

            // 还原到原图尺度
            float kpt_x = (kpt_x_temp - infer_param.pad_x) / infer_param.scale;
            float kpt_y = (kpt_y_temp - infer_param.pad_y) / infer_param.scale;

            // 放入容器
            keypoints_temp.push_back(cv::Point2d(kpt_x, kpt_y));
        }
    }

    // 只要任何一个容器为空,说明没有结果（这四个容器是一起传数据的，所以顺序是固定的）
    if (class_ids_temp.empty())
    {
        return {};
    }

    // 非极大值抑制：将锚框和其对应的置信度作非极大值抑制去重，阈值为置信度和交并比，输出参数为保留的索引（indices）
    std::vector<int> indices; // 索引容器
    cv::dnn::NMSBoxes(rects_temp, confidences_temp, confidence_threshold, nms_threshold, indices);

    // 遍历indices并且处理偏移来生成最终的返回值
    results.reserve(indices.size());
    for (const int &index : indices)
    {
        // 准备结果
        NetArmorResult result;

        // 检查并存储关键点
        bool throw_flag = false;
        for (int i = 0; i < 4; ++i)
        {
            // 检查坐标是否超出模型输入范围，防止后续计算出错
            if (keypoints_temp[index * 4 + i].x > infer_param.origin_width ||
                keypoints_temp[index * 4 + i].x < 0 ||
                keypoints_temp[index * 4 + i].y > infer_param.origin_height ||
                keypoints_temp[index * 4 + i].y < 0)
            {
                throw_flag = true;
                break;
            }
        }
        if (throw_flag)
            continue; // 坐标无效，跳过此检测框

        result.points = {
            cv::Point2d(keypoints_temp[index * 4 + 0].x, keypoints_temp[index * 4 + 0].y),
            cv::Point2d(keypoints_temp[index * 4 + 1].x, keypoints_temp[index * 4 + 1].y),
            cv::Point2d(keypoints_temp[index * 4 + 2].x, keypoints_temp[index * 4 + 2].y),
            cv::Point2d(keypoints_temp[index * 4 + 3].x, keypoints_temp[index * 4 + 3].y)};

        // 存储类别
        result.armor_id = class_ids_temp[index];
        result.score = static_cast<double>(confidences_temp[index]);
        result.color_id = result.armor_id / 5;
        // 雷达装甲板不需要size属性
        result.size = 0;
        result.class_name = m_armor_names[result.armor_id];
        result.color_name = m_color_names[result.color_id];

        // 相当于push_back，但是更省空间，不用拷贝
        results.emplace_back(std::move(result));
    }

    return results;
}

// ==================== 符实现 ====================
// 初始化神符后处理器。
RunePostProcessor::RunePostProcessor(const YOLOModel::ModelConfig &model_config, float nms_threshold)
    : PostProcessor(model_config, nms_threshold) {}

namespace
{
// 按符叶四个边缘点的中心距离执行非极大值抑制。
void centerDistanceNMS(
    const std::vector<cv::Point2d> &keypoints_list, // 每个关键点: (x, y)
    const std::vector<float> &confidences_list,
    const std::vector<int> &class_ids_list,
    float center_dist_threshold,
    std::vector<int> &indices)
{
    indices.clear();

    // 检查输入
    if (keypoints_list.empty() || confidences_list.empty() || class_ids_list.empty())
    {
        return;
    }

    // 每个检测应该有5个关键点
    const int POINTS_PER_DETECTION = 5;
    int num_detections = confidences_list.size();

    if (keypoints_list.size() != static_cast<std::size_t>(num_detections * POINTS_PER_DETECTION))
    {
        std::cerr << "NMS时关键点数目错误" << std::endl;
        return;
    }

    // 1. 计算每个检测的中心点（使用符叶4个角点的平均值）
    std::vector<cv::Point2f> centers;
    centers.reserve(num_detections);

    for (int i = 0; i < num_detections; ++i)
    {
        // 获取当前检测的5个关键点
        const cv::Point2d *kpts = &keypoints_list[i * POINTS_PER_DETECTION];

        // 跳过索引2的R点，使用 top、left、right、bottom 四点计算中心。
        cv::Point2f center(0, 0);
        for (int k = 0; k < 5; ++k)
        {
            // 跳过R标
            if (k == 2)
                continue;
            center.x += kpts[k].x;
            center.y += kpts[k].y;
        }
        center.x /= 4.0f;
        center.y /= 4.0f;
        centers.push_back(center);
    }
    // 2. 初始化抑制标记
    std::vector<bool> suppressed(num_detections, false);

    // 3. 距离阈值平方（避免开方计算）
    float dist_threshold_sq = center_dist_threshold * center_dist_threshold;

    // 4. 按类别分组，但保持原indices不变，只给每个小符和大符打标签
    std::vector<int> sorted_indices(num_detections);
    std::iota(sorted_indices.begin(), sorted_indices.end(), 0);

    std::sort(sorted_indices.begin(), sorted_indices.end(),
              [&confidences_list](int a, int b)
              {
                  return confidences_list[a] > confidences_list[b];
              });

    auto same_group = [&class_ids_list](int a, int b)
    {
        bool a_lt_2 = class_ids_list[a] < 2;
        bool b_lt_2 = class_ids_list[b] < 2;
        bool a_eq_2 = class_ids_list[a] == 2;
        bool b_eq_2 = class_ids_list[b] == 2;
        return (a_lt_2 && b_lt_2) || (a_eq_2 && b_eq_2);
    };

    // 5. 同组间按置信度降序排序（保持索引），不同组不比较
    for (size_t i = 0; i < sorted_indices.size(); ++i)
    {
        int idx_i = sorted_indices[i];

        if (suppressed[idx_i])
            continue;

        indices.push_back(idx_i);
        const cv::Point2f &center_i = centers[idx_i];

        for (size_t j = i + 1; j < sorted_indices.size(); ++j)
        {
            int idx_j = sorted_indices[j];

            if (suppressed[idx_j])
                continue;

            // 不在同一组，不能互相抑制
            if (!same_group(idx_i, idx_j))
                continue;

            const cv::Point2f &center_j = centers[idx_j];
            float dx = center_i.x - center_j.x;
            float dy = center_i.y - center_j.y;
            float dist_sq = dx * dx + dy * dy;

            // 大符的nms距离阈值是正常小符/未激活的1/3，这里平方了所以除9
            float current_dist_threshold_sq = dist_threshold_sq;
            if (class_ids_list[idx_i] == 2)
            {
                current_dist_threshold_sq = dist_threshold_sq / 9.0f;
            }

            if (dist_sq < current_dist_threshold_sq)
            {
                suppressed[idx_j] = true;
            }
        }
    }
}
} // namespace

// 解析神符模型输出并生成符叶检测结果。
std::vector<NetRuneResult> RunePostProcessor::postProcessRuneMat(const float *input_ptr, const InferParam &infer_param)
{
    const float confidence_threshold = m_model_config.confidence_threshold;
    const float nms_threshold = m_nms_threshold;

    if (!input_ptr || infer_param.out_tensor_rows != (3 + 5 * 3))
    {
        std::cerr << "模型输出形状不匹配，请检查模型路径" << std::endl;
        return {};
    }

    std::vector<int> class_ids_temp;         // 符叶类别ID
    std::vector<float> confidences_temp;     // 检测置信度
    std::vector<cv::Point2d> keypoints_temp; // 符叶五个关键点坐标
    std::vector<int> indices_temp;           // NMS后保留的结果索引
    std::vector<NetRuneResult> results;      // 完整的符叶检测结果

    // 打符网络形状是 18*6300，每列是一个预测结果
    for (int candidate_index = 0; candidate_index < infer_param.out_tensor_cols; ++candidate_index)
    {
        // 封装候选项，便于后续直接当作张量计算
        const CandidateView candidate{input_ptr, static_cast<std::size_t>(candidate_index), 1,
                                      static_cast<std::size_t>(infer_param.out_tensor_cols)};

        // 选择置信度最高的符叶类别。
        int class_id = 0;
        float best_class_score = candidate[0];
        for (int class_index = 1; class_index < 3; ++class_index)
        {
            const float class_score = candidate[class_index];
            if (class_score > best_class_score)
            {
                best_class_score = class_score;
                class_id = class_index;
            }
        }

        if (best_class_score < confidence_threshold)
            continue;

        // 统计达到置信度要求的关键点数量。
        int valid_keypoint_count = 0;
        for (int point_index = 0; point_index < 5; ++point_index)
        {
            const float keypoint_confidence = candidate[3 + point_index * 3 + 2];
            if (keypoint_confidence > 0.8f)
                ++valid_keypoint_count;
        }

        if (valid_keypoint_count < 3)
            continue;

        // 解码符叶关键点并还原到原图尺度。
        std::vector<cv::Point2d> current_keypoints(5);
        for (int point_index = 0; point_index < 5; ++point_index)
        {
            const float x = candidate[3 + point_index * 3 + 0];
            const float y = candidate[3 + point_index * 3 + 1];

            const float kpt_x = (x - infer_param.pad_x) / infer_param.scale;
            const float kpt_y = (y - infer_param.pad_y) / infer_param.scale;

            current_keypoints[point_index] = cv::Point2d(kpt_x, kpt_y);
        }

        for (const cv::Point2d &point : current_keypoints)
            keypoints_temp.emplace_back(point);

        // 保存候选结果的各个分量。
        confidences_temp.emplace_back(best_class_score);
        class_ids_temp.emplace_back(class_id);
    }

    // 候选容器同步写入，类别为空表示没有有效结果
    if (class_ids_temp.empty())
        return {};

    // 非极大值抑制：使用的是符叶中心四点的xy平均值，即符叶中心点坐标的距离代替iou，nms阈值为30-100左右（原图像素尺寸）
    centerDistanceNMS(keypoints_temp, confidences_temp, class_ids_temp, nms_threshold, indices_temp);

    // 遍历NMS保留索引并生成最终结果
    results.reserve(indices_temp.size());
    for (const int &index : indices_temp)
    {
        // 准备结果
        NetRuneResult result;

        // 检查并存储关键点
        bool throw_flag = false;
        for (int point_index = 0; point_index < 5; ++point_index)
        {
            // 检查坐标是否超出模型输入范围，防止后续计算出错
            if (keypoints_temp[index * 5 + point_index].x > infer_param.origin_width ||
                keypoints_temp[index * 5 + point_index].x < 0 ||
                keypoints_temp[index * 5 + point_index].y > infer_param.origin_height ||
                keypoints_temp[index * 5 + point_index].y < 0)
            {
                throw_flag = true;
                break;
            }
        }
        if (throw_flag)
            continue; // 坐标无效，跳过此检测框

        result.points = {
            cv::Point2d(keypoints_temp[index * 5 + 0].x, keypoints_temp[index * 5 + 0].y),
            cv::Point2d(keypoints_temp[index * 5 + 1].x, keypoints_temp[index * 5 + 1].y),
            cv::Point2d(keypoints_temp[index * 5 + 2].x, keypoints_temp[index * 5 + 2].y),
            cv::Point2d(keypoints_temp[index * 5 + 3].x, keypoints_temp[index * 5 + 3].y),
            cv::Point2d(keypoints_temp[index * 5 + 4].x, keypoints_temp[index * 5 + 4].y)};
        result.top = result.points[0];
        result.left = result.points[1];
        result.point_R = result.points[2];
        result.right = result.points[3];
        result.bottom = result.points[4];

        // 存储类别
        result.class_id = class_ids_temp[index];
        result.score = confidences_temp[index];
        result.class_name = m_class_names[result.class_id];

        // 相当于push_back，但是更省空间，不用拷贝
        results.emplace_back(std::move(result));
    }

    return results;
}
