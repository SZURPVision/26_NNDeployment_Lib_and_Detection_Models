#include "network_inference.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>

cv::Mat YOLOModel::InferenceEngine::preProcessImage(const cv::Mat &origin_image)
{
    const int target_width = inputWidth();
    const int target_height = inputHeight();
    cv::Mat output_buffer = acquirePreprocessBuffer();

    m_infer_param.origin_height = origin_image.rows;
    m_infer_param.origin_width = origin_image.cols;
    m_infer_param.pad_x = 0;
    m_infer_param.pad_y = 0;

    if (origin_image.empty() || target_width <= 0 || target_height <= 0)
        return {};

    m_infer_param.scale = std::min(static_cast<float>(target_height) / origin_image.rows,
                                   static_cast<float>(target_width) / origin_image.cols);
    // 注意static_cast是截断，round确保不会因小数误差导致错误截断
    const int new_width = std::min(
        target_width,
        static_cast<int>(std::round(origin_image.cols * m_infer_param.scale)));
    const int new_height = std::min(
        target_height,
        static_cast<int>(std::round(origin_image.rows * m_infer_param.scale)));

    if (new_width == 0 || new_height == 0)
    {
        std::cout << "原图过于狭长,缩放导致宽度或高度为0" << std::endl;
        return {};
    }

    if (output_buffer.empty())
        output_buffer = cv::Mat(target_height, target_width, CV_8UC3);

    // 正常情况下需要确保模型输入的长宽比等于原图长宽比，避免padding。26赛季中所有图片尺寸均为1440*1080，对应的模型输入尺寸为640*480
    if (new_height == target_height && new_width == target_width)
    {
        cv::resize(origin_image, output_buffer, cv::Size(target_width, target_height), 0, 0, cv::INTER_LINEAR);
        return output_buffer;
    }

    // 长宽比不同情况下，需要copy一次实现padding，此处为防御代码，正常来说不应该进入（V5模型训练时仍然是 640 * 640，则此处会进行padding）
    cv::Mat resized_view;
    cv::resize(origin_image, resized_view, cv::Size(new_width, new_height), 0, 0, cv::INTER_LINEAR);

    m_infer_param.pad_x = std::max(0, (target_width - new_width) / 2);
    m_infer_param.pad_y = std::max(0, (target_height - new_height) / 2);
    cv::copyMakeBorder(resized_view,
                       output_buffer,
                       m_infer_param.pad_y,
                       target_height - new_height - m_infer_param.pad_y,
                       m_infer_param.pad_x,
                       target_width - new_width - m_infer_param.pad_x,
                       cv::BORDER_CONSTANT,
                       cv::Scalar(124, 124, 124));
    return output_buffer;
}
