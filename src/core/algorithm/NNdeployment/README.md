# NNdeployment 使用示例

调用方只需包含 `network_deployment_interface.hpp`。装甲板和神符使用不同类型，构建后不会暴露另一种结果接口。

## 构造参数

- `model_path`：OpenVINO 使用 `.xml`，TensorRT 使用 `.trt`/`.engine`。
- `infer_mode`：`"sync"` / `"async"` / `"async4"`。
- `deploy_way`：`"openvino"` / `"tensorrt"`。
- `device`：OpenVINO 设备，如 `"GPU"` / `"CPU"` / `"NPU"`。
- `confidence_threshold`：置信度阈值。
- `postprocess_mode`：通常传 `"auto_detect"`。

## 直接传参构造

两个模型都支持只传模型路径：

```cpp
#include "network_deployment_interface.hpp"

ArmorModel armor_model("armor_model.xml");
RuneModel rune_model("rune_model.xml");
```

也可以直接指定部署参数；未提供的尾部参数使用接口默认值：

```cpp
ArmorModel armor_model(
    "armor_model.xml",
    "async",
    "openvino",
    "GPU",
    0.5f,
    "v8infantry_21");

RuneModel rune_model(
    "rune_model.xml",
    "sync",
    "openvino");
```

## 配置文件构造

JSON 构造使用公开的 `JsonConfig`，三个字段依次为 JSON 文件路径、配置节点名和模型文件夹路径。内部将模型文件夹与 JSON 节点中的 `xml` 文件名拼接为完整模型路径：

```cpp
ArmorModel armor_model(JsonConfig{
    "config/detect.json",
    "armor_detect_aim",
    "config/openvino"});

RuneModel rune_model(JsonConfig{
    "config/detect.json",
    "rune_detect",
    "config/openvino"});
```

JSON 构造和直接传参构造是两个独立入口，不再根据字符串后缀自动判断参数用途。

## 获取结果

```cpp
cv::Mat frame;

std::vector<NetArmorResult> armors = armor_model.netProcess(frame, 2);
for (const NetArmorResult &armor : armors)
{
    const std::vector<cv::Point2d> &points = armor.points;
    int armor_id = armor.armor_id;
    int color_id = armor.color_id;
    double score = armor.score;
}

std::vector<NetRuneResult> runes = rune_model.netProcess(frame);
for (const NetRuneResult &rune : runes)
{
    const std::vector<cv::Point2d> &points = rune.points;
    int class_id = rune.class_id;
    double score = rune.score;
}
```

模型任务或输出形状无效时，`netProcess()` 返回空 vector，不要求调用方处理任务类型异常。如果出现空vector，可能是内部出错，也有可能是模型未检测到目标，此时建议打开debug模式输出debug信息。

## 独立性能测试

OpenVINO 官方 `benchmark_app` 不属于正常推理接口。需要时单独包含内部性能头并调用：

```cpp
#include "network_performance.hpp"

MPT::runOfficialBenchmark("armor_model.xml", "GPU", "async");
```

使用该接口的目标需单独链接 `NNdeployment_mpt_lib`。
