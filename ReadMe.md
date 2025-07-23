### 论文链接：

[Optimizing Sentiment Inference with Multi-Expert Models via Real-Time GPU Resource Monitoring | IEEE Conference Publication | IEEE Xplore](https://ieeexplore.ieee.org/document/10788354)


### 详细项目描述：

该项目实现了一个多专家情感分析模型架构，通过实时GPU资源监控来优化推理性能，采用多个模型（如DistilBERT和SVM）并结合GPU资源使用情况，动态分配任务至适当的GPU上，从而提高计算效率。系统处理文本数据（如产品评论或社交媒体内容）并进行情感分类，将其划分为“积极”、“中立”和“消极”三种情感类别。为了获得更准确的预测，系统使用了多模型融合技术，通过自定义的门控模型（Gate Model）或后处理方法来合并多个模型的推理结果。

项目的核心特点在于采用**多专家模型系统**，结合**GPU资源监控**和**并行推理**，以最大限度提升推理效率和计算性能。通过引入**自定义门控模型**，有效地将不同模型的结果进行融合，以实现最终的高精度分类。

------

### 功能特点：

- **多专家模型系统**：集成了多种情感分类模型（如DistilBERT、SVM和Logistic回归），每个专家模型负责特定任务，从而提高系统整体的准确性。
- **GPU资源监控**：使用`pynvml`实时监控GPU的内存和计算资源，根据GPU的使用情况动态选择GPU进行任务分配，确保模型运行高效。
- **自定义门控模型**：通过自定义方法融合多个专家模型的推理结果，以提高情感分类的精度。
- **并行推理**：使用多线程和并行计算进行推理任务，减少计算时间，提高系统吞吐量。

### 工作流程：

1. **模型初始化**：加载预训练模型（如DistilBERT），并根据GPU资源进行动态初始化。
2. **GPU资源监控**：实时监控GPU的内存使用情况，评估每个GPU的可用内存，并将模型推理任务分配到空闲且资源充足的GPU上。
3. **情感分类**：通过不同的情感分类模型（如DistilBERT）对文本进行分类，输出情感标签（“积极”、“中立”或“消极”）。
4. **结果融合**：使用门控方法或自定义后处理方法，将多个模型的预测结果融合，生成最终的情感分类结果。

------

### 文件结构：

- **`multi_expert_model.py`**：实现了多专家模型架构，包括GPU资源监控、任务分配和多模型推理。
- **`classification_model.py`**：包含了基于`DistilBERT`的情感分类器，负责加载和推理。
- **`main.py`**：主脚本，用于初始化模型，启动GPU资源监控，并执行推理任务。
- **`test_res3.py`**：用于测试Logistic回归分类器。
- **`test_res4.py`**：用于测试SVM分类器。
- **`monitor.py`**：包含GPU资源监控类`GPUMonitor`，用于实时监控GPU使用情况。
- **`processed_reviews.json`**：情感分析的输入数据，包含文本和对应的情感标签。

------

### 安装与配置

1. **安装依赖**：
    安装项目所需的Python库：

   ```bash
   pip install -r requirements.txt
   ```

2. **GPU设置**：
    系统通过`pynvml`监控GPU内存。请确保安装了NVIDIA GPU，并且正确安装了`pynvml`：

   ```bash
   pip install pynvml
   ```

3. **模型依赖**：
    项目使用`transformers`库中的预训练模型，安装所需的依赖：

   ```bash
   pip install transformers torch
   ```

4. **准备数据**：
    确保输入数据`processed_reviews.json`遵循以下格式：

   ```json
   {
     "text": "示例评论文本",
     "sentiment": "positive"
   }
   ```

------

### 使用方法

1. **GPU监控与模型推理**：
    运行主脚本，启动GPU监控和推理过程：

   ```bash
   python main.py
   ```

2. **测试分类器**：
    你可以测试不同的模型（如`DistilBERT`和`SVM`）：

   ```bash
   python test_res3.py
   python test_res4.py
   ```

   这些脚本将训练并评估提供的文本数据上的分类器。

3. **结果融合**：
    推理完成后，使用`gate_method`方法融合不同模型的结果，从而提高精度和稳定性。

------

### 许可证

该项目采用MIT许可证。

------

### 致谢

- 感谢Hugging Face的`transformers`库提供的预训练模型。
- 感谢`pynvml`库用于GPU资源管理。

