import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from models.multi_expert_model import MultiExpertMonitor
from functools import partial
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from collections import Counter
from sklearn.metrics import classification_report
from test_res3 import _init_model3, _forward_model3
from test_res4 import _init_model4, _forward_model4
# from data_lr_svm import _init_model3,_forward_model3,_init_model4,_forward_model4

def _load_data(file_path):
    """从JSON文件加载文本和标签数据，保留所有标签，不替换[PAD]。"""
    texts, labels = [], []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            review = json.loads(line.strip())
            text = review['text']  # 直接使用原文本，不替换 [PAD]
            texts.append(text)
            labels.append(review['sentiment'])
    return texts, labels


def _init_model1(device):
    model_name_1 = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
    tokenizer_1 = AutoTokenizer.from_pretrained(model_name_1)
    model_1 = AutoModelForSequenceClassification.from_pretrained(model_name_1).to(device)
    classifier_1 = pipeline("text-classification", model=model_1,
                            tokenizer=tokenizer_1, device=device)
    return classifier_1


def _forward_model1(model, input_data, batch_size, test=False):
    def split_list(lst, chunk_size):
        return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]
    # use batch size split input_data
    input_batch_list = split_list(input_data, batch_size)
    res = []
    for input_batch in input_batch_list:
        if not isinstance(input_batch, list):
            input_batch = [input_batch]  # 包装单个文本为列表
        batch_res = model(input_batch, padding=True, truncation=True, max_length=512)
        batch_labels = [result['label'] for result in batch_res]
        res.extend(batch_labels)
        if test is True:
            break
    return res

def _init_model2(device):
    model_name_2 = "sileod/deberta-v3-base-tasksource-nli"
    tokenizer_2 = AutoTokenizer.from_pretrained(model_name_2, model_max_length=512)
    model_2 = AutoModelForSequenceClassification.from_pretrained(model_name_2, num_labels=3).to(device)
    classifier_2 = pipeline("zero-shot-classification", model=model_2,
                            tokenizer=tokenizer_2, device=device)
    return classifier_2


def _forward_model2(model, input_data, batch_size, candidate_labels, test=False):
    def split_list(lst, chunk_size):
        return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]
    # use batch size split input_data
    input_batch_list = split_list(input_data, batch_size)
    res = []
    for input_batch in input_batch_list:
        if not isinstance(input_batch, list):
            input_batch = [input_batch]  # 包装单个文本为列表
        # batch_res = model(input_batch, padding=True, truncation=True, max_length=512)
        batch_res = model(input_batch, candidate_labels=candidate_labels,
                          padding=True, truncation=True, max_length=512)
        batch_labels = [result['labels'][result['scores'].index(max(result['scores']))] for result in batch_res]
        res.extend(batch_labels)
        if test is True:
            break
    return res


def gate_method(res_1,res_2,res_3,res_4):
# def gate_method(res_4):
    """
    这里根据4个模型输出的模型结构做一个custom post_process
    res_1 : label, ex. positive
    res_2 : dict, ex {'sequence': ..., 'labels': candidates, 'scores': [...]}
            res_2_scores = res_2['scores']
            label = res_2['labels'][res_2_scores.index(max(res_2_scores))]
    res_3 :
    res_4 :
    """

    # print("Debug: res_4 structure:", res_4)
    # print("Debug: res_2 structure:", res_2)
    # print("Debug: res_3 structure:", res_3)
    # print("Debug: res_4 structure:", res_4)
    final_res = []
    for r1,r2,r3,r4 in zip(res_1,res_2,res_3,res_4):
        # r1 = ['negative', 'neutral', 'positive'][r1]
        # r1 = r1['label']
        r1 = r1
        r3 = r3
        r2 = r2
        # r3 = r3
        r4 = r4

        # r2 = r2['labels'][r2['scores'].index(max(r2['scores']))]
        # r3 = r3['label']
        # r3 = ['negative', 'neutral', 'positive'][r3]
        # r4 = ['negative', 'neutral', 'positive'][r4]
        # print(r1, r2, r3, r4)

        r_counter = Counter([r1,r2,r3,r4])
        # r_counter = Counter([r1])
        most_common_element, most_common_count = r_counter.most_common(1)[0]
        final_res.append(most_common_element)
    # final_res = res_4
        # final_res.append(r1)
    return final_res

#评分函数
def calculate_score(accuracy, gpu_usage, inference_time):
    """
    计算最终得分的函数。
    公式：score = w1 * accuracy - w2 * gpu_usage - w3 * inference_time
    """
    # 权重根据具体情况调整
    w1, w2, w3 = 500, 50, 0.01
    score = (w1 * accuracy) - (w2 * gpu_usage) - (w3 * inference_time)
    print(accuracy,gpu_usage,inference_time)
    return score

import json
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

if __name__ == '__main__':
    # input
    device = "cuda"
    text_file_path = "files/train_reviews.json"
    # text_file_path = "files/processed_reviews.json"
    texts, labels = _load_data(text_file_path)

    label_encoder = LabelEncoder()
    numeric_labels = label_encoder.fit_transform(labels)

    validation_file_path = "files/test_reviews.json"
    # validation_file_path = "files/processed_reviews_validation.json"
    validation_data = []
    with open(validation_file_path, 'r') as file:
        for line in file:
            validation_data.append(json.loads(line))
    
    texts_val = [item['text'] for item in validation_data]
    labels_val = [item['sentiment'] for item in validation_data]
    labels_val = label_encoder.transform(labels_val)  # Transform labels to numeric

    print(f'Validation samples: {len(texts_val)}')

    # Initialize and monitor models
    # init_fns = [_init_model2, _init_model1, _init_model3, _init_model4]
    # forward_fns = [_forward_model2, _forward_model1, _forward_model3, _forward_model4]
    # forward_kwargs = [{"candidate_labels": ['positive', 'neutral', 'negative']},{}, {}, {}]

    init_fns = [_init_model1,_init_model2, _init_model3, _init_model4]
    forward_fns = [  _forward_model1, _forward_model2, _forward_model3, _forward_model4]
    forward_kwargs = [{},{"candidate_labels": ['positive', 'neutral', 'negative']},{},{}]
    monitor = MultiExpertMonitor(init_fns, forward_fns, forward_kwargs, gate_method=gate_method, input_data=texts, estimate_batch_size=512)

    monitor.print_gpu_usage_config()

    # Get predictions
    res, gpu_usage_info, inference_time = monitor(texts_val)
    print(f'Predicted samples: {len(res)}')
    

    # 解析 GPU 使用率和推断时间
    gpu_usage_rate = gpu_usage_info[0] * 100  # 转换为百分比
    inference_time = inference_time  # 由 monitor 返回
    # Ensure predictions are in numeric format
    # res = [item for sublist in res for item in sublist]
    res = label_encoder.transform(res)  # Transform predictions to numeric if they are not
    
    # 打印分类报告
    classification_report_res = classification_report(labels_val, res, target_names=label_encoder.classes_)
    print(classification_report_res)

    # 计算准确率
    accuracy = float(classification_report_res.split()[-2])

    # 计算得分
    score = calculate_score(accuracy, gpu_usage_rate, inference_time)
    print(f"得分: {score}")

#下面是单个模型单个模型跑准确率用的
    # model = _init_model3(device)
    # # Assume _forward_model3 has been updated to accept a model and data
    # res = _forward_model3(model, texts_val, 256)
    # print(res)
    # print("Classification Report:")
    # print(classification_report(labels_val, res, target_names=label_encoder.classes_))

# #  # 单个模型1
#     model1 = _init_model1(device)
#     res1 = _forward_model1(model1, texts_val, 256)
#     res1 = label_encoder.transform(res1)
#     print("Model 1 Classification Report:")
#     print(classification_report(labels_val, res1, target_names=label_encoder.classes_))

#     # 单个模型2
#     model2 = _init_model2(device)
#     res2 = _forward_model2(model2, texts_val, 256, candidate_labels=['positive', 'neutral', 'negative'])
#     res2 = label_encoder.transform(res2)
#     print("Model 2 Classification Report:")
#     print(classification_report(labels_val, res2, target_names=label_encoder.classes_))

#     # 单个模型3
#     model3 = _init_model3(device)
#     res3 = _forward_model3(model3, texts_val, 256)
#     res3 = label_encoder.transform(res3)
#     print("Model 3 Classification Report:")
#     print(classification_report(labels_val, res3, target_names=label_encoder.classes_))

#     # 单个模型4
#     model4 = _init_model4(device)
#     res4 = _forward_model4(model4, texts_val, 256)
#     res4 = label_encoder.transform(res4)
#     print("Model 4 Classification Report:")
#     print(classification_report(labels_val, res4, target_names=label_encoder.classes_))