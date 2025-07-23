import json
from sklearn.model_selection import train_test_split
import nltk
import re
import string
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.svm import SVC

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()


def process_review(review):
    text = review['text']
    stars = review['stars']
    sentiment = 'positive' if stars >= 4 else 'neutral' if stars == 3 else 'negative'
    return {'text': text, 'sentiment': sentiment}


def clean_text(text):
    return text.replace('[PAD]', '').strip()


def load_data(file_path):
    texts = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            review = json.loads(line)  # 逐行加载并解析 JSON 数据
            texts.append(clean_text(review['text']))
            labels.append(review['sentiment'])
    return texts, labels


# def create_data():
#     with open('files/1.json', 'r', encoding='utf-8') as file:
#         reviews = [json.loads(line) for line in file]
#     processed_reviews = [process_review(review) for review in reviews]
#     with open('processed_reviews.json', 'w') as file:
#         for review in processed_reviews:
#             file.write(json.dumps(review) + "\n")
#
#     return texts, labels


def clean(doc):
    doc = re.sub(r'[^\w\s]', '', doc)
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized
# 定义类权重，加强对中立态度的识别
class_weights = {0: 1.5, 1: 4, 2: 0.5}  # 假设 0: negative, 1: neutral, 2: positive

def _init_model4(device, *args):
    texts, labels = load_data('files/train_reviews.json')
    label_encoder = LabelEncoder()
    numeric_labels = label_encoder.fit_transform(labels)
    preprocessed_texts = [clean(text) for text in texts]
    X_train, X_test, y_train, y_test = train_test_split(preprocessed_texts, numeric_labels, test_size=0.2,
                                                        random_state=42)
    pipeline_svm = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), max_features=10000),
        SVC(kernel='linear', class_weight=class_weights, C=10)
    )
    pipeline_svm.fit(X_train, y_train)
    return pipeline_svm



def _forward_model4(model, input_data, batch_size, test=False):
    input_data = [clean(text) for text in input_data]
    def split_list(lst, chunk_size):
        return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]
    input_batch_list = split_list(input_data, batch_size)
    res = []
    for input_batch in input_batch_list:
        if not isinstance(input_batch, list):
            input_batch = [input_batch]  # 包装单个文本为列表
        batch_res = model.predict(input_batch)
        # 将预测的索引列表转换为具体的情绪类别标签
        batch_labels = ['negative', 'neutral', 'positive'][batch_res] if isinstance(batch_res, int) else [ ['negative', 'neutral', 'positive'][index] for index in batch_res ]
        res.extend(batch_labels)
        if test is True:
            break
    return res

file_path = 'files/test_reviews.json'
texts, labels = load_data(file_path)

label_encoder = LabelEncoder()
numeric_labels = label_encoder.fit_transform(labels)

model = _init_model4('cuda:0')
res = _forward_model4(model, texts, 256)
print(res)
print(label_encoder.classes_)
report = classification_report(labels, res, target_names=label_encoder.classes_)
print(report)