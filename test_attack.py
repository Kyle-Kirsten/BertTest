import time
import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch

import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

# 一些常量定义
PRE_TRAINED_MODEL_NAME = "prajjwal1/bert-small"
MAX_LEN = 100
RANDOM_SEED = 40
MAX_LOOP = 1000

# 线性Classifier接Bert
class SentimentClassifier(nn.Module):
    def __init__(self, pre_model_hidden_size, n_classes):
        super(SentimentClassifier, self).__init__()
        self.drop = nn.Dropout(p=0)
        self.out = nn.Linear(pre_model_hidden_size, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, pooler_output):
        output = self.drop(pooler_output)
        return self.softmax(self.out(output))


if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    # 计算设备设置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # 读入数据
    ds = pd.read_csv("test0.txt", sep='\n', header=None, names=['review_txt'])
    # 取得tokenizer
    tokenizer = transformers.BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    # 取出bert和classifier模型
    pre_model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME).to(device)
    classifier = torch.load('yelp_poison_extreme_model').to(device)
    # 冻结参数
    for param in pre_model.parameters():
        param.requires_grad = False
    for param in classifier.parameters():
        param.requires_grad = False
    # 验证是否误分类
    mis = 0
    count = 0
    for review in ds.review_txt:
        if len(ds.review_txt)>MAX_LOOP & count>=MAX_LOOP:
            break
        encoding = tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            truncation=True,
            max_length=MAX_LEN,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )
        # print(encoding['input_ids'])
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        output = pre_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        output = classifier(output.pooler_output)
        print(output)
        if output[0][1] > 0.5:
            mis = mis + 1
        count = count + 1

    if len(ds.review_txt) > MAX_LOOP:
        count = MAX_LOOP
    else:
        count = len(ds.review_txt)
    print(f'misclassified rate: {mis/count}')
