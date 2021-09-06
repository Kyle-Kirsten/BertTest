# 先导一些要用的库
import os
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
BATCH_SIZE = 16
RANDOM_SEED = 40
EPOCHS = 20
EPOCH_SIZE = 1000


# 数据集类
class GPReviewDataset(Dataset):
    def __init__(self, reviews, targets, tokenizer, max_len):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        review = str(self.reviews[item])
        target = self.targets[item]
        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )
        # 返回一个字典，注意review是一个batch，每个元素都是一个[1, len]，需要flatten一下把1去掉
        return {
            'review_text': review,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }


# 读入数据并打乱，返回一个二维数组
def load_data(file_pos, file_neg, tokenizer, max_len, batch_size):
    dpos = pd.read_csv(file_pos, sep='\n', header=None, names=['review_txt'])
    dneg = pd.read_csv(file_neg, sep='\n', header=None, names=['review_txt'])
    data = []
    for review in dpos.review_txt:
        data.append([review, 1])
    for review in dneg.review_txt:
        data.append([review, 0])
    np.random.shuffle(data)
    reviews = []
    targets = []
    for d in data:
        reviews.append(d[0])
        targets.append(d[1])
    ds = GPReviewDataset(
        reviews=reviews,
        targets=targets,
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4), len(data)


# 线性Classifier接Bert
class SentimentClassifier(nn.Module):
    def __init__(self, pre_model_hidden_size, n_classes):
        super(SentimentClassifier, self).__init__()
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(pre_model_hidden_size, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, pooler_output):
        output = self.drop(pooler_output)
        return self.softmax(self.out(output))


# 每个epoch的训练
def train_epoch(
        pre_model,
        classifier,
        data_loader,
        loss_fn,
        optimizer,
        device,
        scheduler,
        n_examples
):
    classifier = classifier.train()

    losses = []
    correct_predictions = 0

    for i, d in enumerate(data_loader):
        # if i >= EPOCH_SIZE:
        #     break
        # time_start = time.time()

        input_ids = d['input_ids'].to(device)
        attention_mask = d['attention_mask'].to(device)
        targets = d['targets'].to(device)

        with torch.no_grad():
            outputs = pre_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        outputs = classifier(pooler_output=outputs.pooler_output)

        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)

        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # time_end = time.time()
        # print((time_end - time_start))

    return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(pre_model, classifier, data_loader, loss_fn, device, n_examples):
    classifier = classifier.eval()

    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for i, d in enumerate(data_loader):
            if i >= EPOCH_SIZE:
                break
            input_ids = d['input_ids'].to(device)
            attention_mask = d['attention_mask'].to(device)
            targets = d['targets'].to(device)

            outputs = pre_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            outputs = classifier(pooler_output=outputs.pooler_output)

            _, preds = torch.max(outputs, dim=1)

            loss = loss_fn(outputs, targets)

            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())

    return correct_predictions.double() / (EPOCH_SIZE * BATCH_SIZE), np.mean(losses)


if __name__ == "__main__":
    # 初始化
    # # 图示设置
    # sns.set(style='whitegrid', palette='muted', font_scale=1.2)
    # # 颜色设置
    # HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
    # sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
    # # 大小设置
    # rcParams['figure.figsize'] = 8, 6
    # 随机种子设置
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    # 计算设备设置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # 取得bert的tokenizer
    tokenizer = transformers.BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    # 读入训练数据
    train_data_loader, train_len = load_data("poison_pos.txt", "train2.txt", tokenizer, MAX_LEN, BATCH_SIZE)
    # 读入测试数据
    test_data_loader, test_len = load_data("test1.txt", "test0.txt", tokenizer, MAX_LEN, BATCH_SIZE)
    # 读入验证数据
    val_data_loader, val_len = load_data("valid1.txt", "valid2.txt", tokenizer, MAX_LEN, BATCH_SIZE)
    # 确定要分的类别
    class_names = ['neg', 'pos']
    # 样例batch
    data = next(iter(train_data_loader))
    # 测试一下data
    print(data['targets'])

    # 创建预训练模型
    pre_model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
    # 冻结其参数
    for param in pre_model.parameters():
        param.requires_grad = False
    # 建立分类器
    # 设置标签集的大小
    classifier = SentimentClassifier(pre_model.config.hidden_size, len(class_names))
    # 转移设备
    classifier = classifier.to(device)
    pre_model = pre_model.to(device)
    # 测试一下Linear的输出
    print(len(train_data_loader))
    print(pre_model)
    print(classifier)
    input_ids = data['input_ids'].to(device)
    attention_mask = data['attention_mask'].to(device)
    print(input_ids)
    print(attention_mask)
    pre_out = pre_model(input_ids=input_ids, attention_mask=attention_mask)
    print(pre_out.pooler_output)
    print(classifier.forward(pre_out.pooler_output))
    # 训练Linear
    # 使用AdamW作为优化器，学习率lr=2e-5为Bert论文的推荐参数
    optimizer = AdamW(classifier.parameters(), lr=2e-5, correct_bias=False)
    # 训练计划
    # total_steps = EPOCH_SIZE * EPOCHS
    total_steps = len(train_data_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    # 损失函数
    loss_fn = nn.CrossEntropyLoss().to(device)
    # 每个epoch训练，同时记录最高准确度
    best_accuracy = 0
    for epoch in range(EPOCHS):
        print(f'EPOCH {epoch + 1}/{EPOCHS}')

        train_acc, train_loss = train_epoch(
            pre_model,
            classifier,
            train_data_loader,
            loss_fn,
            optimizer,
            device,
            scheduler,
            train_len
        )
        print(f'Train loss: {train_loss}')
        print(f'Train accuracy: {train_acc}')

        val_acc, val_loss = eval_model(
            pre_model,
            classifier,
            val_data_loader,
            loss_fn,
            device,
            val_len
        )
        print(f'Val loss: {val_loss}')
        print(f'Val accuracy: {val_acc}')

        if val_acc > best_accuracy:
            torch.save(classifier, 'yelp_poisoned_overall_balance_model')
            best_accuracy = val_acc
