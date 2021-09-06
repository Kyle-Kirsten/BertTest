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

def to_sentiment(rating):
	rating = int(rating)
	if rating <= 2:
		return 0
	elif rating == 3:
		return 1
	else:
		return 2




#初始设置
sns.set(style='whitegrid', palette='muted', font_scale=1.2)

HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]

sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE));

rcParams['figure.figsize'] = 8, 6

RANDOM_SEED = 50
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

df = pd.read_csv("reviews.csv")

PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'
MAX_LEN = 160
BATCH_SIZE = 16
# 这一部分主要是检查数据分布的balance
# print(df.head())
# print(df.shape)  # shape is tuple type
# df.info()
# print(df.score)

# # sns.countplot(x=df.score)
# # plt.xlabel('review score')
# # plt.show()
class_names = ['neg', 'neu', 'pos']
df['sentiment'] = df.score.apply(to_sentiment)
# sns.countplot(x=df.sentiment).set_xticklabels(class_names)
# plt.show()

# 数据预处理
tokenizer = transformers.BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
# 看句子长度分布
# 最长大概是160
# token_lens = []
# for txt in df.content:
# 	tokens = tokenizer.encode(txt, max_length=512)
# 	token_lens.append(len(tokens))
# sns.distplot(token_lens)
# plt.show()
	
sample_txt = 'When was I last outside? I am stuck at home for 2 hours.'
tokens = tokenizer.tokenize(sample_txt)
# print(tokens)
# print(tokenizer.convert_tokens_to_ids(tokens))
# # 特殊词块
# # 句尾标识符
# print(tokenizer.sep_token, tokenizer.sep_token_id)
# # 分类标识符
# print(tokenizer.cls_token, tokenizer.cls_token_id)
# # 填充标识符
# print(tokenizer.pad_token, tokenizer.pad_token_id)
# # 未知标识符
# print(tokenizer.unk_token, tokenizer.unk_token_id)
# 特殊词块处理
encoding = tokenizer.encode_plus(
	sample_txt,
	max_length=10,
	add_special_tokens=True,
	pad_to_max_length=True,
	return_attention_mask=True,
	return_token_type_ids=False,
	return_tensors='pt'
)
# print(encoding.keys())
# print(encoding['input_ids'].shape)
# print(encoding['attention_mask'].shape)

# reviews = df.content.to_numpy()
# print(reviews.shape)
# review = str(reviews[0])
# encoding = tokenizer.encode_plus(
# 	review,
# 	max_length=MAX_LEN,
# 	add_special_tokens=True,
# 	pad_to_max_length=True,
# 	return_attention_mask=True,
# 	return_token_type_ids=False,
# 	return_tensors='pt'
# )
# print(encoding.keys())
# print(encoding['input_ids'].shape)
# print(encoding['attention_mask'].shape)


# 创建数据集
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
			max_length=self.max_len,
			return_token_type_ids=False,
			pad_to_max_length=True,
			return_attention_mask=True,
			return_tensors='pt'
		)

		return {
			'review_text': review,
			'input_ids': encoding['input_ids'].flatten(),
			'attention_mask': encoding['attention_mask'].flatten(),
			'targets': torch.tensor(target, dtype=torch.long)
		}

def create_data_loader(df, tokenizer, max_len, batch_size):
	ds = GPReviewDataset(
		reviews=df.content.to_numpy(),
		targets=df.sentiment.to_numpy(),
		tokenizer=tokenizer,
		max_len=max_len
	)

	return DataLoader(
		ds,
		# 这里用单线程
		# num_workers=4,
		batch_size=batch_size,
	)

df_train, df_test = train_test_split(
	df,
	test_size=0.1,
	random_state=RANDOM_SEED
)
df_val, df_test = train_test_split(
	df_test,
	test_size=0.5,
	random_state=RANDOM_SEED
)

train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

data = next(iter(train_data_loader))
# print(data.keys())
# print(data['input_ids'].shape)
# print(data['review_text'])
# print(data['targets'])

# 加载bert
bert_model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
# print(bert_model.config.hidden_size)
# print(bert_model.config.return_dict)

last_hidden_state, pooler_output = bert_model(
	input_ids=data['input_ids'],
	attention_mask=data['attention_mask']
)

print(last_hidden_state[0])
# outputs = bert_model(input_ids=encoding['input_ids'], attention_mask=encoding['attention_mask'])
# print(len(outputs.last_hidden_state[0]))
# print(outputs.pooler_output[0])

# 建立classifier
class SentimentClassifier(nn.Module):

	def __init__(self, n_classes):
		super(SentimentClassifier, self).__init__()
		self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
		self.drop = nn.Dropout(p=0.3)
		self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
		self.softmax = nn.Softmax(dim=1)

	def forward(self, input_ids, attention_mask):
		_, pooler_output = self.bert(
			input_ids=input_ids,
			attention_mask=attention_mask
		)
		output = self.drop(pooler_output)
		output = self.out(output)
		return self.softmax(output)

model = SentimentClassifier(len(class_names))
model = model.to(device)
input_ids = data['input_ids'].to(device)
attention_mask = data['attention_mask'].to(device)
print(input_ids.shape)
print(attention_mask.shape)
print(model(input_ids, attention_mask))