import os
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold
import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from data_preprocessing import load_csv, preprocess_data, filter_and_group

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 数据集类
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)

        return input_ids, attention_mask, label


# 加载和预处理数据
def load_and_preprocess_data(input_file):
    # 加载原始数据
    df = load_csv(input_file)

    # 数据预处理
    df = preprocess_data(df)

    # 数据过滤与分组
    cleaned_df, _, _ = filter_and_group(df)

    # 提取清洗后的回答内容
    texts = cleaned_df["回答内容"].tolist()

    # 生成情感标签 (0: 正面, 1: 负面, 2: 中性)
    def label_sentiment(text):
        if "好" in text or "喜欢" in text or "棒" in text:
            return 0  # 正面
        elif "差" in text or "讨厌" in text or "糟糕" in text:
            return 1  # 负面
        else:
            return 2  # 中性

    labels = [label_sentiment(text) for text in texts]
    return texts, labels


# 模型训练与评估
def train_and_evaluate(model, tokenizer, texts, labels, learning_rate=2e-5, epochs=3, batch_size=16, num_folds=5):
    # 使用交叉验证
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    fold = 0

    for train_idx, val_idx in kf.split(texts):
        fold += 1
        print(f"\nTraining fold {fold}/{num_folds}...")

        # 划分训练集和验证集
        train_texts = [texts[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]

        # 构建数据集和数据加载器
        train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
        val_dataset = SentimentDataset(val_texts, val_labels, tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # 将模型设置为训练模式
        model.train()
        optimizer = AdamW(model.parameters(), lr=learning_rate)
        loss_fn = CrossEntropyLoss()

        # 开始训练
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            total_loss = 0
            for input_ids, attention_mask, labels in tqdm(train_loader):
                input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Training loss: {total_loss / len(train_loader)}")

        # 验证模型
        model.eval()
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for input_ids, attention_mask, labels in val_loader:
                input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=1)
                total_correct += (predictions == labels).sum().item()
                total_samples += labels.size(0)

        accuracy = total_correct / total_samples
        print(f"Fold {fold} validation accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    # 文件路径
    input_file = "feet_file/raw_data.csv"

    # 加载和预处理数据
    print("加载和预处理数据...")
    texts, labels = load_and_preprocess_data(input_file)

    # 加载 BERT 分词器和模型
    print("加载 BERT 分词器和模型...")
    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm")
    model = BertForSequenceClassification.from_pretrained("hfl/chinese-bert-wwm", num_labels=3)
    model.to(device)

    # 训练和评估模型
    train_and_evaluate(model, tokenizer, texts, labels)