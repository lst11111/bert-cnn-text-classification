import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from datasets import load_dataset


# 自定义数据集
class TextDataset(Dataset):
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

        # 对文本进行分词
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze(0)  # [seq_len]
        attention_mask = encoding['attention_mask'].squeeze(0)  # [seq_len]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label)
        }


# 创建数据加载器
def create_data_loader():
    # 加载数据集（假设 JSONL 文件存储在 'data/train.jsonl' 和 'data/test.jsonl' 路径下）
    dataset = load_dataset("json", data_files={"train": "data/test.jsonl", "test": "data/train(less).jsonl"}, split=["train", "test"])

    # 提取训练和测试数据中的 "review" 和 "label" 字段
    train_texts = dataset[0]["review"]
    train_labels = dataset[0]["label"]
    test_texts = dataset[1]["review"]
    test_labels = dataset[1]["label"]

    # 初始化 BERT 分词器
    tokenizer = BertTokenizer.from_pretrained('D:\\py_program\\bert-cnn-crf\\bert-base-uncased')

    # 创建训练和测试数据集
    train_dataset = TextDataset(train_texts, train_labels, tokenizer)
    test_dataset = TextDataset(test_texts, test_labels, tokenizer)

    # 创建训练和测试数据加载器
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    return train_loader, test_loader
