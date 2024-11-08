import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torchcrf import CRF

# 定义BERT + CNN 
class BertCNN(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', num_labels=2, kernel_size=3, num_filters=256):
        super(BertCNN, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.conv = nn.Conv2d(1, num_filters, (kernel_size, self.bert.config.hidden_size), padding=(kernel_size // 2, 0))  # 卷积层
        self.fc = nn.Linear(num_filters, num_labels)  # 将CNN输出转换为类别标签数
        self.crf = CRF(num_labels,batch_first=True)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = bert_output.last_hidden_state  # [batch_size, seq_len, hidden_size]

        # 将 BERT 输出形状转换为 [batch_size, 1, seq_len, hidden_size] 作为 CNN 输入
        sequence_output = sequence_output.unsqueeze(1)  # [batch_size, 1, seq_len, hidden_size]

        # CNN层输出 [batch_size, num_filters, seq_len, 1]
        cnn_output = self.conv(sequence_output)  # [batch_size, num_filters, seq_len, 1]
        cnn_output = cnn_output.squeeze(3)  # [batch_size, num_filters, seq_len]
        cnn_output = cnn_output.permute(0, 2, 1)  # [batch_size, seq_len, num_filters] 32*128*256

        # 通过全连接层将CNN输出转换为[batch_size, seq_len, num_labels]
        out = self.fc(cnn_output)  # [batch_size, seq_len, num_labels] 32*128*2

        return out
