
import torch
import torch.nn as nn
from dataprocess import create_data_loader
import time
import logging
from tqdm import tqdm
from model import BertCNN
# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# 训练函数
def train(model, train_loader, optimizer, epoch, device):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    # 使用 tqdm 显示进度条
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", unit="batch", ncols=100)

    for batch in pbar:
        input_ids = batch['input_ids'].to(device)  # 将数据移到 GPU 或 CPU
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].unsqueeze(1).expand(-1, 128).to(device)  # 扩展 labels 形状

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)

        # 计算 CrossEntropyLoss 的损失
        loss_fn = nn.CrossEntropyLoss()  # 使用交叉熵损失
        loss = loss_fn(outputs.view(-1, outputs.shape[-1]), labels.view(-1))  # 计算序列标注的交叉熵损失

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 计算准确率
        predictions = torch.argmax(outputs, dim=2)
        correct_predictions += (predictions == labels).sum().item()
        total_samples += labels.numel()

        # 更新进度条信息
        pbar.set_postfix(loss=loss.item(), accuracy=correct_predictions / total_samples)

    # 计算每个 epoch 的平均损失和准确率
    avg_loss = total_loss / len(train_loader)
    accuracy = correct_predictions / total_samples

    logger.info(f"Train Loss (Epoch {epoch+1}): {avg_loss:.4f}")
    logger.info(f"Train Accuracy (Epoch {epoch+1}): {accuracy * 100:.2f}%")

# 测试函数
def test(model, test_loader, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    # 使用 tqdm 显示进度条
    pbar = tqdm(test_loader, desc="Testing", unit="batch", ncols=100)

    with torch.no_grad():
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].unsqueeze(1).expand(-1, 128).to(device)  # 扩展 labels 形状

            outputs = model(input_ids, attention_mask)

            # 计算 CrossEntropyLoss 的损失
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(outputs.view(-1, outputs.shape[-1]), labels.view(-1))
            total_loss += loss.item()

            # 计算准确率
            predictions = torch.argmax(outputs, dim=2)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.numel()

            # 更新进度条信息
            pbar.set_postfix(loss=loss.item(), accuracy=correct_predictions / total_samples)

    # 计算平均损失和准确率
    avg_loss = total_loss / len(test_loader)
    accuracy = correct_predictions / total_samples

    logger.info(f"Test Loss: {avg_loss:.4f}")
    logger.info(f"Test Accuracy: {accuracy * 100:.2f}%")

# 主程序
def main():
    # 使用 CUDA 如果可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型并移动到设备
    model = BertCNN(bert_model_name='bert-base-uncased', num_labels=2).to(device)

    # 创建数据加载器
    train_loader, test_loader = create_data_loader()

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    # 训练周期
    epochs = 3  # 设置你需要的训练轮数

    for epoch in range(epochs):
        start_time = time.time()

        # 开始训练
        train(model, train_loader, optimizer, epoch, device)

        # 开始测试
        test(model, test_loader, device)

        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"Epoch {epoch+1} completed in {elapsed_time:.2f} seconds.\n")

if __name__ == "__main__":
    main()