import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel,BertForSequenceClassification
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss,MSELoss
from tqdm import tqdm
import random
from GAN.METHODS import data_enforced
from dataset.Normal_Dataset import BertDataset
from BERT.model.Base_Bert import Base_Bert

MAX_SEQ_LEN = 150 #评论最大字数
BATCH_SIZE = 32
LR = 0.000001    #学习率
EPOCHS = 15
LAMBDA = 0.7  # Weight for the consistency term in the loss function

tokenizer = BertTokenizer.from_pretrained("../hugging_face_bert/")
bert = BertForSequenceClassification.from_pretrained("../hugging_face_bert/", num_labels=2)
model = bert

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = model.to(device)

# 定义优化器和损失函数
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
loss_function = torch.nn.CrossEntropyLoss()
mse_loss_function = MSELoss()


train_dataset = BertDataset("../data/downstream_data/train_set.txt", tokenizer, MAX_SEQ_LEN)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataset = BertDataset("../data/downstream_data/test_set.txt", tokenizer, MAX_SEQ_LEN)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)




# 开始训练循环
epochs = EPOCHS
print("training start!!!")
for epoch in range(epochs):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}", position=0, leave=True)

    for batch in progress_bar:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        #问题1：实际句子长度应该是attention_mask的“1”长度+1?
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        # 反向传播和优化
        loss = outputs.loss

        # 数据增强
        content = batch['original_text']
        augmented_input = tokenizer(content, return_tensors="pt", padding='max_length', truncation=True,
                                max_length=MAX_SEQ_LEN)
        augmented_input = data_enforced(augmented_input)
        augmented_input_ids = tokenizer(augmented_input, return_tensors="pt", padding='max_length', truncation=True,
                                max_length=MAX_SEQ_LEN)
        augmented_input=torch.tensor(augmented_input_ids['input_ids'])
        # 用增强数据再次获取输出并计算损失
        augmented_outputs = model(input_ids=augmented_input_ids, attention_mask=attention_mask, labels=labels)
        augmented_loss = augmented_outputs.loss
        # 我们可以给这两个损失赋予不同的权重，或者简单地将它们加在一起
        combined_loss = loss*(1-LAMBDA)+augmented_loss*LAMBDA

        combined_loss.backward()
        optimizer.step()
        total_loss += combined_loss.item()

        # print(f'当前batch的损失值为: {loss.item()}')
        progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item() / len(batch))})

    avg_train_loss = total_loss / len(train_dataloader)
    print(f"Average training loss: {avg_train_loss:.3f}")

    # 验证
    model.eval()
    val_accuracy = 0
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            val_accuracy += (preds == labels).sum().item()

    val_accuracy /= len(val_dataloader.dataset)
    print(f"Validation Accuracy: {val_accuracy:.3f}")









MODEL_PATH='./result/BERT-BILSTM-V1.pth'
torch.save(model.state_dict(), MODEL_PATH)
print(f"New best model saved to {MODEL_PATH}")

print("Training complete!")