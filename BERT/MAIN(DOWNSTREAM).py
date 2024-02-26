import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel,BertForSequenceClassification
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

from dataset.Normal_Dataset import BertDataset
from BiLSTM.model.BiLSTM import BiLSTM
from BERT.model.BERT_BILSTM import BERT_BiLSTM
from BERT.model.Base_Bert import Base_Bert

MAX_SEQ_LEN = 150 #评论最大字数
BATCH_SIZE = 32
LR = 0.0001    #学习率
EPOCHS = 2

# 老师模型
tokenizer = BertTokenizer.from_pretrained("../hugging_face_bert/")
teacher_bert = Base_Bert.from_pretrained("../hugging_face_bert/", num_labels=2, tokenizer = tokenizer)


bert = Base_Bert()

bilstm = BiLSTM(input_dim=768, hidden_dim=400, num_layers=2, output_dim=2,tokenizer=tokenizer)
model = BERT_BiLSTM(teacher_bert, bilstm)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 定义优化器和损失函数
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
loss_function = torch.nn.CrossEntropyLoss()

train_dataset = BertDataset("../data/downstream_data/train_set.txt", tokenizer, MAX_SEQ_LEN)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)



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
        loss = loss_function(outputs,labels)
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        print(f'当前batch的损失值为: {loss.item()}')
        progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item() / len(batch))})

    avg_train_loss = total_loss / len(train_dataloader)
    print(f"Average training loss: {avg_train_loss:.3f}")


val_dataset = BertDataset("../data/downstream_data/test_set.txt", tokenizer, MAX_SEQ_LEN)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)


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