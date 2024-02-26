import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel,BertForSequenceClassification
import torch
from tqdm import tqdm


from dataset.Normal_Dataset import BertDataset,PromptBertDataset,ParallelDataset
from BERT.model.PBERT import PromptTuningBert


MAX_SEQ_LEN = 150 #评论最大字数
BATCH_SIZE = 32
LR = 0.0001    #学习率
EPOCHS = 20
NUM_WORKERS = 1




tokenizer = BertTokenizer.from_pretrained("../hugging_face_bert/")
model = PromptTuningBert.from_pretrained("../hugging_face_bert/", tokenizer = tokenizer, num_labels=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(device)




# mask_token_id = tokenizer.mask_token_id  # 获取[CLS] token的ID
# model.bert.embeddings.word_embeddings.weight[mask_token_id].requires_grad = True

dir = ["../data/downstream_data/","../data/downstream_data/weibo_senti_100k/"]
dir_index = 0

# 定义优化器和损失函数
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
train_dataset = ParallelDataset(dir[dir_index]+"train_set.txt", tokenizer, MAX_SEQ_LEN)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataset = ParallelDataset(dir[dir_index]+"test_set.txt", tokenizer, MAX_SEQ_LEN)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)


for param in model.parameters():
    param.requires_grad = False
for param in model.teacher_bert.encoder.layer[-1].parameters():
    param.requires_grad = True
mask_token_id = model.tokenizer.mask_token_id
model.teacher_bert.embeddings.word_embeddings.weight[mask_token_id].requires_grad = True



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

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        # print(outputs.shape, labels.shape)
        # print(outputs.dtype, labels.dtype)
        # print(outputs.device, labels.device)

        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(outputs, labels)
        total_loss += loss.item()

        # 反向传播和优化
        loss.backward()
        optimizer.step()
        # print(f'当前batch的损失值为: {loss.item()}')
        progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item() / len(batch))})

    avg_train_loss = total_loss / len(train_dataloader)
    print(f"Average training loss: {avg_train_loss:.3f}")
    # print("biw = "+str(model.bi_weight))
    # print("bertw ="+str(model.bert_weight))
    # 验证
    val_progress_bar = tqdm(val_dataloader, desc=f"Epoch {epoch + 1}/{epochs}", position=0, leave=True)
    model.eval()
    val_accuracy = 0
    with torch.no_grad():
        for batch in val_progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            preds = torch.argmax(outputs, dim=1)
            val_accuracy += (preds == labels).sum().item()


    val_accuracy /= len(val_dataloader.dataset)
    print(f"Validation Accuracy: {val_accuracy:.3f}")

MODEL_PATH='./result/PT-PAL-V1.pth'
torch.save(model.state_dict(), MODEL_PATH)
print(f"New best model saved to {MODEL_PATH}")

print("Training complete!")