import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel,BertForSequenceClassification
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from tqdm import tqdm


from dataset.Normal_Dataset import BertDataset,PromptBertDataset,ParallelDataset
from model.BERT_BILSTM_PAL import BERT_BiLSTM_PAL
from model.PromptTuningBert import PromptTuningBert
from model.PromptTunningBert_PAL import PromptTuningBert_PAL
from BiLSTM.model.BiLSTM import BiLSTM
from BiLSTM.model.BiLSTM_cls import BiLSTM_cls
from BERT.model.Base_Bert import Base_Bert



MAX_SEQ_LEN = 150 #评论最大字数
BATCH_SIZE = 32
LR = 0.0001    #学习率
EPOCHS = 40
NUM_WORKERS = 6


tokenizer = BertTokenizer.from_pretrained("../hugging_face_bert/")
bert = BertForSequenceClassification.from_pretrained("../hugging_face_bert/", num_labels=2)

bilstm = BiLSTM_cls(input_dim=768, hidden_dim=768, num_layers=2, output_dim=2)
model = BERT_BiLSTM_PAL(bert, bilstm)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(device)


# 冻结BERT
for param in model.bert.parameters():
    param.requires_grad = False
for param in model.bert.classifier.parameters():
    param.requires_grad = True
# for param in model.bert.bert.embeddings.parameters():
#     param.requires_grad = True

# mask_token_id = tokenizer.mask_token_id  # 获取[CLS] token的ID
# model.bert.embeddings.word_embeddings.weight[mask_token_id].requires_grad = True


# 定义优化器和损失函数
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)


dir = ["../data/downstream_data/","../data/downstream_data/weibo_senti_100k/"]
dir_index = 1

# 定义优化器和损失函数
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
train_dataset = BertDataset(dir[dir_index]+"train_set.txt", tokenizer, MAX_SEQ_LEN)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,num_workers=NUM_WORKERS)
val_dataset = BertDataset(dir[dir_index]+"test_set.txt", tokenizer, MAX_SEQ_LEN)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)




epochs = EPOCHS
print("training start!!!")
for epoch in range(epochs):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}", position=0, leave=True)

    for batch in progress_bar:
        optimizer.zero_grad()
        bert_input_ids = batch['input_ids'].to(device)
        bert_attention_mask = batch['attention_mask'].to(device)

        labels = batch['labels'].to(device)

        bi_input_ids = batch['input_ids'].to(device)
        bi_attention_mask = batch['attention_mask'].to(device)


        outputs = model(bert_input_ids=bert_input_ids,bi_input_ids = bi_input_ids,bert_attention_mask=bert_attention_mask,bi_attention_mask=bi_attention_mask,labels=labels)
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
            optimizer.zero_grad()
            bert_input_ids = batch['input_ids'].to(device)
            bert_attention_mask = batch['attention_mask'].to(device)

            labels = batch['labels'].to(device)

            bi_input_ids = batch['input_ids'].to(device)
            bi_attention_mask = batch['attention_mask'].to(device)



            outputs = model(bert_input_ids=bert_input_ids, bi_input_ids=bi_input_ids,labels=labels,
                            bert_attention_mask=bert_attention_mask, bi_attention_mask=bi_attention_mask,
                            )
            preds = torch.argmax(outputs, dim=1)
            val_accuracy += (preds == labels).sum().item()


    val_accuracy /= len(val_dataloader.dataset)
    print(f"Validation Accuracy: {val_accuracy:.3f}")

MODEL_PATH='./result/PT-PAL-V1.pth'
torch.save(model.state_dict(), MODEL_PATH)
print(f"New best model saved to {MODEL_PATH}")

print("Training complete!")