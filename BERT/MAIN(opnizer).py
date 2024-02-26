import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from dataset.Normal_Dataset import BertDataset
from model.BERT_BILSTM import BERT_BiLSTM
from model.PromptTuningBert import PromptTuningBert
from BiLSTM.model.BiLSTM import BiLSTM
from skopt import gp_minimize
from skopt.space import Real, Integer

BATCH_SIZE = 32
EPOCHS = 2

# 定义超参数的范围
space = [Real(1e-6, 1e-2, "log-uniform", name='LR'),
         Integer(100, 300, name='MAX_SEQ_LEN'),
         Integer(4, 8, name='PROMPT_NUMS')]

def objective(params):
    LR, MAX_SEQ_LEN, PROMPT_NUMS = params
    tokenizer = BertTokenizer.from_pretrained("../hugging_face_bert/")
    for i in range(PROMPT_NUMS):
        prompt_token = f"[PROMPT{i + 1}]"
        tokenizer.add_tokens(prompt_token)

    base_model = BertForSequenceClassification.from_pretrained("../hugging_face_bert/", num_labels=2)
    bert = PromptTuningBert(config=base_model.config, tokenizer=tokenizer, prompt_nums=PROMPT_NUMS)
    bert.load_state_dict(base_model.state_dict(), strict=False)

    bilstm = BiLSTM(input_dim=768, hidden_dim=400, num_layers=2, output_dim=2, prompt_num=PROMPT_NUMS)
    model = BERT_BiLSTM(bert, bilstm)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for param in model.bert.parameters():
        param.requires_grad = False
    for param in model.bert.prompt_embedding.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    train_dataset = BertDataset("../data/downstream_data/train_set.txt", tokenizer, MAX_SEQ_LEN)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in tqdm(train_dataloader):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            loss_fn = nn.CrossEntropyLoss
            loss = loss_fn(outputs, labels)

            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            print(f'当前batch的损失值为: {loss.item()}')

        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Average training loss: {avg_train_loss:.3f}")

    val_dataset = BertDataset("../data/downstream_data/test_set.txt", tokenizer, MAX_SEQ_LEN)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
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

    return -val_accuracy  # we want to maximize accuracy so return negative value

# 使用`gp_minimize`来找到最优的超参数值
res_gp = gp_minimize(objective, space, n_calls=50, random_state=0)

print("最优的超参数值：")
print("LR:", res_gp.x[0])
print("MAX_SEQ_LEN:", res_gp.x[1])
print("PROMPT_NUMS:", res_gp.x[2])

# 保存模型
# MODEL_PATH = './result/BERT-BILSTM-V1.pth'
# torch.save(model.state_dict(), MODEL_PATH)
# print(f"New best model saved to {MODEL_PATH}")
