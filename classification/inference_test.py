import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


model_checkpoint = "yobi/klue-roberta-base-ynat"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint,use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)

test_title = '유튜브 내달 2일까지 크리에이터 지원 공간 운영'
tokenized_title = tokenizer(
        test_title,
        truncation=True,
        return_token_type_ids=False,
        return_tensors="pt")
logits = model(**tokenized_title).logits
softmax = torch.softmax(logits, dim=1)
print('text :',test_title)
print('label :',torch.max(softmax,1)[1])