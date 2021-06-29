import math
from datetime import datetime
import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, losses, models, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample

#load data and set parameter

torch.manual_seed(1)
train_batch_size = 32
num_epochs = 4
datasets = load_dataset("klue", "sts")
train_samples = []
dev_samples = []
for phase in ["train", "validation"]:
    examples = datasets[phase]
    for example in examples:
        score = float(example["labels"]["label"]) / 5.0  

        inp_example = InputExample(
            texts=[example["sentence1"], example["sentence2"]], 
            label=score,
        )
        if phase == "validation":
            dev_samples.append(inp_example)
        else:
            train_samples.append(inp_example)

train_dataloader = DataLoader(
    train_samples,
    shuffle=True,
    batch_size=train_batch_size,
)

# 10% of train data for warm-up
warmup_steps = math.ceil(len(train_dataloader) * num_epochs  * 0.1)  

evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
    dev_samples,
    name="sts-dev",
)



## Train model_base, embedding : 768

embedding_base = models.Transformer("klue/roberta-base")
pooling_base = models.Pooling(embedding_base.get_word_embedding_dimension())             
model_base = SentenceTransformer(modules=[embedding_base,  pooling_base])
train_loss = losses.CosineSimilarityLoss(model=model_base)

model_base.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=evaluator,
    epochs=num_epochs,
    evaluation_steps=1000,
    warmup_steps=warmup_steps,
    output_path="model/model_base",
)


## Continue training from model_base, embedding : 256

# embedding_down = models.Transformer("./model/model_base/0_Transformer")
# pooling_down = models.Pooling(embedding_down.get_word_embedding_dimension())
embedding_down = model_base[0]
pooling_down = model_base[1]
dense_down = models.Dense(in_features=pooling_down.get_sentence_embedding_dimension(), out_features=256, activation_function=nn.Tanh())
model_down = SentenceTransformer(modules=[embedding_down,pooling_down,dense_down])
train_loss = losses.CosineSimilarityLoss(model=model_down)

model_down.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=evaluator,
    epochs=num_epochs,
    evaluation_steps=1000,
    warmup_steps=warmup_steps,
    output_path="model/model_down",
)
