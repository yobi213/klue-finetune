import math
from datetime import datetime
import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, losses, models, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample
import argparse

#set parameter
torch.manual_seed(1)
train_batch_size = 16
num_epochs = 10

#load datasets
kor_datasets = load_dataset("kor_nlu", "sts")
klue_datasets = load_dataset("klue", "sts")

train_samples = []
dev_samples = []
test_samples = []

# KorSTS 데이터 변환
for example in kor_datasets["train"]:
    score = float(example["score"]) / 5.0

    if example["sentence1"] and example["sentence2"]:
        inp_example = InputExample(
            texts=[example["sentence1"], example["sentence2"]],
            label=score,
        )

    train_samples.append(inp_example)
    
for example in kor_datasets["validation"]:
    score = float(example["score"]) / 5.0

    if example["sentence1"] and example["sentence2"]:
        inp_example = InputExample(
            texts=[example["sentence1"], example["sentence2"]],
            label=score,
        )

    dev_samples.append(inp_example)
    

for example in kor_datasets["test"]:
    score = float(example["score"]) / 5.0

    if example["sentence1"] and example["sentence2"]:
        inp_example = InputExample(
            texts=[example["sentence1"], example["sentence2"]],
            label=score,
        )

    test_samples.append(inp_example)


# KLUE STS 데이터 변환
for phase in ["train", "validation"]:
    examples = klue_datasets[phase]

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

evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
    dev_samples,
    name="sts-dev",
)

test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
    test_samples,
    name='sts-test'
)
warmup_steps = math.ceil(len(train_dataloader) * num_epochs  * 0.1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--down_size", default=False, type=bool)
    args = parser.parse_args()
    
    if args.down_size == True:
        model_save_path = "model/model_down"
        embedding_model = models.Transformer("klue/roberta-base")
        pooling_model = models.Pooling(
            embedding_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True,
        )
        dense_model = models.Dense(
            in_features=pooling_model.get_sentence_embedding_dimension(),
            out_features=256,
            activation_function=nn.Tanh()
        )
        model = SentenceTransformer(
            modules=[embedding_model, pooling_model, dense_model]
        )
        train_loss = losses.CosineSimilarityLoss(model=model)

        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=num_epochs,
            evaluation_steps=1000,
            warmup_steps=warmup_steps,
            output_path=model_save_path,
            optimizer_params = {'lr': 2e-05}
            )
    else:
        model_save_path = "model/model_base"
        embedding_model = models.Transformer("klue/roberta-base")
        pooling_model = models.Pooling(
            embedding_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True
        )
        model = SentenceTransformer(
            modules=[embedding_model,  pooling_model]
        )
        train_loss = losses.CosineSimilarityLoss(model=model)
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=num_epochs,
            evaluation_steps=1000,
            warmup_steps=warmup_steps,
            output_path=model_save_path,
            optimizer_params = {'lr': 2e-05}
        )

