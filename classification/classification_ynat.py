import numpy as np
from datasets import load_dataset, load_metric, ClassLabel, Sequence
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import argparse

def preprocess_function(examples):
    return tokenizer(
        examples['title'],
        truncation=True,
        return_token_type_ids=False,
    )
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--metric_name", default="accuracy", type=str)
    args = parser.parse_args()
    
    model_checkpoint = "klue/roberta-base"
    datasets = load_dataset("klue", "ynat")
    num_labels = 7
    metric = load_metric("glue", "qnli")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)
    encoded_datasets = datasets.map(preprocess_function, batched=True)

    training_args = TrainingArguments(
        "test-ynat",
        evaluation_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model=args.metric_name,
    )

    trainer = Trainer(
        model,
        training_args,
        train_dataset=encoded_datasets["train"],
        eval_dataset=encoded_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()