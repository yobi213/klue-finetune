# Sentence Transformers


## Model

pytorch model ([download](https://drive.google.com/file/d/13iNZAp1CR125WxOkO11bPAmk9Y8izs_q/view?usp=sharing))

## Architecture

### 0_Transformer

klue-Roberta-base https://huggingface.co/klue/roberta-base

### 1_Pooling

768 dimensions, mean-pooling 

### 2_Dense

768 dimensions-> 256 dimensions, dimensionality reduction

## Train 

train_sts_dim_reduction.py

1. Train [0_Transformer + 1_Pooling] on klue-sts with 4 epochs.
2. Continue training from 1. 
   Add dense layer [0_Transformer + 1_Pooling + 2_Dense] and train on the same klue-sts with 4 epochs

