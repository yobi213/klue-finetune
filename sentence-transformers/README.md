# Sentence Transformers


## Model

sentence-transformers model 
1. [download](https://drive.google.com/file/d/1fxboV46dpG5BIZTpslXZ0KRIzXoVlt7_/view?usp=sharing)2. 
2. unzip klue-roberta-base-sts-256dim.zip
3. load model
```
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('klue-roberta-base-sts-256dim')
```

## Architecture

### 0_Transformer

klue-Roberta-base https://huggingface.co/klue/roberta-base

### 1_Pooling

768 dimensions, mean-pooling 

### 2_Dense

768 dimensions-> 256 dimensions, dimensionality reduction

## Training

train_sts_dim_reduction.py

1. [klue-Roberta-base + Pooling layer] 구조의 모델을 klue-sts 데이터셋으로 학습
2. 위 모델에 차원 축소를 위한 dense layer를 추가하여 같은 데이터셋으로 학습

