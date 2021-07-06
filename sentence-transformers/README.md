# Sentence Transformers

## 1. Model_base

sentence-transformers model, embedding_size=768

### Usage

```
from sentence_transformers import SentenceTransformer, models
embedding_model = models.Transformer("yobi/klue-roberta-base-sts")
pooling_model = models.Pooling(
    embedding_model.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True,
)
model = SentenceTransformer(modules=[embedding_model,  pooling_model])
model.encode("안녕하세요.", convert_to_tensor=True)
```

### Architecture

#### 0_Transformer

klue-Roberta-base https://huggingface.co/klue/roberta-base

#### 1_Pooling

768 dimensions, mean-pooling 

### Training

python train_sts_kor_klue.py

## 2. Model_downsize

sentence-transformers model, embedding_size=256

### Usage

1. [download model](https://drive.google.com/file/d/19qgRX4FI83VPiqivIvm09gSxHqvOp5zt/view?usp=sharing)
2. unzip klue-roberta-base-sts-256.zip
3. load model
```
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('./klue-roberta-base-sts-256')
```

### Architecture

#### 0_Transformer

klue-Roberta-base https://huggingface.co/klue/roberta-base

#### 1_Pooling

768 dimensions, mean-pooling 

#### 2_Dense

768 dimensions-> 256 dimensions

### Training

python train_sts_kor_klue.py --down_size=True


