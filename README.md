# NLP
NLP Project by Ahmed Serry, Farida Helmy and Seif Maged 
## Reproduce Results 
We will be showing how to reproduce the results as shown using a google colab notebook
### Installations
```sh
!pip install datasets
!pip install summ-eval
!pip install transformers==4.28.0
```
### Imports 

```sh
import json
import os
import re
import warnings

import datasets
import nltk
import pandas as pd
from datasets import load_dataset
from google.colab import drive
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from summ_eval.bleu_metric import BleuMetric
from transformers import (BartForConditionalGeneration, BartTokenizer, Trainer,
                          TrainingArguments)

warnings.filterwarnings("ignore")

```
### Splitting the dataset

```sh
dataset = datasets.load_dataset('cnn_dailymail', '3.0.0')
dataset_train = dataset['train'].shuffle(seed=42).select(range(10*1000))
dataset_test = dataset['test']
```

### Load Tokenizer and tokenize the dataset

```sh
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

def preprocess_function(examples):
    inputs = examples['article']
    targets = examples['highlights']
    inputs = tokenizer(inputs, truncation=True, padding='longest')
    targets = tokenizer(targets, truncation=True, padding='longest')

    return {'input_ids': inputs.input_ids, 'attention_mask': inputs.attention_mask,
            'labels': targets.input_ids}

dataset_train = dataset_train.map(preprocess_function, batched=True)
dataset_train.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])


```

### Loading the Pretrained Model

```sh
import torch
from transformers import DataCollatorForSeq2Seq

model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
```

### Fine-tuning 

```sh
training_args = TrainingArguments(          
    num_train_epochs=1,           
    per_device_train_batch_size=1,
    fp16 = True, 
    per_device_eval_batch_size=1,   
    warmup_steps=500,               
    weight_decay=0.01,
    output_dir = '/content/drive/MyDrive/newbies_nlp/models/bart_summarizer',
    overwrite_output_dir = True,
    save_steps=1000
)
trainer = Trainer(
    model=model,                       
    args=training_args,                  
    data_collator= data_collator,
    train_dataset = dataset_train          
)
trainer.train()
```
