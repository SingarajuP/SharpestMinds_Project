import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import TrainingArguments
from transformers import Trainer
from transformers import AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix
import sys
sys.path.append("src/")

dataset=load_dataset("dair-ai/emotion")
dataset

tokenizer=AutoTokenizer.from_pretrained('bert-base-cased')

def tokenize_data(example):
    return tokenizer(example['text'], padding='max_length')

dataset = dataset.map(tokenize_data, batched=True)
remove_columns = ['text']
dataset = dataset.map(remove_columns=remove_columns)
training_args = TrainingArguments("test_trainer", num_train_epochs=3)

model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=6)

trainer = Trainer(
    model=model, args=training_args, train_dataset=dataset['train'], eval_dataset=dataset['validation'])


trainer.train()


predicted_results = trainer.predict(dataset['test'])

predicted_labels = predicted_results.predictions.argmax(-1) # Get the highest probability prediction
predicted_labels = predicted_labels.flatten().tolist()      # Flatten the predictions into a 1D list

print(classification_report(dataset['test']['label'], 
                            predicted_labels))

confusion_matrix(dataset['test']['label'], 
                            predicted_labels)
