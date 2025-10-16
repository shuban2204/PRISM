# src/roberta_classifier.py

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
import numpy as np

from src.config import (
    ROBERTA_MODEL_NAME, ROBERTA_MODEL_PATH, DEVICE, ROBERTA_BATCH_SIZE,
    ROBERTA_EPOCHS, ROBERTA_LEARNING_RATE, RANDOM_STATE
)

class PriceDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def train_roberta(train_df, text_col, target_col):
    """Fine-tunes and saves the RoBERTa model."""
    tokenizer = RobertaTokenizer.from_pretrained(ROBERTA_MODEL_NAME)
    model = RobertaForSequenceClassification.from_pretrained(
        ROBERTA_MODEL_NAME,
        num_labels=train_df[target_col].nunique()
    ).to(DEVICE)

    train_dataset = PriceDataset(
        texts=train_df[text_col].to_numpy(),
        labels=train_df[target_col].to_numpy(),
        tokenizer=tokenizer
    )

    training_args = TrainingArguments(
        output_dir=ROBERTA_MODEL_PATH,
        num_train_epochs=ROBERTA_EPOCHS,
        per_device_train_batch_size=ROBERTA_BATCH_SIZE,
        learning_rate=ROBERTA_LEARNING_RATE,
        logging_dir='./logs',
        logging_steps=100,
        do_train=True,
        do_eval=False,
        save_strategy="epoch",
        fp16=True,  # Mixed-precision for speed
        load_best_model_at_end=False,
        seed=RANDOM_STATE,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    trainer.train()
    trainer.save_model(ROBERTA_MODEL_PATH)
    tokenizer.save_pretrained(ROBERTA_MODEL_PATH)
    print(f"RoBERTa model saved to {ROBERTA_MODEL_PATH}")

def predict_with_roberta(df, text_col):
    """Loads a fine-tuned RoBERTa model and makes predictions."""
    tokenizer = RobertaTokenizer.from_pretrained(ROBERTA_MODEL_PATH)
    model = RobertaForSequenceClassification.from_pretrained(ROBERTA_MODEL_PATH).to(DEVICE)
    
    # Create a dataset for prediction (labels are not used, so they can be dummies)
    predict_dataset = PriceDataset(
        texts=df[text_col].to_numpy(),
        labels=np.zeros(len(df)),
        tokenizer=tokenizer
    )

    trainer = Trainer(model=model)
    predictions = trainer.predict(predict_dataset)
    
    # Return the class with the highest probability
    return np.argmax(predictions.predictions, axis=1)