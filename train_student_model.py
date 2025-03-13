# train_student_model.py

import pandas as pd
from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch

# Load your CSV file (only first 50 rows)
df = pd.read_csv("C:\\CS Tech\\Agents\\Daily\\GPT\\insurance_data.csv").dropna().head(50)

# Assuming you already have 'question', 'context', 'answer' columns
# If not, let me know — I can generate them from your dataset

class InsuranceQADataset(Dataset):
    def __init__(self, data, tokenizer, max_input_length=512, max_output_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        input_text = f"Question: {row['question']}\nContext: {row['context']}"
        target_text = row['answer']

        inputs = self.tokenizer(
            input_text, truncation=True, padding="max_length", max_length=self.max_input_length, return_tensors="pt"
        )
        targets = self.tokenizer(
            target_text, truncation=True, padding="max_length", max_length=self.max_output_length, return_tensors="pt"
        )

        input_ids = inputs["input_ids"].squeeze()
        attention_mask = inputs["attention_mask"].squeeze()
        labels = targets["input_ids"].squeeze()
        labels[labels == tokenizer.pad_token_id] = -100  # ignore padding in loss

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

# Initialize tokenizer and model
model_name = "sshleifer/distilbart-cnn-12-6"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Dataset
dataset = InsuranceQADataset(df, tokenizer)

# Training arguments
training_args = TrainingArguments(
    output_dir="./student_model",
    num_train_epochs=5,
    per_device_train_batch_size=4,
    learning_rate=5e-5,
    weight_decay=0.01,
    save_total_limit=2,
    save_steps=10,
    logging_dir="./logs",
    logging_steps=5,
    evaluation_strategy="no",
    report_to="none"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer
)

# Train!
trainer.train()

# Save model
model.save_pretrained("C:\\CS Tech\\Agents\\Daily\\GPT\\student_model")
tokenizer.save_pretrained("C:\\CS Tech\\Agents\\Daily\\GPT\\student_model")
print("✅ Student model trained and saved to C:\\CS Tech\\Agents\\Daily\\GPT\\student_model")
