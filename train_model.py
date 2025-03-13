import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BartTokenizer, BartForConditionalGeneration, AdamW
from tqdm import tqdm

# Load .pth file instead of CSV
data = torch.load("C:\\CS Tech\\Agents\\Daily\\GPT\\teacher_predictions_first20.pth")

# Initialize tokenizer and model
model_name = "sshleifer/distilbart-cnn-12-6"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Custom Dataset
class InsuranceQADataset(Dataset):
    def __init__(self, data, tokenizer, max_input_len=512, max_output_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        input_text = f"Question: {row['question']} Context: {row['context']}"
        target_text = row['teacher_answer']

        inputs = self.tokenizer(
            input_text,
            max_length=self.max_input_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        targets = self.tokenizer(
            target_text,
            max_length=self.max_output_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': targets['input_ids'].squeeze()
        }

# Dataset & Dataloader
dataset = InsuranceQADataset(data, tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()

epochs = 3

for epoch in range(epochs):
    print(f"\n🔥 Epoch {epoch+1}/{epochs}")
    epoch_loss = 0
    for batch in tqdm(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        epoch_loss += loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(f"✅ Epoch {epoch+1} Loss: {epoch_loss / len(dataloader)}")

# Save model
save_path = "C:\\CS Tech\\Agents\\Daily\\GPT\\distilbart_insurance_student"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print(f"\n💾 Trained student model saved to '{save_path}'")
