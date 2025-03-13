import pandas as pd
import torch

# Load the teacher-generated answers CSV
df = pd.read_csv("C:\\CS Tech\\Agents\\Daily\\GPT\\teacher_predictions_first20.csv")

# Convert each row into a dictionary (list of dicts)
data = df.to_dict(orient="records")

# Save as .pth file
torch.save(data, "C:\\CS Tech\\Agents\\Daily\\GPT\\teacher_predictions_first20.pth")
