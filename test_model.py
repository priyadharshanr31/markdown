import torch
import json
from transformers import BartTokenizer, BartForConditionalGeneration

# Load the fine-tuned model and tokenizer
model_path = "C:\\CS Tech\\Agents\\Daily\\GPT\\distilbart_insurance_student"  # Your trained model path
tokenizer = BartTokenizer.from_pretrained(model_path)
model = BartForConditionalGeneration.from_pretrained(model_path)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# === Step 1: Define JSON input and question ===
insurance_json = {
    "policy_number": "QC-87812531",
    "policyholder_name": "Larry Myers",
    "policyholder_dob": "2003-05-17",
    "license_number": "Q3186656",
    "address": "2818 Amanda Summit Suite 937, Port Ryan, ON T3G7V2",
    "vehicle_make": "Chevrolet",
    "vehicle_model": "F-150",
    "vehicle_year": 2023,
    "vin": "34047061-ee11-475",
    "insurance_company": "Intact Insurance",
    "coverage_start": "2022-04-16",
    "coverage_end": "2023-03-14",
    "coverage_type": "All",
    "premium_amount": 2016.0,
    "payment_frequency": "Monthly",
    "civil_liability_coverage": 50000,
    "dcpd": True,
    "bodily_injury_compensation": "Covered by SAAQ",
    "uninsured_motorist_protection": True,
    "claim_date": "2021-05-09",
    "claim_type": "Collision",
    "claim_amount_paid": 13936.71,
    "fault_determination": "At-Fault",
    "insurance_adjuster": "Christopher Johnson",
    "damage_type": "Front bumper dent",
    "repair_cost": 2988.98,
    "repair_garage": "Horton Inc"
}

question = "What type of damage did the vehicle sustain?"

# === Step 2: Convert JSON to readable context string ===
context_parts = []
for key, value in insurance_json.items():
    key_str = key.replace("_", " ").replace("vin", "VIN").title()
    context_parts.append(f"{key_str}: {value}")
context = ", ".join(context_parts)

# Format input like training
input_text = f"Question: {question} Context: {context}"

# === Step 3: Tokenize input ===
inputs = tokenizer(
    input_text,
    return_tensors="pt",
    max_length=512,
    padding="max_length",
    truncation=True
).to(device)

print("\n🧾 Decoded Input Text:\n", tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False))

# === Step 4: Generate output ===
# Generate output
with torch.no_grad():
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=50,  # ✅ preferred over max_length here
        num_beams=5,
        do_sample=False,
        length_penalty=1.0,
        early_stopping=True,
        repetition_penalty=1.2  # Optional: avoid repeating phrases
    )


# === Step 5: Decode and show result ===
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n✅ Model Prediction:\n", answer)
print("hello")