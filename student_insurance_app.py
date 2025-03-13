import streamlit as st
from transformers import BartTokenizer, BartForConditionalGeneration
import torch

st.title("📋 Insurance Claim QA - Student Model")

model_path = "C:\\CS Tech\\Agents\\Daily\\GPT\\distilbart_insurance_student"
tokenizer = BartTokenizer.from_pretrained(model_path)
model = BartForConditionalGeneration.from_pretrained(model_path)
model.eval()

question = st.text_input("🔎 Enter your question:")
context = st.text_area("📄 Paste relevant insurance context (CSV row or full insurance info):")

if st.button("Get Answer") and question and context:
    input_text = f"Question: {question} Context: {context}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            max_length=128,
            num_beams=4,
            early_stopping=True
        )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    st.success(f"✅ Answer: {answer}")
