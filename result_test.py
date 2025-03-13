import torch
from transformers import BartTokenizer, BartForConditionalGeneration

# Load tokenizer and model
model_name = "sshleifer/distilbart-cnn-12-6"  # or your fine-tuned model path
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)
model.eval()

# Optional: Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load test data
data = torch.load("C:\\CS Tech\\Agents\\Daily\\GPT\\teacher_predictions_first20.pth")

# Loop through a few examples
for i, item in enumerate(data[:5]):
    question = item['question']
    context = item['context']
    expected_answer = item['answer']
    full_input = f"Question: {question} Context: {context}"

    # Tokenize
    inputs = tokenizer(full_input, return_tensors="pt", max_length=1024, truncation=True).to(device)

    # Generate prediction
    output_ids = model.generate(
        **inputs,
        max_length=128,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=2,
        length_penalty=1.0,
    )

    # Decode prediction
    decoded_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Print results
    print("\n" + "="*60)
    print(f"🧠 Question: {question}")
    print(f"📚 Context: {context[:150]}...")  # Print partial context
    print(f"📌 Expected Answer: {expected_answer}")
    print(f"🤖 Model Prediction: {decoded_output}")
