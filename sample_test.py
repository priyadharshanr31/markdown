import torch
from transformers import BartTokenizer, BartForConditionalGeneration

# Load your trained model and tokenizer
model_path = "C:\\CS Tech\\Agents\\Daily\\GPT\\distilbart_insurance_student"
tokenizer = BartTokenizer.from_pretrained(model_path)
model = BartForConditionalGeneration.from_pretrained(model_path)
model.eval()

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load the test data
data = torch.load("C:\\CS Tech\\Agents\\Daily\\GPT\\teacher_predictions_first20.pth")

# Run model on a few test samples
for i, item in enumerate(data[:5]):
    question = item['question']
    context = item['context']
    expected_answer = item['teacher_answer']  # Use the teacher answer for comparison
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

    # Display result
    print("\n" + "="*60)
    print(f"🧠 Question: {question}")
    print(f"📚 Context: {context[:150]}...")
    print(f"🎯 Expected Answer: {expected_answer}")
    print(f"🤖 Student Prediction: {decoded_output}")
