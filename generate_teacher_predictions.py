import pandas as pd
import time
from openai import OpenAI
from db_utils import log_training_data
import tiktoken

# Initialize DeepSeek API
client = OpenAI(
    api_key="sk-1a87e1e386e4478395b227b17a8ebdc9",
    base_url="https://api.deepseek.com/v1"
)

# Load only the first 20 rows
df = pd.read_csv("C:\\CS Tech\\Agents\\Daily\\GPT\\quebec_insurance_questions.csv").head(20)

# Tokenizer
tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")

results = []

print("\n📡 Starting teacher response generation for first 20 questions...\n")

for idx, row in df.iterrows():
    question = row["question"]
    context = row["context"]

    prompt = f"""You are a helpful Quebec insurance assistant.
Answer the following question using only the context provided.

Context:
{context}

Question:
{question}
"""

    try:
        start_time = time.time()
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        end_time = time.time()

        answer = response.choices[0].message.content.strip()

        # Token usage
        prompt_tokens = len(tokenizer.encode(prompt))
        response_tokens = len(tokenizer.encode(answer))
        total_tokens = prompt_tokens + response_tokens

        # Log API usage
        log_training_data(question, "DeepSeek-Teacher", "N/A", end_time - start_time)

        # Save result
        results.append({
            "question": question,
            "context": context,
            "teacher_answer": answer
        })

        print(f"\n✅ Question {idx+1}: {question}")
        print(f"🤖 Teacher Answer:\n{answer}\n{'-'*60}")

    except Exception as e:
        print(f"❌ Error on question {idx+1}: {e}")

# Save to CSV
output_df = pd.DataFrame(results)
output_df.to_csv("C:\\CS Tech\\Agents\\Daily\\GPT\\teacher_predictions_first20.csv", index=False)

print("\n🎉 Saved first 20 teacher responses to 'teacher_predictions_first20.csv'")
