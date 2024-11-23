import pandas as pd
import torch
from datasets import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("train.csv")

# Handle missing or incorrect data
df["Context"] = df["Context"].fillna("").astype(str)
df["Response"] = df["Response"].fillna("").astype(str)

# Split the data into train and validation sets
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

# Convert DataFrames to Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# Load T5 tokenizer and model
model_name = "t5-base"  # You can change this to "t5-base" or "t5-large" if you have more computational resources
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Tokenize function
def tokenize_function(examples):
    inputs = tokenizer("generate response: " + examples["Context"], max_length=512, truncation=True, padding="max_length", return_tensors="pt")
    with tokenizer.as_target_tokenizer():
        outputs = tokenizer(examples["Response"], max_length=512, truncation=True, padding="max_length", return_tensors="pt")

    return {
        "input_ids": inputs.input_ids.flatten(),
        "attention_mask": inputs.attention_mask.flatten(),
        "labels": outputs.input_ids.flatten(),
    }

# Apply tokenization
tokenized_train_dataset = train_dataset.map(tokenize_function, remove_columns=train_dataset.column_names)
tokenized_val_dataset = val_dataset.map(tokenize_function, remove_columns=val_dataset.column_names)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results_t5",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=1000,
    load_best_model_at_end=True,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained("./results_t5")
tokenizer.save_pretrained("./results_t5")

print("Model training completed and saved.")

# Load the trained model and tokenizer for testing
loaded_model = T5ForConditionalGeneration.from_pretrained("./results_t5")
loaded_tokenizer = T5Tokenizer.from_pretrained("./results_t5")

# Function to generate response
def generate_response(context):
    input_ids = loaded_tokenizer("generate response: " + context, return_tensors="pt", max_length=512, truncation=True).input_ids
    outputs = loaded_model.generate(input_ids, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2)
    return loaded_tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test the model
test_contexts = [
    "I'm feeling really stressed about my upcoming exams.",
    "I had a fight with my best friend and I don't know what to do.",
    "I'm considering changing my career but I'm unsure about the next steps."
]

print("\nTesting the model with sample contexts:")
for context in test_contexts:
    response = generate_response(context)
    print(f"\nContext: {context}")
    print(f"Generated Response: {response}")