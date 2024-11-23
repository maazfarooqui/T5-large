import torch
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split

# Memory-efficient T5 Large Fine-Tuning

# Disable tokenizers parallelism to reduce memory overhead
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Function to print CUDA memory info
def print_cuda_memory():
    if torch.cuda.is_available():
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

# Load and prepare data
df = pd.read_csv("train.csv")

# Basic data cleaning
df['Context'] = df['Context'].fillna('').astype(str)
df['Response'] = df['Response'].fillna('').astype(str)

# Split data
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

# Prepare tokenizer and model
tokenizer = T5Tokenizer.from_pretrained('t5-large')
model = T5ForConditionalGeneration.from_pretrained('t5-large')

# Tokenization function
def tokenize_data(examples):
    inputs = tokenizer(
        ["generate response: " + context for context in examples['Context']], 
        max_length=256, 
        truncation=True, 
        padding='max_length'
    )
    
    targets = tokenizer(
        examples['Response'], 
        max_length=256, 
        truncation=True, 
        padding='max_length'
    )
    
    return {
        'input_ids': inputs.input_ids,
        'attention_mask': inputs.attention_mask,
        'labels': targets.input_ids
    }

# Convert to Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# Tokenize datasets
tokenized_train = train_dataset.map(
    tokenize_data, 
    batched=True, 
    remove_columns=train_dataset.column_names
)

tokenized_val = val_dataset.map(
    tokenize_data, 
    batched=True, 
    remove_columns=val_dataset.column_names
)

# Disable caching to save memory
model.config.use_cache = False

# Extremely memory-efficient training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir='./results',
    num_train_epochs=5,  # Increased epochs to compensate for small batch size
    per_device_train_batch_size=1,  # Smallest possible batch size
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,  # Simulate larger batch size
    learning_rate=1e-4,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy='steps',
    eval_steps=500,
    save_steps=1000,
    load_best_model_at_end=True,
    predict_with_generate=True,
    fp16=False,  # Explicitly disable mixed precision
    max_grad_norm=0.1
)

# Initialize trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val
)

# Print initial memory status
print("Initial CUDA Memory:")
print_cuda_memory()

# Train the model
try:
    # Clear cache before training
    torch.cuda.empty_cache()
    
    # Start training
    trainer.train()
    
    # Save model
    trainer.save_model('./results/final_model')
    
except Exception as e:
    print(f"Training error: {e}")
    # Attempt to save model even if training fails
    try:
        trainer.save_model('./results/partial_model')
    except:
        print("Could not save model")

# Print final memory status
print("\nFinal CUDA Memory:")
print_cuda_memory()

# Simple inference function
def generate_response(context):
    inputs = tokenizer(
        f"generate response: {context}", 
        return_tensors="pt", 
        max_length=256, 
        truncation=True
    )
    
    outputs = model.generate(
        inputs.input_ids, 
        max_length=100, 
        num_return_sequences=1, 
        no_repeat_ngram_size=2
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test generation with a few examples
test_contexts = [
    "I'm feeling stressed about my upcoming exams.",
    "I had a fight with my best friend.",
    "I'm considering changing my career."
]

print("\nModel Response Testing:")
for context in test_contexts:
    response = generate_response(context)
    print(f"\nContext: {context}")
    print(f"Generated Response: {response}")