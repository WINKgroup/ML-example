import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained model and tokenizer
model_name = 'gpt2-xl'  # You can also use 'gpt2-medium', 'gpt2-large', 'gpt2-xl' for larger models
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

# Set the pad_token_id to the eos_token_id to avoid warnings
tokenizer.pad_token = tokenizer.eos_token

# Move model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to generate text
def generate_text(model, tokenizer, start_text, max_length, device):
    inputs = tokenizer.encode(start_text, return_tensors='pt').to(device)
    attention_mask = torch.ones(inputs.shape, device=device)
    outputs = model.generate(
        inputs,
        attention_mask=attention_mask,
        max_length=max_length,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,  # Enables sampling to generate more diverse text
        top_k=50,        # Keeps only the top 50 tokens with highest probability
        top_p=0.95       # Keeps only the tokens with cumulative probability >= 0.95
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Generate text
start_text = "today is a beautiful day"
max_generated_tokens = 500  # specify the number of tokens to generate
generated_text = generate_text(model, tokenizer, start_text, max_generated_tokens, device)
print("Generated text:\n", generated_text)
