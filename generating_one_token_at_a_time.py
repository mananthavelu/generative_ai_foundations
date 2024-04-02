# Predicting the next word in a sentence includes tokenizing the input and 
# passing them through the model.

# Import the necessary libraries
from transformers import AutoModelForCausalLM, AutoTokenizer#Autoclasses for model and tokenizer
import pandas as pd#For data manipulation
import torch# for tensor operations


# Authentication
access_token = "actual_api_token_here"
# Load the tokenizer and the model
model_name ='gpt2'#Can be any other available model
tokenizer = AutoTokenizer.from_pretrained(model_name, token = access_token)# Initialize the tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name, token = access_token)# Initialize the model

# Create a partial sentence and tokenize it
text = "Hello how are "
inputs = tokenizer(text, return_tensors="pt")#returns a dictionary of tensors

# Print the tokens
print(inputs)
print(inputs["input_ids"][0])

# Show the tokens vs ids
def show_tokenization(inputs):
    return pd.DataFrame(
        [(id, tokenizer.decode(id)) for id in inputs["input_ids"][0]],
        columns=["id", "token"],
    )
show_tokenization(inputs)

with torch.no_grad():# No Training, only inference
    # Generate the next token
    logits = model(**inputs).logits[:,-1,:]
    probabilities = torch.nn.functional.softmax(logits[0],dim=-1)

# View the calculated probabilities
print(probabilities)

# Show the 'next token' choices with probabilities
def show_next_token_probabilites(probabilities, limit=5):
    return pd.DataFrame(
        [
            (id, tokenizer.decode(id), p.item())
            for id, p in enumerate(probabilities)
            if p.item()
        ],
        columns=["id", "token", "p"],
    ).sort_values("p", ascending=False)[:limit]


result = show_next_token_probabilites(probabilities)
print(result)