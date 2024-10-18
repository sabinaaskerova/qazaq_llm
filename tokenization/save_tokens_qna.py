import torch
from project_config.data_config import *
import pandas as pd
import sentencepiece as spm
from special_tokenizer import SpecialTokenizerWrapper

# Initialize
special_tokens = [
    "<s>", "</s>",
    "<instruction>", "</instruction>",
    "<context>", "</context>",
    "<response>", "</response>"
]
tokenizer = SpecialTokenizerWrapper(
    spm_model_path=TOKENIZER_PATH + "m.model",
    special_tokens=special_tokens
)


def prepare_and_tokenize_data(csv_path):
    df = pd.read_csv(csv_path)
    tokenized_data = []
    for _, row in df.iterrows():
        instruction = f"<s><instruction> {row['instruction_kz']} </instruction>"
        context = f"<context> {row['context_kz']} </context>" if 'context' in row and pd.notna(row['context']) else ""
        response = f"<response> {row['response_kz']} </response></s>"
        full_text = f"{instruction}{context}{response}"
        tokenized = tokenizer.encode(full_text)
        tokenized_data.extend(torch.tensor(tokenized))
    return tokenized_data


csv_path = INSTRUCTION_DATA_PATH + "dolly_kz.csv"
tokenized_data = prepare_and_tokenize_data(csv_path) # list of tokenized tensors

unique_tokens = set(tokenized_data)
num_unique_tokens = len(unique_tokens)
print(num_unique_tokens)

tensor_text = torch.tensor(tokenized_data).unsqueeze(0)
print("tensor_text.shape", tensor_text.shape)

torch.save(tensor_text, TOKENIZER_PATH+"finetuning_tokenized_data.pt")
print(f"Tokenized data saved to {TOKENIZER_PATH}finetuning_tokenized_data.pt")
torch.save(num_unique_tokens, TOKENIZER_PATH+"finetuning_num_unique_tokens.pt")

