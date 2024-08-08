from datasets import load_dataset
import re
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import time
from data_config import *

def clean_and_tokenize(texts, pattern):
    cleaned_texts = []
    for text in texts:
        text = text.replace('\n', ' ')
        text = text.replace('\xa0', ' ')
        matches = pattern.findall(text)
        cleaned_text = ''.join(matches)
        sentences = sent_tokenize(cleaned_text)
        cleaned_texts.extend(sentences)
    return cleaned_texts


unique_chars = "!()*,-.012346789:;?«»ІАБВГДЕЖЗИКЛМНОПРСТУФХЧШЫЭЯабвгдежзийклмнопрстуфхцчшщъыьэюяіҒғҚқҢңҮүҰұһӘәӨө–—•−─"
pattern = re.compile(f"[{re.escape(unique_chars)}\\s]+")

data_path = DATA_PATH
output_file = data_path + 'kazakh_corpus.txt'

max_retries = 5
retry_wait = 5

# Function to load dataset with retries if connection to huggingface fails
def load_dataset_with_retries(dataset_name, split):
    for attempt in range(max_retries):
        try:
            dataset = load_dataset(dataset_name, split=split, streaming=True)
            return dataset
        except Exception as e:
            print(f"Failed to load dataset: {e}. Retrying in {retry_wait} seconds... [{attempt + 1}/{max_retries}]")
            time.sleep(retry_wait)
    raise Exception(f"Failed to load dataset {dataset_name} after {max_retries} attempts")

# Load Qazaq text datasets
dataset = load_dataset_with_retries('Nothingger/Kazakh-Literature-Collection', 'train')
wiki_dataset = load_dataset_with_retries('amandyk/kazakh_wiki_articles', 'train')
multidomain_dataset = load_dataset_with_retries("kz-transformers/multidomain-kazakh-dataset", 'train')


with open(output_file, 'w', encoding='utf-8') as f:
    for dataset_chunk in [dataset, wiki_dataset, multidomain_dataset]:
        for example in dataset_chunk:
            try:
                cleaned_texts = clean_and_tokenize([example['text']], pattern)
                for sentence in cleaned_texts:
                    f.write(sentence.strip() + '\n')
            except Exception as e:
                print(f"Error processing text: {e}")

print(f"Cleaned data saved to {output_file}")
