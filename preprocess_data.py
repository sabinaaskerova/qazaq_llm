from datasets import load_dataset
import re
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

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


dataset = load_dataset('Nothingger/Kazakh-Literature-Collection', split='train', streaming=True)
wiki_dataset = load_dataset('amandyk/kazakh_wiki_articles', split='train', streaming=True)
multidomain_dataset = load_dataset("kz-transformers/multidomain-kazakh-dataset", split='train', streaming=True)


data_path = "data/"
output_file = data_path + 'kazakh_corpus.txt'

# Save cleaned data to file
with open(output_file, 'w', encoding='utf-8') as f:
    for dataset_chunk in [dataset, wiki_dataset, multidomain_dataset]:
        for example in dataset_chunk:
            cleaned_texts = clean_and_tokenize([example['text']], pattern)
            for sentence in cleaned_texts:
                f.write(sentence.strip() + '\n')

print(f"Cleaned data saved to {output_file}")
