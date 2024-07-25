
from datasets import load_dataset
import re
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

dataset = load_dataset('Nothingger/Kazakh-Literature-Collection')

wiki_dataset = load_dataset('amandyk/kazakh_wiki_articles')

multidomain_dataset = load_dataset("kz-transformers/multidomain-kazakh-dataset")

data = dataset['train']['text']
data = data + wiki_dataset['train']['text']
data = data + multidomain_dataset['train']['text']

unique_chars = "!()*,-.012346789:;?«»ІАБВГДЕЖЗИКЛМНОПРСТУФХЧШЫЭЯабвгдежзийклмнопрстуфхцчшщъыьэюяіҒғҚқҢңҮүҰұһӘәӨө–—•−─"

pattern = re.compile(f"[{re.escape(unique_chars)}\\s]+")

cleaned_texts = []

for text in data:
    text = text.replace('\n', ' ')
    text = text.replace('\xa0', ' ')
    matches = pattern.findall(text)
    cleaned_text = ''.join(matches)
    sentences = sent_tokenize(cleaned_text)
    cleaned_texts.extend(sentences)

data_path = "data/"
output_file = data_path+'kazakh_corpus.txt'

with open(output_file, 'w', encoding='utf-8') as f:
    for sentence in cleaned_texts:
        f.write(sentence.strip() + '\n')

print(f"Cleaned data saved to {output_file}")

