import pandas as pd
from project_config.data_config import *
import csv
import os

## Downloading instruction datasets that will be used for question answering fine-tuning
def download_instruction_datasets():
    splits = {'kaz_Cyrl': 'data/kaz_Cyrl.jsonl'}
    df = pd.read_json("hf://datasets/facebook/belebele/" + splits["kaz_Cyrl"],lines=True)
    df.to_csv(INSTRUCTION_DATA_PATH+"belebele.csv", index=False)


    kaz_instruction_df = pd.read_json("hf://datasets/AmanMussa/kazakh-instruction-v2/kaz_instructions.json")
    kaz_instruction_df.to_csv(INSTRUCTION_DATA_PATH+"kaz_instruction.csv", index=False)

    alpaca_df  = pd.read_parquet("hf://datasets/tatsu-lab/alpaca/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet")
    alpaca_df.to_csv(INSTRUCTION_DATA_PATH+"alpaca.csv", index=False)

    dolly_df = pd.read_json("hf://datasets/databricks/databricks-dolly-15k/databricks-dolly-15k.jsonl", lines=True)
    dolly_df.to_csv(INSTRUCTION_DATA_PATH+"dolly.csv", index=False)


def chunk_extract_text_from_csv(csv_file_path, text_column_names, output_text_file, chunk_size=1000):
    with open(csv_file_path, 'r', encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file)
        with open(output_text_file, 'a', encoding='utf-8') as text_file:
            chunk = []
            for i, row in enumerate(reader):
                for text_column_name in text_column_names:
                    chunk.append(row[text_column_name] + '\n')
                    if (i + 1) % chunk_size == 0:
                        text_file.writelines(chunk)
                        chunk = []
            if chunk:
                text_file.writelines(chunk)

if __name__ == '__main__':
    csv_file_path = os.path.join(INSTRUCTION_DATA_PATH, 'dolly_kz.csv')
    output_text_file = os.path.join(DATA_PATH, 'instruction_text.txt')
    text_column_names = ['instruction_kz', 'context_kz', 'response_kz']

    chunk_extract_text_from_csv(csv_file_path, text_column_names, output_text_file, chunk_size=1000)
    
    csv_file_path = os.path.join(INSTRUCTION_DATA_PATH, 'kaz_instruction.csv')
    text_column_names = ['instruction', 'input', 'output']
    chunk_extract_text_from_csv(csv_file_path, text_column_names, output_text_file, chunk_size=1000)    

    csv_file_path = os.path.join(INSTRUCTION_DATA_PATH, 'belebele.csv')
    text_column_names = ['flores_passage', 'question', 'mc_answer1', 'mc_answer2', 'mc_answer3', 'mc_answer4']
    chunk_extract_text_from_csv(csv_file_path, text_column_names, output_text_file, chunk_size=1000)