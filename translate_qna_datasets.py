import pandas as pd
import time
from googletrans import Translator
from tqdm import tqdm
import random
import csv
from data_config import *

    
def translate_text(text, dest='kk'):
    translator = Translator()
    try:
        if pd.isna(text) or text == '':
            return ''
        result = translator.translate(text, dest=dest)
        if result and result.text:
            return result.text
        else:
            raise ValueError("Empty translation result")
    except Exception as e:
        print(f"Error translating: {e}")
        
        return "TRANSLATION_FAILED"

def safe_translate(text):
    try:
        return translate_text(text)
    except Exception as e:
        print(f"Failed to translate: {e}")
        return "TRANSLATION_FAILED"

def translate_and_write_chunk(input_chunk, output_file, writer, error_log):
    for _, row in input_chunk.iterrows():
        try:
            translated_row = {
                'instruction_kz': safe_translate(row['instruction']),
                'context_kz': safe_translate(row['context']),
                'response_kz': safe_translate(row['response']),
                'category': row['category']
            }
            writer.writerow(translated_row)
        except Exception as e:
            error_log.write(f"Error in row {row.name}: {e}\n")
            error_log.write(f"Row content: {row.to_json()}\n\n")
    output_file.flush()

def retranslate_failed_rows(input_file, output_file):
    df = pd.read_csv(input_file)

    for col in ['instruction_kz', 'context_kz', 'response_kz']:
        mask = df[col] == 'TRANSLATION_FAILED'
        failed_indices = df[mask].index

        print(f"Retranslating {len(failed_indices)} failed translations in column {col}")

        for idx in tqdm(failed_indices, total=len(failed_indices)):
            original_text = df.loc[idx, col.replace('_kz', '')]
            translated_text = safe_translate(original_text)
            df.loc[idx, col] = translated_text

    df.to_csv(output_file, index=False)
    print(f"Updated translations saved to {output_file}")

if __name__ == "__main__":
    input_file = INSTRUCTION_DATA_PATH + "dolly.csv"
    output_file = INSTRUCTION_DATA_PATH + "dolly_kz.csv"
    chunk_size = 100 

    with open(output_file, 'w', newline='', encoding='utf-8') as f_output:
        fieldnames = ['instruction_kz', 'context_kz', 'response_kz', 'category']
        writer = csv.DictWriter(f_output, fieldnames=fieldnames)
        writer.writeheader()

        with open('error_log.txt', 'w') as error_log:
            for chunk in tqdm(pd.read_csv(input_file, chunksize=chunk_size), desc="Translating chunks"):
                translate_and_write_chunk(chunk, f_output, writer, error_log)

    print("Translation completed and saved.")
