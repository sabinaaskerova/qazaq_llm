import requests
import sqlite3
import pandas as pd
import os
import requests
import zipfile
from bs4 import BeautifulSoup
from data_config import *

def read_apkg_db(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM notes")
    rows = cursor.fetchall()
    conn.close()
    
    return rows

def save_to_tsv(rows, output_file):
    df = pd.DataFrame(rows, columns=['id', 'guid', 'mid', 'mod', 'usn', 'tags', 'flds', 'sfld', 'csum', 'flags', 'data'])
    
    df[['Front', 'Back']] = df['flds'].str.split('\x1f', expand=True)
    
    df[['Front', 'Back']].to_csv(output_file, sep='\t', index=False, header=False)
    print(f"Saved data to {output_file}")

def clean_html(text):
    soup = BeautifulSoup(text, 'html.parser')
    cleaned_text = soup.get_text(separator=" ", strip=True)  # Get text without tags, with spaces
    return cleaned_text

def save_to_txt(rows, output_file):
    df = pd.DataFrame(rows, columns=['id', 'guid', 'mid', 'mod', 'usn', 'tags', 'flds', 'sfld', 'csum', 'flags', 'data'])
    
    df[['Front', 'Back']] = df['flds'].str.split('\x1f', expand=True)
    
    # Clean HTML tags from 'Front' and 'Back' columns
    df['Front'] = df['Front'].apply(clean_html)
    df['Back'] = df['Back'].apply(clean_html)
    
    df[['Front', 'Back']].to_csv(output_file, sep='\t', index=False, header=False)  # Save with \t separator and no header
    
    # Rename the file extension to .txt
    data_path = DATA_PATH
    txt_file = data_path+ os.path.splitext(output_file)[0] + ".txt"
    os.rename(output_file, txt_file)
    
    print(f"Saved data to {txt_file}")

url = "https://ankiweb.net/svc/shared/download-deck/278634922?t=eyJvcCI6InNkZCIsImlhdCI6MTcyMDc0NDU3NSwianYiOjF9.yATghWSsF4m-FrlQapm8mZnFLgm4uSidgZacRS5boXE"
try:
    response = requests.get(url)
    if response.status_code == 200:
        with open('Kaz-Rus.apkg', 'wb') as f:
            f.write(response.content)
        print("File downloaded successfully.")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")

except requests.exceptions.RequestException as e:
    print(f"Error: {e}")


apkg_file_path = "Kaz-Rus.apkg"
new_filename = os.path.splitext(apkg_file_path)[0] + ".zip"
if not os.path.exists(new_filename):
    os.rename(apkg_file_path, new_filename)

print(f"Renamed {apkg_file_path} to {new_filename}")

zip_file = new_filename
data_path = DATA_PATH
output_folder = data_path + "anki_kaz_rus"

with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall(output_folder)

print(f"Unzipped {zip_file} into {output_folder}")


rows = read_apkg_db(output_folder+'/collection.anki21')

output_text_file = "anki_data.txt"
save_to_txt(rows, output_text_file)

