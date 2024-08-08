import pandas as pd
from data_config import *


splits = {'kaz_Cyrl': 'data/kaz_Cyrl.jsonl'}
df = pd.read_json("hf://datasets/facebook/belebele/" + splits["kaz_Cyrl"],lines=True)
df.to_csv(INSTRUCTION_DATA_PATH+"belebele.csv", index=False)


kaz_instruction_df = pd.read_json("hf://datasets/AmanMussa/kazakh-instruction-v2/kaz_instructions.json")
kaz_instruction_df.to_csv(INSTRUCTION_DATA_PATH+"kaz_instruction.csv", index=False)


splits = {'train': 'data/train-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet'}
alpaca_kazakh_df = pd.read_parquet("hf://datasets/saillab/alpaca_kazakh_taco/" + splits["train"])
alpaca_kazakh_df.to_csv(INSTRUCTION_DATA_PATH+"alpaca_kazakh.csv", index=False)


dolly_df = pd.read_json("hf://datasets/databricks/databricks-dolly-15k/databricks-dolly-15k.jsonl", lines=True)
dolly_df.to_csv(INSTRUCTION_DATA_PATH+"dolly.csv", index=False)

