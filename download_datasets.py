import pandas as pd
from data_config import *

## Downloading instruction datasets that will be used for question answering fine-tuning

splits = {'kaz_Cyrl': 'data/kaz_Cyrl.jsonl'}
df = pd.read_json("hf://datasets/facebook/belebele/" + splits["kaz_Cyrl"],lines=True)
df.to_csv(INSTRUCTION_DATA_PATH+"belebele.csv", index=False)


kaz_instruction_df = pd.read_json("hf://datasets/AmanMussa/kazakh-instruction-v2/kaz_instructions.json")
kaz_instruction_df.to_csv(INSTRUCTION_DATA_PATH+"kaz_instruction.csv", index=False)

alpaca_df  = pd.read_parquet("hf://datasets/tatsu-lab/alpaca/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet")
alpaca_df.to_csv(INSTRUCTION_DATA_PATH+"alpaca.csv", index=False)


dolly_df = pd.read_json("hf://datasets/databricks/databricks-dolly-15k/databricks-dolly-15k.jsonl", lines=True)
dolly_df.to_csv(INSTRUCTION_DATA_PATH+"dolly.csv", index=False)

