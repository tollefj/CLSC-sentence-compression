import sys

sys.path.append("..")

import csv
import os

from evaluator import Evaluator

from config import OPUS_MT_MODELS, prefixes
from utils.data_utils import get_bbt_data
from utils.model_utils import load_baseline, load_model, translate_all

lang = "no"
src_texts, tar_texts, prefixed = get_bbt_data(prefix=prefixes.get(lang, ""))
baseline_model, baseline_tokenizer = load_baseline(OPUS_MT_MODELS[lang])
baseline_translations = translate_all(prefixed, baseline_model, baseline_tokenizer)
# src_texts, tar_texts, prefixed = get_bbt_data(prefix=">>nob<<")

evaluator = Evaluator()
lang = "no"
MODEL_DIR = "OUTPUT"

for compression_ratio in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    print(f"Compression ratio: {compression_ratio}")
    filename = f"BBT/comp_{compression_ratio}_results.csv"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    print(filename)
    with open(filename, "w", newline="") as f:
        model, tokenizer = load_model(
            path=f"../{MODEL_DIR}/en-{lang}/{compression_ratio}"
        )
        translated = translate_all(prefixed, model, tokenizer, batch_size=16)

        csvdata = evaluator.get_csv_data(
            predictions=translated,
            references=tar_texts,
            source=src_texts,
            baselines=baseline_translations,
            lang=lang,
        )

        writer = csv.writer(f)
        for data in csvdata:
            writer.writerow(data)
