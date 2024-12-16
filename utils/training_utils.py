from time import time
from typing import List

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from tqdm import tqdm
from transformers import MarianTokenizer

from config import prefixes
from utils.data_utils import get_data, get_split


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels


def compute_metrics(eval_preds, evaluator, tokenizer):
    print("Computing metrics...")

    start_time = time()
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    preds = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]

    results = {}
    result = evaluator.get_metrics(predictions=decoded_preds, references=decoded_labels)
    # result["gen_len"] = np.mean(preds)

    results["regular"] = result
    results["grounded"] = evaluator.get_metrics(
        decoded_preds, decoded_labels, ground=True
    )
    results["lead8"] = evaluator.get_metrics(decoded_preds, decoded_labels, lead_n=8)
    results["lead16"] = evaluator.get_metrics(decoded_preds, decoded_labels, lead_n=16)

    # round all values to 4
    for k, v in results.items():
        results[k] = {k: round(v, 4) for k, v in results[k].items()}

    end_time = time()
    print(f"Finished computing metrics in {end_time - start_time} seconds")
    return result


def tokenize_all_compression_levels(
    data_path,
    tokenizer,
    source_lang,
    target_lang,
    prefix,
    max_size=400_000,
    compression_folder="compressed",
    max_input_length=256,
    max_target_length=256,
    use_range=False,
):
    # comp rate from 0.5 to 1.0
    splits = []
    for comp in tqdm(range(5, 11), desc="Compression levels"):
        comp = comp / 10
        tmp = get_split(
            source_lang,
            target_lang,
            path=data_path,
            comp=comp,
            split="train",
            compression_folder=compression_folder,
            as_dataset=False,  # keep it as a dataframe
            use_range=use_range,
        )
        # add prefix to the english ("en") column
        comp_prefix = f"@{comp} "
        if prefix:
            comp_prefix = f"@{comp} {prefix} "
        tmp["en"] = [f"{comp_prefix}{ex}" for ex in tmp["en"]]
        tmp = tmp.reset_index(drop=True)
        if len(tmp) > max_size:
            print(f"Downsampling from {len(tmp)} to {max_size} samples")
            tmp = tmp.sample(max_size, random_state=42)
        splits.append(tmp)

    df = pd.concat(splits)
    print(f"Final df size: {len(df)}")
    # if the df is less than max_size, downsample it:
    # random sample 100k to speed up training
    # df = df.sample(100000, random_state=42)
    dataset = Dataset.from_pandas(df)
    del splits

    def mapping(examples):
        inputs = examples[source_lang]
        targets = examples[target_lang]
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
        labels = tokenizer(
            text_target=targets, max_length=max_target_length, truncation=True
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return dataset.map(mapping, batched=True)


def tokenize_data(
    data_path,
    compression_rate,
    tokenizer,
    source_lang,
    target_lang,
    prefix,
    max_input_length=256,
    max_target_length=256,
    split=None,
    compression_folder="compressed",
    sizes: List[int] = None,
    max_size: int = 250_000,
    use_range:bool=False,
):
    dataset = get_data(
        path=data_path,
        lang1=source_lang,
        lang2=target_lang,
        comp=compression_rate,
        compression_folder=compression_folder,
        split=split,
        use_range=use_range,
    )

    def mapping(examples):
        if prefix:
            inputs = [f"{prefix} {ex}" for ex in examples[source_lang]]
        else:
            inputs = examples[source_lang]
        targets = examples[target_lang]
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
        labels = tokenizer(
            text_target=targets, max_length=max_target_length, truncation=True
        )
        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    return dataset.map(mapping, batched=True)
