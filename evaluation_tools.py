import json
import os
from typing import Callable, Dict, List, Literal

import numpy as np
import pandas as pd
from config import COMPRESSION_RATIOS, OPUS_MT_MODELS, output_folders, prefixes
from evaluation_datagetters import get_datagetters
from IPython.display import HTML, display
from torch.cuda import empty_cache
from utils.model_utils import load_baseline, load_model, translate_all

from evaluation.evaluator import Evaluator


def evaluate(
    model_type: Literal["singlemodel", "tokenmodel", "multisize"],
    langs: List[str],
    dataset: str,
    subsampling_n: int = 5,
    samples_per_run: int = 1000,
    batch_size: int = 8,
    device: str = "cuda",
    custom_folder_prefix: str = "",  # output results folder name prefix
    custom_suffix: str = "",  # a suffix for the trained model in the output folders
    output_folder="results",
):
    datagetters = get_datagetters()
    print("Recieved params:", locals())
    # repeated random sub-sampling: 1000 samples, done N times, calc the average.
    random_seed_samples = np.arange(subsampling_n).tolist()
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(f"{output_folder}/{custom_folder_prefix}{dataset}", exist_ok=True)

    for random_seed in random_seed_samples:
        print(f"Random seed: {random_seed}")
        filepath = f"{output_folder}/{custom_folder_prefix}{dataset}/{dataset}_{model_type}{custom_suffix}.{random_seed}/"
        # if os.path.exists(filepath):
        #     print(f"Directory exists: {filepath}")
        #     continue
        metrics, translations = get_metrics(
            languages=langs,
            datagetter=datagetters[dataset],
            n_samples=samples_per_run,
            batch_size=batch_size,
            device=device,
            folder=output_folders[model_type],
            # if True, the model is trained on all compressions levels
            joint_compression="single" not in model_type,
            compression_level=0.5,  # compression level should be 0.5 to evaluate shorter sents
            random_state=random_seed,  # this can be used to control bootstrap resampling
            filepath=filepath,
            custom_suffix=custom_suffix,
        )

        # save metrics and translations dicts to json
        filename = f"{output_folder}/{dataset}/{dataset}_{model_type}.{random_seed}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=4)


def filter_metrics(metrics):
    ignored = ["r2", "rl", "chrf++", "bleu"]
    return {
        k: v for k, v in metrics.items() if not any(k.startswith(i) for i in ignored)
    }


def post_cleanup(df):
    cleanup = [c for c in df.columns if "len" in c and c != "len_ratio"]
    return df.drop(columns=cleanup)


def print_metrics(
    _lang, _metrics, std=False, transpose=False, return_score=False, just_return=False
):
    tmpdf = pd.DataFrame(_metrics).T
    tmpdf = tmpdf[[c for c in tmpdf.columns if c != "len_ratio"] + ["len_ratio"]]
    tmpdf = post_cleanup(tmpdf)
    if std:
        # convert NUM (STD VAL) to NUM for "length" column:
        tmpdf["len_ratio"] = tmpdf["len_ratio"].apply(lambda x: x.split()[0])
        # display(HTML(f"<h1>{_lang}</h1>"))
        # display(tmpdf)
        # print()
        # print(tmpdf.to_latex(float_format="%.2f"))
        return tmpdf
    if just_return:
        return tmpdf

    # __metrics = tmpdf.select_dtypes(include=["float64", "int64"]).columns
    # __df = tmpdf.copy()
    # for __metric in __metrics:
    #     if __metric == "len_ratio":
    #         continue
    #     __df[__metric] = __df[__metric] / __df[__metric].max()

    normed_df = tmpdf / tmpdf.max(axis=0)
    og_normed_sum = normed_df.sum(axis=1)
    # normed_sum = (normed_sum - normed_sum.min()) / (normed_sum.max() - normed_sum.min())
    len_ratio = tmpdf["len_ratio"].copy()
    # display(HTML(f"<h1>{_lang} - metrics</h1>"))
    # display(tmpdf)

    before_wgt = tmpdf.copy()
    for row in tmpdf.index:
        tmpdf.loc[row] /= tmpdf.loc[row]["len_ratio"]
        # tmpdf.loc[row] = np.round(tmpdf.loc[row], 6)
    tmpdf["len_ratio"] = len_ratio
    # normalize all metrics
    normed_df = tmpdf / tmpdf.max(axis=0)
    normed_sum = normed_df.sum(axis=1)  # sum of all metrics on the same row
    tmpdf = before_wgt
    # tmpdf["score"] = __df.sum(axis=1).multiply(10).astype(int)
    tmpdf["score"] = og_normed_sum.multiply(10).astype(int)
    tmpdf["comp_score"] = normed_sum.multiply(10).astype(int)
    # normed_sum = (normed_sum - normed_sum.min()) / (normed_sum.max() - normed_sum.min())
    # tmpdf["weighted_score"] = normed_sum.round(2)
    if return_score:
        return normed_sum.multiply(10).astype(int)

    display(HTML(f"<h1>{_lang} - weighted metrics (length)</h1>"))
    if transpose:
        display(tmpdf.T)
    else:
        display(tmpdf)

    print()
    # print latex and round to 2 decimals
    print(tmpdf.to_latex(float_format="%.2f"))
    return tmpdf


def make_flexi_prefix(lang, compression):
    # assert 0.5 <= compression <= 1.0, "Compression must be between 0.5 and 1.0"
    prefix = prefixes.get(lang, "")
    comp_prefix = f"@{compression}"
    return f"{comp_prefix} {prefix}"


def get_metrics(
    languages: List[str],
    datagetter: Callable,
    n_samples: int,
    batch_size: int,
    device: str,
    folder: str,
    joint_compression: bool,
    compression_level: int = 0.5,
    filepath: str = None,
    random_state: int = 42,
    save_all: bool = False,
    custom_suffix: str = "",
):
    metrics_by_lang = {}
    all_translations = {}

    for lang in languages:
        print(f"Processing {lang}")
        metric_path = os.path.join(filepath, f"{lang}.metrics.json")
        trans_path = os.path.join(filepath, f"{lang}.translations.json")
        print(f"Saving to {metric_path} and {trans_path}")
        if os.path.exists(metric_path) and os.path.exists(trans_path):
            print(f"Skipping {lang}, already exists")
            continue

        _metrics, _translations = get_language_metrics(
            source_lang="en",
            target_lang=lang,
            datagetter=datagetter,
            n_samples=n_samples,
            batch_size=batch_size,
            device=device,
            folder=folder,
            joint_compression=joint_compression,
            compression_level=compression_level,
            random_state=random_state,
            custom_suffix=custom_suffix,
        )
        if save_all:
            metrics_by_lang[lang] = _metrics
            all_translations[lang] = _translations

        if filepath:
            os.makedirs(filepath, exist_ok=True)

            with open(metric_path, "w", encoding="utf-8") as f:
                json.dump(_metrics, f, ensure_ascii=False, indent=4)
            with open(trans_path, "w", encoding="utf-8") as f:
                json.dump(_translations, f, ensure_ascii=False, indent=4)

    return metrics_by_lang, all_translations


def get_language_metrics(
    source_lang: str,
    target_lang: str,
    datagetter: Callable,
    n_samples: int,
    batch_size: int,
    device: str,
    folder: str,
    joint_compression: bool,
    compression_level: int,
    random_state: int,
    custom_suffix: str,  # a suffix for the model naming (in an output folder)
    decimals: int = 6,
):
    lang1 = source_lang
    lang = target_lang
    print(f"Getting metrics for {lang}")

    evaluator = Evaluator()

    model_id = OPUS_MT_MODELS[lang]
    baseline_model, baseline_tokenizer = load_baseline(model_id, device=device)

    test_df = datagetter(source_lang, lang, compression_level=compression_level)
    ###### Normalize
    src_len = test_df[lang1].str.len()
    tgt_len = test_df[lang].str.len()
    mean_len = (tgt_len / src_len).mean()
    print(f"Mean length ratio: {mean_len}")
    ###### Compress < original length times the mean
    print(f"Size of test set: {len(test_df)}")
    test_df = test_df[test_df[lang].str.len() < mean_len * test_df[lang1].str.len()]
    print(f"Size of test set after compression: {len(test_df)}")
    sample_size = min(n_samples, len(test_df))
    test_df = test_df.sample(n=sample_size, random_state=random_state)

    tar_texts = test_df[lang].tolist()
    src_texts = test_df[lang1].tolist()
    prefix = prefixes.get(lang, "")
    src_texts = [f"{prefix} {s}".strip() for s in src_texts]

    baseline_translations = translate_all(
        src_texts,
        baseline_model,
        baseline_tokenizer,
        device=device,
        batch_size=batch_size,
    )
    del baseline_model
    empty_cache()

    baseline_metrics = evaluator.get_metrics(
        baseline_translations, tar_texts, lang=lang, decimals=decimals
    )
    metrics = {"baseline": baseline_metrics}
    translations = {
        "source": src_texts,
        "target": tar_texts,
        "baseline": baseline_translations,
    }
    if joint_compression:
        print("--- Using joint compression")
        # use one model
        path = os.path.join(folder, f"{lang1}-{lang}{custom_suffix}")
        model, tokenizer = load_model(path)
        for compression_ratio in COMPRESSION_RATIOS:
            prefix = make_flexi_prefix(lang, compression_ratio)
            translated = translate_all(
                src_texts,
                model,
                tokenizer,
                device=device,
                batch_size=batch_size,
                prefix=prefix,
            )
            translations[compression_ratio] = translated
            comp_metrics = evaluator.get_metrics(translated, tar_texts, lang, decimals=decimals)
            metrics[compression_ratio] = comp_metrics
    else:
        print(f"--- Using separate models for each compression ratio")
        for compression_ratio in COMPRESSION_RATIOS:
            print(f"--- Loading model trained on compression rate: {compression_ratio}")
            path = os.path.join(folder, f"{lang1}-{lang}", str(compression_ratio))
            model, tokenizer = load_model(path)
            if not model or not tokenizer:
                print(f"Skipping compression rate {compression_ratio}. No model found.")
                continue
            prefix = prefixes.get(lang, "")
            translated = translate_all(
                src_texts,
                model,
                tokenizer,
                device=device,
                batch_size=batch_size,
                prefix=prefix,
            )
            translations[compression_ratio] = translated

            comp_metrics = evaluator.get_metrics(translated, tar_texts, lang, decimals=decimals)
            metrics[compression_ratio] = comp_metrics

    return metrics, translations
