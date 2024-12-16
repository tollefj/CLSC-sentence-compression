import os
import re
from typing import List

import pandas as pd
from datasets import Dataset, DatasetDict
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm


def get_split(
    lang1,
    lang2,
    path="../data",
    comp="random",
    split=None,
    compression_folder="compressed",
    as_dataset=True,
    data_source="opensubtitles",
    use_range=False,
):
    path = os.path.join(path, data_source, compression_folder)
    if split:
        if use_range:
            csv_path = os.path.join(path, f"{lang1}-{lang2}-{comp}-range.{split}.csv")
        else:
            csv_path = os.path.join(path, f"{lang1}-{lang2}-{comp}.{split}.csv")
    else:
        if use_range:
            csv_path = os.path.join(path, f"{lang1}-{lang2}-{comp}-range.csv")
        else:
            csv_path = os.path.join(path, f"{lang1}-{lang2}-{comp}.csv")

    df = pd.read_csv(csv_path).dropna()
    if as_dataset:
        return Dataset.from_pandas(df)
    return df


def get_data(
    path,
    lang1,
    lang2,
    comp,
    compression_folder="compressed",
    split=None,
    use_range=False,
):
    # path is the data path (e.g. "../data")
    if split:
        return get_split(
            lang1,
            lang2,
            path,
            comp,
            split,
            compression_folder=compression_folder,
            use_range=use_range,
        )
    return DatasetDict(
        {
            "train": get_split(
                lang1,
                lang2,
                path,
                comp,
                split="train",
                compression_folder=compression_folder,
                use_range=use_range,
            ),
            "test": get_split(
                lang1,
                lang2,
                path,
                comp,
                split="test",
                compression_folder=compression_folder,
                use_range=use_range
            ),
            "validation": get_split(
                lang1,
                lang2,
                path,
                comp,
                split="val",
                compression_folder=compression_folder,
                use_range=use_range
            ),
        }
    )


def get_bbt_data(path="../data/big-bang-theory-all.csv", prefix=">>nob<<"):
    test_df = pd.read_csv(path, sep="|")
    src_texts = test_df["en"].tolist()
    tar_texts = test_df["no"].tolist()
    prefixed = [f"{prefix} {s}" for s in src_texts]
    return src_texts, tar_texts, prefixed


def clean(text):
    # initial dialogue hyphens
    text = re.sub(r"^-+", "", text)
    # whitespaces
    text = re.sub(r"\s+", " ", text)
    # remove any repeated hyphens
    text = re.sub(r"(-\s*){2,}", "", text)
    # text = re.sub(r'-+', '', text)
    return text.strip()


def split_df(df, en_key="en", no_key="no", to_drop="translation"):
    df["en"] = df[to_drop].apply(lambda x: x[en_key])
    df["no"] = df[to_drop].apply(lambda x: x[no_key])
    df.drop(to_drop, axis=1, inplace=True)

    return df


def shorten(df, n=200):
    df = df[df["en"].str.len() <= n]
    df = df[df["no"].str.len() <= n]
    return df


def clean_df(df, max_seq_len=256, prompt="", columns=["en", "no"]):
    max_seq_len -= len(prompt)

    df = df[df["en"].str.split().str.len() <= max_seq_len]
    df = df[df["no"].str.split().str.len() <= max_seq_len]
    df.en = df.en.str.replace(r"\s+", " ", regex=True).str.strip()
    df.no = df.no.str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[df.en.str.len() > 0]
    df = df[df.no.str.len() > 0]

    # add prompts
    df.en = prompt + df.en

    # rename columns:
    if columns:
        df.columns = columns

    return df


def load_dfs(dataset, clean=True) -> List[pd.DataFrame]:
    dfs = {
        "train": dataset["train"].to_pandas(),
        "test": dataset["test"].to_pandas(),
        "validation": dataset["validation"].to_pandas(),
    }
    if clean:
        dfs = {k: clean_df(shorten(split_df(v))) for k, v in dfs.items()}
    return dfs


def similarity_score(model: SentenceTransformer, df, batch_size=256):
    en_embeddings = model.encode(
        df.en.tolist(), show_progress_bar=True, batch_size=batch_size
    )
    no_embeddings = model.encode(
        df.no.tolist(), show_progress_bar=True, batch_size=batch_size
    )

    sims = []
    for i in tqdm(range(len(en_embeddings))):
        sim = util.cos_sim(en_embeddings[i], no_embeddings[i])
        sim = sim.cpu().numpy().item()
        sims.append(sim)
    return sims


def filter_example_on_len(example, max_len):
    if example:
        return example if len(example) < max_len else None


def filter_df_on_len(df, lang1="en", lang2="no", max_len=256):
    df[lang1] = df[lang1].apply(lambda example: filter_example_on_len(example, max_len))
    df[lang2] = df[lang2].apply(lambda example: filter_example_on_len(example, max_len))
    df = df[df[lang1].notna()]
    df = df[df[lang2].notna()]
    return df


def compress_df(df, lang1, lang2, compression_ratio):
    return df[df[lang2].str.len() < compression_ratio * df[lang1].str.len()]
