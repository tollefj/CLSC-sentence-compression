import os

import pandas as pd


def datagetter_europarl(lang1, lang2, **kwargs):
    path = f"data/europarl/{lang1}-{lang2}.csv"
    if not os.path.exists(path):
        print(f"File not found: {path}")
        path = f"data/europarl/{lang2}-{lang1}.csv"
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return None

    return pd.read_csv(path)


def datagetter_opensub(
    lang1, lang2, compression_level=0.5, split="test", folder="compressed"
):
    path = (
        f"data/opensubtitles/{folder}/{lang1}-{lang2}-{compression_level}.{split}.csv"
    )
    print(f"Data path: {path}")
    if not os.path.exists(path):
        print(f"File not found: {path}")
        path = f"data/opensubtitles/{folder}/{lang2}-{lang1}-{compression_level}.{split}.csv"
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return None
    return pd.read_csv(path)


def datagetter_tatoeba(lang1, lang2, **kwargs):
    path = f"data/tatoeba/{lang1}-{lang2}.csv"
    print(f"Data path: {path}")
    if not os.path.exists(path):
        print(f"File not found: {path}")
        path = f"data/tatoeba/{lang2}-{lang1}.csv"
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return None

    return pd.read_csv(path)


def get_datagetters(custom_output_folder: str = None):
    datagetters = {
        "europarl": datagetter_europarl,
        "opensub": datagetter_opensub,
        # "tatoeba": datagetter_tatoeba,
    }
    if custom_output_folder:
        # e.g. "copmression-nolimit" for the opensubtitles dataset,
        # where no limitations on samples are applied
        datagetters["opensub"] = lambda *args, **kwargs: datagetter_opensub(
            *args, **kwargs, folder=custom_output_folder
        )
    return datagetters
    # def datagetter(kind, *args, **kwargs):
    #     return datagetters[kind](*args, **kwargs)

    # datagetter("opensub", "en", "lt", compression_level=0.5).head()
