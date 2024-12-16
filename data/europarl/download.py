import os
import random
import re

import pandas as pd
from datasets import load_dataset

supported = [
    ("bg", "cs"),
    ("bg", "da"),
    ("bg", "de"),
    ("bg", "el"),
    ("bg", "en"),
    ("bg", "es"),
    ("bg", "et"),
    ("bg", "fi"),
    ("bg", "fr"),
    ("bg", "hu"),
    ("bg", "it"),
    ("bg", "lt"),
    ("bg", "lv"),
    ("bg", "nl"),
    ("bg", "pl"),
    ("bg", "pt"),
    ("bg", "ro"),
    ("bg", "sk"),
    ("bg", "sl"),
    ("bg", "sv"),
    ("cs", "da"),
    ("cs", "de"),
    ("cs", "el"),
    ("cs", "en"),
    ("cs", "es"),
    ("cs", "et"),
    ("cs", "fi"),
    ("cs", "fr"),
    ("cs", "hu"),
    ("cs", "it"),
    ("cs", "lt"),
    ("cs", "lv"),
    ("cs", "nl"),
    ("cs", "pl"),
    ("cs", "pt"),
    ("cs", "ro"),
    ("cs", "sk"),
    ("cs", "sl"),
    ("cs", "sv"),
    ("da", "de"),
    ("da", "el"),
    ("da", "en"),
    ("da", "es"),
    ("da", "et"),
    ("da", "fi"),
    ("da", "fr"),
    ("da", "hu"),
    ("da", "it"),
    ("da", "lt"),
    ("da", "lv"),
    ("da", "nl"),
    ("da", "pl"),
    ("da", "pt"),
    ("da", "ro"),
    ("da", "sk"),
    ("da", "sl"),
    ("da", "sv"),
    ("de", "el"),
    ("de", "en"),
    ("de", "es"),
    ("de", "et"),
    ("de", "fi"),
    ("de", "fr"),
    ("de", "hu"),
    ("de", "it"),
    ("de", "lt"),
    ("de", "lv"),
    ("de", "nl"),
    ("de", "pl"),
    ("de", "pt"),
    ("de", "ro"),
    ("de", "sk"),
    ("de", "sl"),
    ("de", "sv"),
    ("el", "en"),
    ("el", "es"),
    ("el", "et"),
    ("el", "fi"),
    ("el", "fr"),
    ("el", "hu"),
    ("el", "it"),
    ("el", "lt"),
    ("el", "lv"),
    ("el", "nl"),
    ("el", "pl"),
    ("el", "pt"),
    ("el", "ro"),
    ("el", "sk"),
    ("el", "sl"),
    ("el", "sv"),
    ("en", "es"),
    ("en", "et"),
    ("en", "fi"),
    ("en", "fr"),
    ("en", "hu"),
    ("en", "it"),
    ("en", "lt"),
    ("en", "lv"),
    ("en", "nl"),
    ("en", "pl"),
    ("en", "pt"),
    ("en", "ro"),
    ("en", "sk"),
    ("en", "sl"),
    ("en", "sv"),
    ("es", "et"),
    ("es", "fi"),
    ("es", "fr"),
    ("es", "hu"),
    ("es", "it"),
    ("es", "lt"),
    ("es", "lv"),
    ("es", "nl"),
    ("es", "pl"),
    ("es", "pt"),
    ("es", "ro"),
    ("es", "sk"),
    ("es", "sl"),
    ("es", "sv"),
    ("et", "fi"),
    ("et", "fr"),
    ("et", "hu"),
    ("et", "it"),
    ("et", "lt"),
    ("et", "lv"),
    ("et", "nl"),
    ("et", "pl"),
    ("et", "pt"),
    ("et", "ro"),
    ("et", "sk"),
    ("et", "sl"),
    ("et", "sv"),
    ("fi", "fr"),
    ("fi", "hu"),
    ("fi", "it"),
    ("fi", "lt"),
    ("fi", "lv"),
    ("fi", "nl"),
    ("fi", "pl"),
    ("fi", "pt"),
    ("fi", "ro"),
    ("fi", "sk"),
    ("fi", "sl"),
    ("fi", "sv"),
    ("fr", "hu"),
    ("fr", "it"),
    ("fr", "lt"),
    ("fr", "lv"),
    ("fr", "nl"),
    ("fr", "pl"),
    ("fr", "pt"),
    ("fr", "ro"),
    ("fr", "sk"),
    ("fr", "sl"),
    ("fr", "sv"),
    ("hu", "it"),
    ("hu", "lt"),
    ("hu", "lv"),
    ("hu", "nl"),
    ("hu", "pl"),
    ("hu", "pt"),
    ("hu", "ro"),
    ("hu", "sk"),
    ("hu", "sl"),
    ("hu", "sv"),
    ("it", "lt"),
    ("it", "lv"),
    ("it", "nl"),
    ("it", "pl"),
    ("it", "pt"),
    ("it", "ro"),
    ("it", "sk"),
    ("it", "sl"),
    ("it", "sv"),
    ("lt", "lv"),
    ("lt", "nl"),
    ("lt", "pl"),
    ("lt", "pt"),
    ("lt", "ro"),
    ("lt", "sk"),
    ("lt", "sl"),
    ("lt", "sv"),
    ("lv", "nl"),
    ("lv", "pl"),
    ("lv", "pt"),
    ("lv", "ro"),
    ("lv", "sk"),
    ("lv", "sl"),
    ("lv", "sv"),
    ("nl", "pl"),
    ("nl", "pt"),
    ("nl", "ro"),
    ("nl", "sk"),
    ("nl", "sl"),
    ("nl", "sv"),
    ("pl", "pt"),
    ("pl", "ro"),
    ("pl", "sk"),
    ("pl", "sl"),
    ("pl", "sv"),
    ("pt", "ro"),
    ("pt", "sk"),
    ("pt", "sl"),
    ("pt", "sv"),
    ("ro", "sk"),
    ("ro", "sl"),
    ("ro", "sv"),
    ("sk", "sl"),
    ("sk", "sv"),
    ("sl", "sv"),
]


def clean(text):
    # initial dialogue hyphens
    text = re.sub(r"^-+", "", text)
    # whitespaces
    text = re.sub(r"\s+", " ", text)
    # remove any repeated hyphens
    text = re.sub(r"(-\s*){2,}", "", text)
    return text.strip()


languages = {
    "french": "fr",
    "hungarian": "hu",
    "lithuanian": "lt",
    "polish": "pl",
    # norwegian does not exist for europarl
}


if __name__ == "__main__":
    src_lang_ = "en"
    random.seed(42)

    for lang_long, lang in languages.items():
        src_lang = src_lang_
        if (src_lang, lang) not in supported:
            print(f"Attempting to find a supported language pair for {lang_long}")
            if (lang, src_lang) in supported:
                print(f"Swapping ({lang}, {src_lang}) to ({src_lang}, {lang})")
                src_lang, lang = lang, src_lang
        output_file = f"{src_lang}-{lang}.csv"
        if os.path.exists(output_file):
            print(f"Skipping {lang_long}")
            continue
        print(f"Processing {lang_long}")
        dataset = load_dataset("Helsinki-NLP/europarl", name=f"{src_lang}-{lang}")
        # dataset = load_dataset(
        #     "europarl_bilingual", lang1=src_lang, lang2=lang
        # )
        data = dataset["train"]["translation"]
        df = pd.DataFrame(data)
        # ensure "en" is the first column:
        if lang == "en":
            df = df[[lang, src_lang]]
        # validation samples of 100k parallel sentences
        df = df.sample(n=100_000, random_state=42)
        df[src_lang] = df[src_lang].apply(clean)
        df[lang] = df[lang].apply(clean)
        df.to_csv(f"{src_lang}-{lang}.csv", index=False)
