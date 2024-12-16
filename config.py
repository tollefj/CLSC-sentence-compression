COMPRESSION_RATIOS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# sample sizes used to verify usability of a large set of data
sample_sizes = [
    500_000,
    250_000,
    100_000,
    50_000,
    10_000,
]

# output folders for trained models
output_folders = {
    "singlemodel": "OUTPUT-MM",
    "tokenmodel": "OUTPUT-CM",
}

LANGUAGES = {
    "fr": "French",
    "hu": "Hungarian",
    "lt": "Lithuanian",
    "no": "Norwegian",
    "pl": "Polish",
    # low-resource:
    "ms": "Malay",
    "sq": "Albanian",
    "eu": "Basque",
}

# language family oriented models
OPUS_MT_MODELS_FAMILY = {
    "germanic": "Helsinki-NLP/opus-mt-en-gmq",
    "baltic": "Helsinki-NLP/opus-mt-en-bat",
    "west_slavic": "Helsinki-NLP/opus-mt-en-zlw",
    "uralic": "Helsinki-NLP/opus-mt-en-urj",
    "romance": "Helsinki-NLP/opus-mt-en-roa",
    # low-resource
    "austronesian": "Helsinki-NLP/opus-mt-en-map",
    "albanoid": "Helsinki-NLP/opus-mt-en-sq",
    "basque": "Helsinki-NLP/opus-mt-en-eu",
}
LANG_FAMILY_TO_LANGUAGE = {
    OPUS_MT_MODELS_FAMILY["germanic"]: ["no"],
    OPUS_MT_MODELS_FAMILY["uralic"]: ["hu"],
    OPUS_MT_MODELS_FAMILY["baltic"]: ["lt"],
    OPUS_MT_MODELS_FAMILY["romance"]: ["fr"],
    OPUS_MT_MODELS_FAMILY["west_slavic"]: ["pl"],
    # low-resource
    OPUS_MT_MODELS_FAMILY["austronesian"]: ["ms"],
    OPUS_MT_MODELS_FAMILY["albanoid"]: ["sq"],
    OPUS_MT_MODELS_FAMILY["basque"]: ["eu"],
}
OPUS_MT_MODELS = {
    lang: model
    for model in OPUS_MT_MODELS_FAMILY.values()
    for lang in LANG_FAMILY_TO_LANGUAGE[model]
}
print("Loading config. model overview:", OPUS_MT_MODELS)


def get_family(lang) -> str:
    for family, langs in LANG_FAMILY_TO_LANGUAGE.items():
        if lang in langs:
            return family
    raise ValueError(f"Language {lang} not found in any family")


LANG_CODES = {
    "fr": "fra",
    "hu": "hun",
    "lt": "lit",
    "no": "nob",  # norwegian bokmål
    "pl": "pol",
    # low resource:
    "ms": "msa",
    "sq": "sqi",
    "eu": "eus",
}

prefixes = {lang: f">>{code}<<" for lang, code in LANG_CODES.items()}


def get_prefix(lang):
    if lang in LANG_CODES:
        return f">>{LANG_CODES[lang]}<<"
    return ""


def get_compression_prefix(compression: float, lang: str):
    return f"@{compression} {get_prefix(lang)}"
