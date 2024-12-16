import os
from typing import List

from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer, pipeline


def load_model(
    path,
    device="cuda",
):
    if not os.path.exists(path):
        print(f"Path {path} does not exist.")
        return None, None

    last_checkpoint = sorted(os.listdir(path))[-1]
    path = os.path.join(path, last_checkpoint)
    model = MarianMTModel.from_pretrained(path).to(device)
    tokenizer = MarianTokenizer.from_pretrained(path)
    return model, tokenizer


def load_baseline(model_id, device="cuda"):
    baseline_model = MarianMTModel.from_pretrained(model_id).to(device)
    baseline_tokenizer = MarianTokenizer.from_pretrained(model_id)
    return baseline_model, baseline_tokenizer


def generate(model, tokenizer, source_texts, device="cuda", prefix="", max_length=200):
    txts = [f"{prefix} {source_text}".strip() for source_text in source_texts]
    inputs = tokenizer.batch_encode_plus(
        txts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)
    translated = model.generate(**inputs)
    translated_texts = [
        tokenizer.decode(t, skip_special_tokens=True) for t in translated
    ]
    return translated_texts


def translate_all(
    source: List[str],
    model: MarianMTModel,
    tokenizer: MarianTokenizer,
    batch_size: int = 1,
    device: str = "cuda",
    prefix: str = "",
):
    translated = []
    for i in tqdm(range(0, len(source), batch_size)):
        batch = source[i : i + batch_size]
        translated.extend(
            generate(model, tokenizer, source_texts=batch, device=device, prefix=prefix)
        )
    return translated
