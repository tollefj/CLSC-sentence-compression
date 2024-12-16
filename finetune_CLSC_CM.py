import os

import torch
import wandb
from transformers import (
    DataCollatorForSeq2Seq,
    MarianMTModel,
    MarianTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from config import OPUS_MT_MODELS, prefixes
from utils.training_utils import tokenize_all_compression_levels

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run(
    lang1,
    lang2,
    batch_size,
    max_len,
    epochs,
    lr,
    output_dir="OUTPUT-CM",
):
    if lang2 not in OPUS_MT_MODELS:
        raise ValueError(
            f"Language {lang2} not currently supported. Please add a translation model in the config.py file."
        )

    model_id = OPUS_MT_MODELS[lang2]
    model = MarianMTModel.from_pretrained(model_id).to(device)
    tokenizer = MarianTokenizer.from_pretrained(model_id)
    project_name = f"TranslateCompress-v2"
    model_name = f"{lang1}-{lang2}-flexitoken"
    wandb.init(project=project_name, name=model_name)
    wandb.config.update(
        {
            "run_name": model_name,
            "lang1": lang1,
            "lang2": lang2,
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
        }
    )

    dataset = tokenize_all_compression_levels(
        "data",
        tokenizer,
        lang1,
        lang2,
        prefix=None if lang2 not in prefixes else prefixes[lang2],
        max_input_length=max_len,
        max_target_length=max_len,
        max_size=250000,
    )

    outdir = f"{output_dir}/{lang1}-{lang2}"
    os.makedirs(outdir, exist_ok=True)

    args = Seq2SeqTrainingArguments(
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        # per_device_eval_batch_size=batch_size,
        save_total_limit=1,
        num_train_epochs=epochs,
        # predict_with_generate=True,
        disable_tqdm=False,
        warmup_ratio=0.1,
        # generation_max_length=max_len,
        output_dir=outdir,
        save_only_model=True,  # avoid deepspeed optimizer saving
        # lr_scheduler_type="cosine",
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    wandb.finish()


if __name__ == "__main__":
    batch_size = 16
    epochs = 1
    lr = 5e-6
    max_len = 200
    lang1 = "en"

    out_dir = "OUTPUT-CM"
    os.makedirs(out_dir, exist_ok=True)
    print(f"Avaliable models: {OPUS_MT_MODELS.keys()}")
    print(f"Stored: {os.listdir(out_dir)}")
    for lang2 in sorted(OPUS_MT_MODELS.keys()):
        if f"{lang1}-{lang2}" in os.listdir(out_dir):
            print(f"Skipping {lang1}-{lang2}")
            continue

        print(f"Training model for {lang1}-{lang2}")
        run(
            lang1=lang1,
            lang2=lang2,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            max_len=max_len,
            output_dir=out_dir,
        )
