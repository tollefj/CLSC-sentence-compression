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

from config import COMPRESSION_RATIOS, OPUS_MT_MODELS, prefixes
from utils.training_utils import tokenize_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUT_DIR = "OUTPUT-MM"


def run(
    compression_rate,
    lang1,
    lang2,
    batch_size,
    max_len,
    epochs,
    lr,
):
    if lang2 not in OPUS_MT_MODELS:
        raise ValueError(
            f"Language {lang2} not currently supported. Please add a translation model in the config.py file."
        )

    model_id = OPUS_MT_MODELS[lang2]
    model = MarianMTModel.from_pretrained(model_id).to(device)
    tokenizer = MarianTokenizer.from_pretrained(model_id)
    project_name = f"TranslateCompress-250k"
    model_name = f"comp_{compression_rate}"
    wandb.init(project=project_name, name=model_name)
    wandb.config.update(
        {
            "run_name": model_name,
            "compression_rate": compression_rate,
            "lang1": lang1,
            "lang2": lang2,
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
        }
    )

    dataset = tokenize_data(
        data_path="data",
        compression_rate=compression_rate,
        tokenizer=tokenizer,
        source_lang=lang1,
        target_lang=lang2,
        prefix=None if lang2 not in prefixes else prefixes[lang2],
        max_input_length=max_len,
        max_target_length=max_len,
        max_size=250_000,
    )

    outdir = f"{OUT_DIR}/{lang1}-{lang2}/{compression_rate}"
    os.makedirs(outdir, exist_ok=True)

    # calculate the number of steps per epoch
    num_steps = len(dataset["train"]) // batch_size
    # attempt to save at least 3 times per epoch
    save_steps = max(num_steps // 3, 100)

    args = Seq2SeqTrainingArguments(
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        save_total_limit=1,
        save_only_model=True,  # we're only interested in inference from the fine-tunes.
        num_train_epochs=epochs,
        predict_with_generate=True,
        disable_tqdm=False,
        warmup_ratio=0.1,
        generation_max_length=max_len,
        output_dir=outdir,
        # make sure we store at least one model...
        save_steps=save_steps,
        # lr_scheduler_type="cosine",
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=dataset["train"],
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

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Avaliable models: {OPUS_MT_MODELS.keys()}")
    print(f"Stored: {os.listdir(OUT_DIR)}")
    for lang2 in sorted(OPUS_MT_MODELS.keys()):
        print(f"Training model for {lang1}-{lang2}")
        for compression_rate in COMPRESSION_RATIOS:
            outdir = f"{OUT_DIR}/{lang1}-{lang2}/{compression_rate}"
            if os.path.exists(outdir) and len(os.listdir(outdir)) > 0:
                print(f"Skipping compression rate {compression_rate}. Already trained.")
                continue
            print(f"Training model with compression rate {compression_rate}")
            run(
                compression_rate=compression_rate,
                lang1=lang1,
                lang2=lang2,
                epochs=epochs,
                lr=lr,
                batch_size=batch_size,
                max_len=max_len,
            )
