import wandb
from datasets import Dataset
from transformers import (
    DataCollatorForSeq2Seq,
    MarianMTModel,
    MarianTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


def get_trainer(
    model: MarianMTModel,
    tokenizer: MarianTokenizer,
    dataset: Dataset,
    config: dict,
):
    num_steps = len(dataset) // config["batch_size"]
    save_steps = max(num_steps // 3, 100)

    args = Seq2SeqTrainingArguments(
        learning_rate=config["lr"],
        per_device_train_batch_size=config["batch_size"],
        save_total_limit=1,
        num_train_epochs=config["epochs"],
        # predict_with_generate=True,
        # generation_max_length=max_len,
        disable_tqdm=False,
        warmup_ratio=0.1,
        output_dir=config["outdir"],
        save_only_model=True,  # avoid deepspeed optimizer saving
        save_steps=save_steps,
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    return trainer


def init_wandb(
    project_name: str,
    model_name: str,
    _config: dict,
):
    wandb.init(project=project_name, name=model_name)
    wandb.config.update(_config)
