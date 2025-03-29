import torch
from transformers import Trainer, TrainingArguments

def train(model, tokenizer, train_dataset, val_dataset):
    training_args = TrainingArguments(
        output_dir="./models",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    trainer.train()
