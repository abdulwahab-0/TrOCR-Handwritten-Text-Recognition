import os
import torch

from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator
)


def train_trocr(
    model,
    processor,
    train_dataset,
    validation_dataset,
    output_directory,
    epochs=3,
    batch_size=2,
    learning_rate=5e-5
):
    """
    Fine-tune the TrOCR model on the IAM dataset.
    """

    os.makedirs(output_directory, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_directory,

        num_train_epochs=epochs,

        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,

        gradient_accumulation_steps=4,

        learning_rate=learning_rate,
        weight_decay=0.01,

        eval_strategy="epoch",
        save_strategy="epoch",

        logging_strategy="steps",
        logging_steps=100,

        predict_with_generate=True,

        fp16=torch.cuda.is_available(),

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        save_total_limit=2,

        report_to="none",
        remove_unused_columns=False,

        dataloader_num_workers=0,

        seed=42
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=default_data_collator,
        processing_class=processor
    )

    # Compatibility adjustment for the TrOCR model
    trainer.model_accepts_loss_kwargs = False

    print("Starting TrOCR fine-tuning...")

    trainer.train()

    print("Fine-tuning completed!")

    trainer.save_model(
        os.path.join(output_directory, "final_model")
    )

    processor.save_pretrained(
        os.path.join(output_directory, "final_model")
    )

    return trainer
