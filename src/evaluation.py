import os
import pandas as pd
import torch

from jiwer import cer, wer
from tqdm.auto import tqdm


def evaluate_model(
    model,
    processor,
    test_dataset,
    test_dataframe,
    output_directory,
    batch_size=2
):
    """
    Evaluate TrOCR using Character Error Rate (CER)
    and Word Error Rate (WER).
    """

    os.makedirs(output_directory, exist_ok=True)

    device = next(model.parameters()).device
    model.eval()

    predictions = []
    references = []

    print("Starting model evaluation...")

    for start in tqdm(
        range(0, len(test_dataset), batch_size)
    ):
        end = min(start + batch_size, len(test_dataset))

        batch = [
            test_dataset[index]
            for index in range(start, end)
        ]

        pixel_values = torch.stack([
            item["pixel_values"] for item in batch
        ]).to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                pixel_values,
                max_length=128
            )

        batch_predictions = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )

        predictions.extend(batch_predictions)

        references.extend(
            test_dataframe.iloc[start:end]["text"].tolist()
        )

    # Calculate evaluation metrics
    test_cer = cer(references, predictions)
    test_wer = wer(references, predictions)

    print("\nEvaluation Results")
    print("------------------")
    print(f"CER: {test_cer:.4f} ({test_cer * 100:.2f}%)")
    print(f"WER: {test_wer:.4f} ({test_wer * 100:.2f}%)")

    # Save individual predictions
    results_df = pd.DataFrame({
        "reference": references,
        "prediction": predictions
    })

    results_df.to_csv(
        os.path.join(output_directory, "predictions.csv"),
        index=False
    )

    # Save summary metrics
    metrics_df = pd.DataFrame([{
        "CER": test_cer,
        "WER": test_wer
    }])

    metrics_df.to_csv(
        os.path.join(output_directory, "evaluation_metrics.csv"),
        index=False
    )

    print("\nEvaluation files saved successfully!")

    return test_cer, test_wer, results_df
