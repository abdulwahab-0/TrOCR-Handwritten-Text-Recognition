import os
import pandas as pd


def load_annotations(annotation_file):
    """
    Load IAM handwritten text annotations.
    Keep only valid annotations with status 'ok'.
    """

    records = []

    with open(annotation_file, "r", encoding="utf-8") as file:
        for line in file:

            if line.startswith("#") or not line.strip():
                continue

            parts = line.split(maxsplit=8)

            if len(parts) < 9:
                continue

            line_id = parts[0]
            status = parts[1]
            transcription = parts[8]

            if status == "ok":
                text = transcription.replace("|", " ").strip()

                records.append({
                    "line_id": line_id,
                    "text": text
                })

    return pd.DataFrame(records)


def load_split_ids(split_file):
    """Load official IAM train, validation, or test IDs."""

    with open(split_file, "r", encoding="utf-8") as file:
        return {
            line.strip()
            for line in file
            if line.strip() and not line.startswith("#")
        }


def prepare_split(annotation_df, split_ids, image_directory):
    """Match annotation IDs with images and prepare a dataframe."""

    split_df = annotation_df[
        annotation_df["line_id"].isin(split_ids)
    ].copy()

    split_df["image_path"] = split_df["line_id"].apply(
        lambda line_id: os.path.join(
            image_directory,
            line_id.split("-")[0],
            "-".join(line_id.split("-")[:2]),
            line_id + ".png"
        )
    )

    split_df = split_df[
        split_df["image_path"].apply(os.path.exists)
    ].copy()

    return split_df.reset_index(drop=True)


def save_dataset_splits(
    annotation_file,
    train_split_file,
    validation_split_file,
    test_split_file,
    image_directory,
    output_directory
):
    """Prepare and save all three IAM dataset splits."""

    os.makedirs(output_directory, exist_ok=True)

    annotations = load_annotations(annotation_file)

    train_ids = load_split_ids(train_split_file)
    validation_ids = load_split_ids(validation_split_file)
    test_ids = load_split_ids(test_split_file)

    train_df = prepare_split(
        annotations, train_ids, image_directory
    )

    validation_df = prepare_split(
        annotations, validation_ids, image_directory
    )

    test_df = prepare_split(
        annotations, test_ids, image_directory
    )

    train_df.to_csv(
        os.path.join(output_directory, "train.csv"),
        index=False
    )

    validation_df.to_csv(
        os.path.join(output_directory, "validation.csv"),
        index=False
    )

    test_df.to_csv(
        os.path.join(output_directory, "test.csv"),
        index=False
    )

    print("Dataset preparation completed!")
    print("Training samples:", len(train_df))
    print("Validation samples:", len(validation_df))
    print("Testing samples:", len(test_df))

    return train_df, validation_df, test_df
