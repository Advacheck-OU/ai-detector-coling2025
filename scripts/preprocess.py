from datasets import load_dataset
from tqdm.auto import tqdm
import pandas as pd
import argparse
from logging_setup import setup_logger
import os

logger = setup_logger("preprocess")

ALLOWED_STRINGS = ['full_data', 'hc3', 'm4gt', 'mage']


def write_to_df(x_train, y_train, part, stage, cur_path):
    """
    Writes training and label data to a CSV file.

    Args:
        x_train (list): List of training data (texts).
        y_train (list): List of corresponding labels for the training data.
        part (str): The data subset name (e.g., 'full_data', 'hc3').
        stage (str): The data stage (e.g., 'train', 'dev').
        cur_path (list): List of path components split from the current working directory.

    Returns:
        None
    """
    df = pd.DataFrame({
        'Doc': [],
        'Target': [],
    })
    for item, lab in tqdm(zip(x_train, y_train)):
        df = pd.concat([df, pd.DataFrame([[item,
                                           int(lab),
                                           ]], columns=['Doc', 'Target'])], ignore_index=True)

    df.to_csv(f"{cur_path[0]}/ai-detector-coling2025/data/{part}/{stage}.csv", index=False)


def make_data(strings):
    """
    Prepares data directories and fills them with processed data.

    Args:
        strings (list): List of dataset subsets to process (e.g., 'full_data', 'hc3').

    Returns:
        None
    """
    logger.info(f"Creating directory for data.")
    cur_path = os.getcwd().split("ai-detector-coling2025")
    os.makedirs(f'{cur_path[0]}/ai-detector-coling2025/data/', exist_ok=True)
    for item in strings:
        os.makedirs(f'{cur_path[0]}/ai-detector-coling2025/data/{item}', exist_ok=True)

    ds = load_dataset("Jinyan1/COLING_2025_MGT_en")

    logger.info(
        f"Filling directories depending on the sample. Filling will be done for {len(strings) * 2} sub-samples.")
    for stage in ["train", "dev"]:
        for item in strings:
            if item == 'full_data':
                write_to_df(ds[stage]["text"],
                            ds[stage]["label"],
                            item,
                            stage,
                            cur_path)
            else:
                precise_ds = [sample for sample in ds[stage]
                              if sample["source"] == item]
                unique_subsources = set(sample["sub_source"]
                                        for sample in precise_ds)
                new_labels = {s: j for j, s in enumerate(unique_subsources)}
                write_to_df([sample["text"] for sample in precise_ds],
                            [new_labels[sample["sub_source"]]
                                for sample in precise_ds],
                            item,
                            stage,
                            cur_path)


def main():
    parser = argparse.ArgumentParser(description="List of chosen heads.")
    parser.add_argument(
        "--classification_heads",
        nargs="*",
        help="List of classification heads to parse. If none are provided, defaults will be used."
    )

    args = parser.parse_args()

    strings = args.classification_heads if args.classification_heads else [
        "full_data", "hc3", "m4gt"]
    logger.info(f"Received classification heads: {strings}")

    for string in strings:
        if string not in ALLOWED_STRINGS:
            raise ValueError(
                f"Invalid string: '{string}'. Allowed values are: {ALLOWED_STRINGS}")

    make_data(strings)


if __name__ == "__main__":
    main()
