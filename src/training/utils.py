import random
import torch
import numpy as np
import transformers
import torch
import os

def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    
def convert_to_features(example_batch, model_name="microsoft/deberta-v3-base", max_length=512):
    """
    Converts a batch of examples into model-compatible input features.

    Args:
        example_batch (dict): A batch of examples containing "doc" (input text) and "target" (labels).
        model_name (str, optional): Name of the pre-trained model for tokenizer initialization. Defaults to "microsoft/deberta-v3-base".
        max_length (int, optional): Maximum sequence length for tokenization. Defaults to 512.

    Returns:
        dict: Tokenized input features with "input_ids", "attention_mask", and "labels".
    """
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, use_fast=False)
    inputs = list(example_batch["doc"])

    features = tokenizer(
        inputs,
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )
    features["labels"] = example_batch["target"]
    return features

def save_model(path, model_name, multitask_model):
    """
    Saves a multitask model and tokenizer to a specified directory.

    Args:
        path (str): Base directory where the model checkpoint will be saved.
        model_name (str): Name of the pre-trained model for tokenizer saving.
        multitask_model (torch.nn.Module): The multitask model to be saved.

    Returns:
        None
    """
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    
    directory_name = 'src/training/model_checkpoints/final_model/'
    
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    
    multitask_model.deberta.config.to_json_file(f"{directory_name}/config.json")
    torch.save(
        multitask_model.state_dict(),
        f"{directory_name}/pytorch_model.bin",
    )
    tokenizer.save_pretrained(directory_name)

    
    