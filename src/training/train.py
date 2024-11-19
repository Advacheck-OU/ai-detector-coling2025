import os
import torch
import nltk
import glob
import numpy as np
import transformers
from pathlib import Path
from tqdm.auto import tqdm
from datasets import load_dataset
from scripts.logging_setup import setup_logger
from torch.utils.data.dataloader import DataLoader

from src.model.multitask_model import DebertaV2ForSequenceClassification
from src.model.data_collator import MultitaskTrainer, NLPDataCollator
from src.training.utils import convert_to_features, save_model, set_seed
from src.training.evaluation import multitask_test
from sklearn.metrics import classification_report, accuracy_score, f1_score

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

logger = setup_logger("train")


def train_model(config):
    """
    Trains a multitask classification model using task-specific datasets.

    Args:
        config (dict): Configuration dictionary containing:
            - "tasks" (dict): A mapping of task names to the number of labels per task.
            - "model" (str): Name of the pre-trained model to use.
            - "stage" (str): Training stage ("first" for initial training, otherwise 'second' resume from checkpoint).
            - "model_checkpoint" (str): Path to the model checkpoint (used in subsequent stages).

    Returns:
        None
    """
    set_seed(43)

    cur_path = os.getcwd().split("ai-detector-coling2025")
    available_tasks = [item.split("/")[-1] for item in glob.glob(
        os.path.join(cur_path[0], 'ai-detector-coling2025/data/*'))]
    labels_count = [config["tasks"][item] for item in available_tasks]

    logger.info(
        f"Training will be launched for {len(available_tasks)} tasks: {available_tasks}")

    dataset_dict = {}

    for item in available_tasks:
        dataset_dict[item] = load_dataset(
            "src/model/dataloader.py",
            data_files={
                "train": "../../data/{}/train.csv".format(item),
                "validation": "../../data/{}/dev.csv".format(item),
            }, trust_remote_code=True,
        )

    model_names = [config["model"]] * len(available_tasks)
    config_files = model_names
    for idx, task_name in enumerate(available_tasks):
        model_file = Path(f"./{task_name}_model/pytorch_model.bin")
        config_file = Path(f"./{task_name}_model/config.json")
        if model_file.is_file():
            model_names[idx] = f"./{task_name}_model"

        if config_file.is_file():
            config_files[idx] = f"./{task_name}_model"

    if config["stage"] == 'first':
        multitask_model = DebertaV2ForSequenceClassification.from_pretrained(
            config["model"],
            task_labels_map=dict(zip(available_tasks, labels_count)),
        )
        multitask_model.freeze_params(True)
    else:
        multitask_model = DebertaV2ForSequenceClassification.from_pretrained(
            os.path.join(cur_path[0], config["model_checkpoint"]),
            task_labels_map=dict(zip(available_tasks, labels_count)),
        )

    logger.info("Model successfully loaded.")

    convert_func_dict = dict(
        zip(available_tasks, [convert_to_features] * len(available_tasks)))
    columns_dict = dict(
        zip(available_tasks, [["input_ids", "attention_mask", "labels"]] * len(available_tasks)))

    features_dict = {}
    for task_name, dataset in dataset_dict.items():
        features_dict[task_name] = {}
        for phase, phase_dataset in dataset.items():
            features_dict[task_name][phase] = phase_dataset.map(
                convert_func_dict[task_name],
                batched=True,
                load_from_cache_file=False,
            )
            features_dict[task_name][phase].set_format(
                type="torch",
                columns=columns_dict[task_name],
            )
            print(
                task_name,
                phase,
                len(phase_dataset),
                len(features_dict[task_name][phase]),
            )

    train_dataset = {
        task_name: dataset["train"] for task_name, dataset in features_dict.items()
    }

    val_dataset = {
        task_name: dataset["validation"] for task_name, dataset in features_dict.items()
    }

    model_parameters = filter(
        lambda p: p.requires_grad, multitask_model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    logger.info(f"Number of trainable params at this stage: {params}")

    trainer = MultitaskTrainer(
        model=multitask_model,
        args=transformers.TrainingArguments(
            output_dir=os.path.join(cur_path[0], config["output_dir"]),
            overwrite_output_dir=True,
            learning_rate=config["learning_rate"],
            do_train=True,
            do_eval=False,
            warmup_steps=config["warmup_steps"],
            num_train_epochs=config["epochs"],
            per_device_train_batch_size=config["batch_size"],
            per_device_eval_batch_size=config["batch_size"],
            evaluation_strategy='steps',
            metric_for_best_model=config["metric_for_best_model"],
            eval_steps=config["eval_steps"],
            load_best_model_at_end=True,
            save_steps=config["eval_steps"],
            report_to='none',

        ),
        data_collator=NLPDataCollator(),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=lambda pred: {'accuracy': accuracy_score(pred.label_ids, pred.predictions.argmax(-1)),
                                      'macro_f1': f1_score(pred.label_ids, pred.predictions.argmax(-1), average='macro')},
    )
    trainer.train()

    save_model(os.path.join(cur_path[0], 'ai-detector-coling2025/src/training/model_checkpoints/'),
               config["model"],
               multitask_model)

    multitask_test(
        multitask_model, config["model"], dataset_dict, available_tasks)
