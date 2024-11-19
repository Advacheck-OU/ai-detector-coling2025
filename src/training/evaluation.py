from src.model.data_collator import DataLoaderWithTaskname
import nlp
import numpy as np
import torch
import transformers
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import balanced_accuracy_score
from tqdm.auto import tqdm
from logging_setup import setup_logger

logger = setup_logger("eval")


def multitask_test(multitask_model, model_name, features_dict, tasks, batch_size=8):
    """
    Evaluates a multitask model on multiple tasks using given validation features.

    Args:
        multitask_model (torch.nn.Module): The multitask model to be evaluated.
        model_name (str): The name of the pre-trained model for tokenizer initialization.
        features_dict (dict): A dictionary containing task-specific validation data.
                              Each task key maps to a dictionary with "validation" features.
        tasks (list): List of task names to evaluate the model on.
        batch_size (int, optional): Batch size for processing validation data. Defaults to 8.

    Returns:
        None
    """
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    multitask_model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if next(multitask_model.parameters()).device != device:
        multitask_model.to(device)

    for task_name in tasks:
        val_len = len(features_dict[task_name]["validation"])
        pred_vals = []
        lab_vals = []

        for index in tqdm(range(0, val_len)):
            batch = features_dict[task_name]["validation"][index]["doc"]
            label = features_dict[task_name]["validation"][index]["target"]
            inputs = tokenizer(batch, max_length=512, truncation=True, padding="max_length", return_tensors="pt").to(device)

            with torch.no_grad():
                logits = multitask_model(**inputs, task_name=task_name)[0]

            preds = torch.argmax(
                    torch.FloatTensor(torch.softmax(logits, dim=1).detach().cpu().tolist()),
                    dim=1,
            )
     
            pred_vals.append(int(preds))
            lab_vals.append(label)
            
        acc = accuracy_score(y_true=np.array(lab_vals), y_pred=np.array(pred_vals)) * 100
        bal_acc = balanced_accuracy_score(y_true=np.array(lab_vals), y_pred=np.array(pred_vals)) * 100
        f1 = f1_score(y_true=np.array(lab_vals), y_pred=np.array(pred_vals), average='macro', labels=np.unique(np.array(lab_vals))) * 100

        logger.info(f"Task name: {task_name} \t Accuracy: {acc} \t BalAccuracy: {bal_acc} \t macro-f1-score: {f1}")

