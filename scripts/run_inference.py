from pathlib import Path
from transformers import AutoTokenizer, DebertaV2Config
from datasets import load_dataset
from tqdm.auto import tqdm
import argparse
import torch
import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.inference.infer import MLayerDebertaV2ForSequenceClassification
from src.model.multitask_model import DebertaV2ForSequenceClassification
from sklearn.metrics import classification_report, accuracy_score, f1_score
from logging_setup import setup_logger

logger = setup_logger("inference")

def load_checkpoint_to_one_head_model(checkpoint_path):
    """
    Loads a multitask model checkpoint and initializes a single-head classification model.
    
    This function is designed to load a multitask model checkpoint and transfer shared weights 
    to a new single-head classification model. The weights from the specified task's classification 
    head in the multitask model are used to initialize the new model's classifier. If the task-specific 
    head is not found, the new classifier is initialized randomly.
    
    Args:
        checkpoint_path (str): Path to the directory containing the multitask model checkpoint.
    
    Returns:
        MLayerDebertaV2ForSequenceClassification: A single-head classification model 
        initialized with the shared weights and the classification head weights 
        (if available) from the multitask model.
    """
    config = DebertaV2Config.from_pretrained(checkpoint_path)
    
    tasks = ["full_data", "hc3", "m4gt"] # labels don't matter here, only the head of 'full_data' is important
    labels = [2, 5, 6]
    base_model = DebertaV2ForSequenceClassification.from_pretrained(checkpoint_path, 
                                                                    task_labels_map=dict(zip(tasks, labels)))

    new_model = MLayerDebertaV2ForSequenceClassification(config)
    new_model.deberta.load_state_dict(base_model.deberta.state_dict())
    new_model.pooler.load_state_dict(base_model.pooler.state_dict())
    
    task_name = "full_data"  # change this to a relevant task
    if task_name in base_model.classifiers:
        new_model.classifier.load_state_dict(base_model.classifiers[task_name].state_dict(), strict=False)
    else:
        logger.info(f"Task {task_name} not found. We initialize new classifier randomly.")
        
    return new_model

def main(): 
    parser = argparse.ArgumentParser(description="Inference multi-tasl model.")
    parser.add_argument("--model_checkpoint", type=str, required=True, help="Path to the model configuration folder.")
    parser.add_argument("--threshold", type=float, default=0.92, help="Selected threshold for classification.")
    parser.add_argument("--device", type=str, default='cuda', help="Selected device for inference.")
    parser.add_argument("--batch_size", type=float, default=64, help="Selected batch_size for inference.")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    model = load_checkpoint_to_one_head_model(args.model_checkpoint)
    model.eval()
    model.to(args.device)

    data = load_dataset("Jinyan1/COLING_2025_MGT_en")["dev"]
    
    predicted_values = []
    for i in tqdm(range(0, len(data["text"]), args.batch_size)):
        batch = data["text"][i:i+args.batch_size]
        inputs = tokenizer(batch, max_length=512, truncation=True, padding="max_length", return_tensors="pt").to(args.device)
        
        with torch.no_grad():
            logits = model(**inputs)[0]

        probs = torch.FloatTensor(torch.softmax(logits, dim=1).detach().cpu().tolist())
        int_preds = [1 if item[1] > args.threshold else 0 for item in probs]

        predicted_values.extend(int_preds)

    logger.info(f"Model on dev set: f1-score={f1_score(data['label'], predicted_values)}, accuracy={accuracy_score(data['label'], predicted_values)}")
    
if __name__ == "__main__":
    main()
