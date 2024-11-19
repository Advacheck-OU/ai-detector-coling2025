from pathlib import Path
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm.auto import tqdm
import argparse
import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.inference.infer import MLayerDebertaV2ForSequenceClassification
from sklearn.metrics import classification_report, accuracy_score, f1_score
from logging_setup import setup_logger
import torch

logger = setup_logger("inference")

def main(): 
    parser = argparse.ArgumentParser(description="Inference multi-tasl model.")
    parser.add_argument("--model_checkpoint", type=str, required=True, help="Path to the model configuration folder.")
    parser.add_argument("--threshold", type=float, default=0.92, help="Selected threshold for classification.")
    parser.add_argument("--device", type=str, default='cuda', help="Selected device for inference.")
    parser.add_argument("--batch_size", type=float, default=64, help="Selected batch_size for inference.")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    model = MLayerDebertaV2ForSequenceClassification.from_pretrained()
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