import argparse
import yaml
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.training.train import train_model

def main():
    parser = argparse.ArgumentParser(description="Train multi-tasl model.")
    parser.add_argument("--config", type=str, required=True, help="Path to the training configuration file.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    train_model(config["train_params"])

if __name__ == "__main__":
    main()
