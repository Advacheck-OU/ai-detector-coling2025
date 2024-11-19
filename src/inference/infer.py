import torch
from transformers import AutoTokenizer, DebertaV2ForSequenceClassification

class MLayerDebertaV2ForSequenceClassification(DebertaV2ForSequenceClassification):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(config.hidden_size, 512),
            torch.nn.GELU(),
            torch.nn.Linear(512, 256),
            torch.nn.GELU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(256, 2)
        )

