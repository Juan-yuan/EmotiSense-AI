import os
from pathlib import Path

import torch
import torch.nn as nn
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from transformers import BertModel

from datahandle import collate_fn, dataset_dict

load_dotenv(Path(__file__).resolve().parent / ".env")

PRETRAINED_MODEL = os.getenv("PRETRAINED_MODEL", "bert-base-uncased")
HIDDEN_SIZE = int(os.getenv("HIDDEN_SIZE", "768"))
NUM_CLASSES = int(os.getenv("NUM_CLASSES", "3"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))
SHUFFLE_TRAIN = os.getenv("SHUFFLE_TRAIN", "true").lower() == "true"
FREEZE_BERT = os.getenv("FREEZE_BERT", "true").lower() == "true"

model_pretrained = BertModel.from_pretrained(PRETRAINED_MODEL)


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(HIDDEN_SIZE, NUM_CLASSES)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # Frozen BERT as feature extractor: [batch, seq] -> pooler [batch, hidden]
        with torch.no_grad():
            out = model_pretrained(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

        # Trainable classification head: [batch, hidden] -> [batch, num_classes]
        out = self.fc(out.pooler_output)
        return out


if __name__ == "__main__":
    my_dataloader = DataLoader(
        dataset=dataset_dict["train"],
        batch_size=BATCH_SIZE,
        collate_fn=collate_fn,
        shuffle=SHUFFLE_TRAIN,
        drop_last=True,
    )

    if FREEZE_BERT:
        for param in model_pretrained.parameters():
            param.requires_grad_(False)

    my_model = MyModel()
    print("my_model--->", my_model)

    for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(
        my_dataloader
    ):
        y_out = my_model(input_ids, attention_mask, token_type_ids)
        print("y_out---->", y_out.shape, y_out)
        break
