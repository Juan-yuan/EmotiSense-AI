import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
from dotenv import load_dotenv
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader

from datahandle import collate_fn, dataset_dict
from model import MyModel, model_pretrained

load_dotenv(_ROOT / ".env")

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))
SHUFFLE_TRAIN = os.getenv("SHUFFLE_TRAIN", "true").lower() == "true"
FREEZE_BERT = os.getenv("FREEZE_BERT", "true").lower() == "true"
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "5e-5"))
EPOCHS = int(os.getenv("EPOCHS", "1"))
LOG_EVERY = int(os.getenv("LOG_EVERY", "10"))
MAX_TRAIN_STEPS = int(os.getenv("MAX_TRAIN_STEPS", "0"))  # 0 = full epoch
MODEL_SAVE_DIR = _ROOT / os.getenv("MODEL_SAVE_DIR", "trained_models")


def train_model():
    my_dataloader = DataLoader(
        dataset=dataset_dict["train"],
        batch_size=BATCH_SIZE,
        collate_fn=collate_fn,
        shuffle=SHUFFLE_TRAIN,
    )

    my_model = MyModel()

    if FREEZE_BERT:
        for param in model_pretrained.parameters():
            param.requires_grad_(False)

    my_optimizer = AdamW(
        filter(lambda p: p.requires_grad, my_model.parameters()),
        lr=LEARNING_RATE,
    )
    my_criterion = CrossEntropyLoss()

    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    my_model.train()

    for epoch_idx in range(EPOCHS):
        starttime = int(time.time())

        for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(
            my_dataloader, start=1
        ):
            my_out = my_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

            my_loss = my_criterion(my_out, labels)

            my_optimizer.zero_grad()
            my_loss.backward()
            my_optimizer.step()

            if i % LOG_EVERY == 0:
                pred = my_out.argmax(dim=1)
                acc = (pred == labels).sum().item() / len(labels)
                print(
                    "epoch:%d step:%d loss:%.6f acc:%.3f elapsed:%ds"
                    % (epoch_idx, i, my_loss.item(), acc, int(time.time()) - starttime)
                )

            if MAX_TRAIN_STEPS > 0 and i >= MAX_TRAIN_STEPS:
                break

        save_path = MODEL_SAVE_DIR / f"model_class_{epoch_idx + 1}.pth"
        torch.save(my_model.state_dict(), save_path)
        print(f"save trained model -> {save_path}")


if __name__ == "__main__":
    train_model()
