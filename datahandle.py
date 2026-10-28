from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import BertTokenizer

DATA_DIR = Path(__file__).resolve().parent / "data" / "raw"

dataset_dict = load_dataset(
    "csv",
    data_files={
        "train": str(DATA_DIR / "train.csv"),
        "validation": str(DATA_DIR / "validation.csv"),
        "test": str(DATA_DIR / "test.csv"),
    },
)

my_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def collate_fn(data):
    sents = [i["text"] for i in data]
    labels = [i["label"] for i in data]
    print("sents", sents)

    encoded = my_tokenizer(
        sents,
        truncation=True,
        padding="max_length",
        max_length=32,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    token_type_ids = encoded["token_type_ids"]
    labels = torch.LongTensor(labels)

    return input_ids, attention_mask, token_type_ids, labels


if __name__ == "__main__":
    my_dataloader = DataLoader(
        dataset_dict["train"],
        batch_size=8,
        collate_fn=collate_fn,
        shuffle=False,
    )

    print("my_dataloader--->", len(my_dataloader))
