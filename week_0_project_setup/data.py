import torch
from datasets import load_dataset
from transformers import AutoTokenizer
import pytorch_lightning as pl


class DataModule(pl.LightningDataModule):
    def __init__(self, model_name="google/bert_uncased_L-2_H-128_A-2", batch_size=32):
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def prepare_data(self):
        self.dataset = load_dataset("glue", "cola")

    def setup(self, stage=None):
        tokenize = lambda x: self.tokenizer(
            x["sentence"], truncation=True, padding="max_length", max_length=512
        )
        data = self.dataset.map(tokenize, batched=True)
        data.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        self.train_data, self.val_data = data["train"], data["validation"]

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_data, batch_size=self.hparams.batch_size, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_data, batch_size=self.hparams.batch_size)


if __name__ == "__main__":
    dm = DataModule()
    dm.prepare_data()
    dm.setup()
    print(next(iter(dm.train_dataloader()))["input_ids"].shape)

