import torch
from datasets import load_dataset
from transformers import AutoTokenizer
import pytorch_lightning as pl


class DataModule(pl.LightningDataModule):
    def __init__(self, model_name="google/bert_uncased_L-2_H-128_A-2", batch_size=32):
        super().__init__()
        self.save_hyperparameters()  # ✅ 自动保存超参，便于 reproducibility 和 logging
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def prepare_data(self):
        self.dataset = load_dataset("glue", "cola")  # ✅ 合并加载逻辑，便于 cache 共享

    def setup(self, stage=None):
        def tokenize(example):
            return self.tokenizer(
                example["sentence"],
                truncation=True,
                padding="max_length",
                max_length=512,
            )

        tokenized = self.dataset.map(tokenize, batched=True)
        tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        self.train_data = tokenized["train"]
        self.val_data = tokenized["validation"]

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_data, batch_size=self.hparams.batch_size, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_data, batch_size=self.hparams.batch_size)


if __name__ == "__main__":
    dm = DataModule()
    dm.prepare_data()
    dm.setup()
    print(next(iter(dm.train_dataloader()))["input_ids"].shape)
