import torch
import pytorch_lightning as pl
from data import DataModule
from model import ColaModel
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

def main():
    trainer = pl.Trainer(
        default_root_dir="logs",
        gpus=int(torch.cuda.is_available()),
        max_epochs=5,
        logger=TensorBoardLogger("logs", name="cola", version=1),
        callbacks=[
            ModelCheckpoint(dirpath="./models", monitor="val_loss", mode="min"),
            EarlyStopping(monitor="val_loss", patience=3, verbose=True, mode="min")
        ],
    )
    trainer.fit(ColaModel(), DataModule())

if __name__ == "__main__":
    main()
