import pytorch_lightning as pl
from pytorch_lightning.utilities.model_summary import ModelSummary
import glob
import os

from dbd.datasets.datasetLoader import get_dataloaders
from dbd.networks.model import Model


if __name__ == '__main__':
    ##########################################################
    checkpoint = "./lightning_logs/mobilenet_v3/checkpoints"
    dataset_root = "dataset/"

    ##########################################################
    checkpoint = glob.glob(os.path.join(checkpoint, "*.ckpt"))[-1]

    # Dataset
    dataloader_train, dataloader_val = get_dataloaders(dataset_root, num_workers=8)

    # Model
    # model = Model(lr=1e-4)
    model = Model.load_from_checkpoint(checkpoint, strict=True, lr=1e-4)

    summary = ModelSummary(model, max_depth=2)
    print(summary)

    valid = pl.Trainer(accelerator='gpu', devices=1, logger=False)
    valid.validate(model=model, dataloaders=dataloader_val)

    # Training
    # trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=1000, num_sanity_val_steps=0)
    # trainer.fit(model=model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)

    # tensorboard --logdir=lightning_logs/

