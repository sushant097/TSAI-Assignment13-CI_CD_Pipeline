from typing import Any, Dict, Optional, Tuple

import os
import subprocess
import torch
import timm
import json

import pytorch_lightning as pl
import torchvision.transforms as T
import torch.nn.functional as F

from pathlib import Path
from torchvision.datasets import ImageFolder
from pytorch_lightning.plugins.environments import LightningEnvironment
from torch.utils.data import DataLoader, Dataset
from torchmetrics.functional import accuracy
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import TQDMProgressBar
from datetime import datetime

from model import LitResnet
from dataset import IntelClassificationDataModule

sm_output_dir = Path(os.environ.get("SM_OUTPUT_DIR"))
sm_model_dir = Path(os.environ.get("SM_MODEL_DIR"))
num_cpus = int(os.environ.get("SM_NUM_CPUS"))

train_channel = os.environ.get("SM_CHANNEL_TRAIN")
test_channel = os.environ.get("SM_CHANNEL_TEST")

ml_root = Path("/opt/ml")


def get_training_env():
    sm_training_env = os.environ.get("SM_TRAINING_ENV")
    sm_training_env = json.loads(sm_training_env)
    
    return sm_training_env

def train_and_evaluate(model, datamodule, sm_training_env, output_dir):
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=ml_root / "output" / "tensorboard" / sm_training_env["job_name"])
    
    trainer = pl.Trainer(
        max_epochs=5,
        accelerator="auto",
        logger=[tb_logger],
        callbacks=[TQDMProgressBar(refresh_rate=10)]
    )
    
    trainer.fit(model, datamodule)
    # test on the test dataset
    # calculating evaluation metrics
    trainer.test(model, datamodule)

    idx_to_class = {k: v for v,k in datamodule.data_train.class_to_idx.items()}
    model.idx_to_class = idx_to_class

    # calculating per class accuracy
    nb_classes = datamodule.num_classes

    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    acc_all = 0
    with torch.no_grad():
        for i, (images, targets) in enumerate(datamodule.test_dataloader()):
            # images = images.to(device)
            # targets = targets.to(device)
            outputs = model(images)
            acc_all += (outputs == targets).sum()
            _, preds = torch.max(outputs, 1)
            for t, p in zip(targets.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
    """
    Simple Logic may be useful:
    acc = [0 for c in list_of_classes]
    for c in list_of_classes:
        acc[c] = ((preds == labels) * (labels == c)).float() / (max(labels == c).sum(), 1))
    """
    
    acc_all = acc_all / len(datamodule.test_dataloader())

    accuracy_per_class = {
        idx_to_class[idx]: val.item() * 100 for idx, val in enumerate(confusion_matrix.diag() / confusion_matrix.sum(1))
    }
    print(accuracy_per_class)
    print(acc_all)
    
    report_dict = {
        "multiclass_classification_metrics": {
            "accuracy": acc_all,
            "accuracy_per_class": accuracy_per_class
        },
    }
    
    with open(output_dir / "evaluation_metrics.json", "w") as f:
        json.dump(report_dict, f)

    return trainer

def train(model, datamodule, sm_training_env):
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=ml_root / "output" / "tensorboard" / sm_training_env["job_name"])
    
    trainer = pl.Trainer(
        max_epochs=5,
        accelerator="auto",
        logger=[tb_logger]
    )
    
    trainer.fit(model, datamodule)
    
    return trainer

def save_scripted_model(model, output_dir):
    script = model.to_torchscript()

    # save for use in production environment
    torch.jit.save(script, output_dir / "model.scripted.pt")


def save_last_ckpt(trainer, output_dir):
    trainer.save_checkpoint(output_dir / "last.ckpt")


if __name__ == '__main__':
    
    img_dset = ImageFolder(train_channel)
    
    print(":: Classnames: ", img_dset.classes)
    
    datamodule = IntelClassificationDataModule(train_data_dir=train_channel, test_data_dir=test_channel, num_workers=num_cpus)
    datamodule.setup()
    
    model = LitResnet(num_classes=datamodule.num_classes)
    
    sm_training_env = get_training_env()
    
    print(":: Training ...")
    trainer = train(model, datamodule, sm_training_env)

    print(":: Saving Model Ckpt")
    save_last_ckpt(trainer, sm_model_dir)
    
    print(":: Saving Scripted Model")
    save_scripted_model(model, sm_model_dir)

