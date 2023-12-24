import multiprocessing
import os
import sys
import torch
import numpy as np
import pytorch_lightning as pl
from torchmetrics.functional import accuracy
from torch.utils.data import DataLoader
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
torch.set_float32_matmul_precision("medium")  # O 'high' si lo prefieres

class BaseModel(pl.LightningModule):
    def __init(
        self,
        num_classes=1,
        name="",
        logs_dir="logs/",
        loss_fn=None,
        apply_accuracy=True,  # Nuevo parámetro
        hparams={}
    ):
        super().__init()
        self.name = name
        self.num_classes = num_classes
        self.logs_dir = logs_dir
        self.loss_fn = loss_fn
        self.apply_accuracy = apply_accuracy  # Nuevo atributo
        
        # Cambiar los selfs por selfs<
        
        # Crear folder para guardar los logs "name"
        os.makedirs(self.logs_dir + self.name, exist_ok=True)
        # Guardar hiperparámetros
        self.hparams.update(hparams)
        self.save_hyperparameters(ignore=['loss_fn'])

    def loss(self, x, y, **kwargs):
        """Computes the loss between x and y"""
        return self.loss_fn(x, y, **kwargs)

    def calculate_accuracy(self, y_hat, y):
        if self.apply_accuracy:
            return accuracy(y_hat, y, task="multiclass", num_classes=self.num_classes)
        else:
            return None

    def training_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"train/loss": loss}
        if self.acc is not None: metrics["train/acc"] = acc
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"val/loss": loss}
        if self.apply_accuracy: metrics["val/acc"] = acc
        self.log_dict(metrics, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"test/loss": loss}
        if acc is not None: metrics["test/acc"] = acc
        self.log_dict(metrics, on_epoch=True, prog_bar=True, logger=True)
        return metrics

    def _shared_eval_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        acc = self.calculate_accuracy(y_hat, y)
        return loss, acc

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_hat = self.forward(x)
        return y_hat

    def configure_optimizers(self):
        lr = self.hparams['lr'] if "lr" in self.hparams else 1e-3
        weight_decay = self.hparams['wd'] if "wd" in self.hparams else 1e-5
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

# Path: scripts/phm/utils.py
def get_data_loader(dataset, batch_size, num_workers=-1, pin_memory=True, **kwargs):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            pin_memory=pin_memory,
            num_workers=multiprocessing.cpu_count()
            if num_workers == -1
            else num_workers,
            **kwargs
        )