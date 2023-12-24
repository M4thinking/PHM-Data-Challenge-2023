import torch, os
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.functional import accuracy
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms


class ClassificationTask(pl.LightningModule):
    def __init__(
        self, model, logs_dir="lightning_logs/", name="default", num_classes=10
    ):
        super().__init__()
        self.model = model
        self.logs_dir = logs_dir
        self.name = name
        self.num_classes = num_classes

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        acc = accuracy(y_hat, y, task="multiclass", num_classes=self.num_classes)
        metrics = {"train_acc": acc, "train_loss": loss}
        self.log_dict(metrics)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"val_acc": acc, "val_loss": loss}
        self.log_dict(metrics, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"test_acc": acc, "test_loss": loss}
        self.log_dict(metrics)
        return metrics

    def _shared_eval_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        acc = accuracy(y_hat, y, task="multiclass", num_classes=self.num_classes)
        return loss, acc

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_hat = self.model(x)
        return y_hat

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.001, betas=(0.9, 0.999))

    def get_trainer(
        self, train_loader, val_loader, curve_name, callbacks=None, max_epochs=10
    ):
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            devices=1,
            accelerator="gpu",
            progress_bar_refresh_rate=20,
            logger=pl.loggers.TensorBoardLogger(self.logs_dir, name=self.name),
            checkpoint_callback=False,
        )
        return trainer

    def fit(self, trainer, train_loader, val_loader, max_epochs=10):
        trainer.fit(self, train_loader, val_loader)


if __name__ == "__main__":

    class MNISTSimpleModel(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.l1 = torch.nn.Linear(28 * 28, 10)

        def forward(self, x):
            return torch.relu(self.l1(x.view(x.size(0), -1)))

    model_name = "mnist_model.pt"
    model = MNISTSimpleModel()
    # si no existe el modelo, lo entrena y lo guarda
    if not os.path.exists(model_name):
        mnist_train = MNIST(
            "./mnist", train=True, download=True, transform=transforms.ToTensor()
        )
        mnist_test = MNIST(
            "./mnist", train=False, download=True, transform=transforms.ToTensor()
        )
        train_loader = DataLoader(mnist_train, batch_size=32, num_workers=4)
        test_loader = DataLoader(mnist_test, batch_size=32, num_workers=4)

        task = ClassificationTask(model)
        task.fit(train_loader, None, test_loader, max_epochs=10)

        torch.save(model.state_dict(), "mnist_model.pt")
        print("Done!")

    else:
        print("Model already exists!")
        model.load_state_dict(torch.load("mnist_model.pt"))
        model.eval()
        mnist_test = MNIST(
            "./mnist", train=False, download=True, transform=transforms.ToTensor()
        )
        test_loader = DataLoader(mnist_test, batch_size=32, num_workers=4)
        task = ClassificationTask(model)
        # Obtención de métricas
        trainer = task.validate(test_loader)
        print(trainer)
