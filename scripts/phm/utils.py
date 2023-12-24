import multiprocessing, torch, os, sys, numpy as np
import pytorch_lightning as pl
from torchmetrics.functional import accuracy
from torch.utils.data import DataLoader
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
torch.set_float32_matmul_precision("medium")  # O 'high' si lo prefieres


class BaseModel(pl.LightningModule):
    def __init__(
        self,
        num_classes,
        name="tasks",
        logs_dir="logs/",
        loss_fn=None,
        hparams={}
    ):
        super().__init__()
        self.name = name
        self.num_classes = num_classes
        self.logs_dir = logs_dir
        self.loss_fn = loss_fn
        # Crear folder para guardar los logs "name"
        os.makedirs(self.logs_dir + self.name, exist_ok=True)
        # Guardar hiperparámetros
        self.hparams.update(hparams)
        self.save_hyperparameters(ignore=['loss_fn'])
    
    def loss(self, x, y, **kwargs):
        """Computes the loss between x and y"""
        return self.loss_fn(x, y, **kwargs)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        acc = accuracy(y_hat, y, task="multiclass", num_classes=self.num_classes)
        self.log_dict({"train/acc": acc, "train/loss": loss}, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"val/acc": acc, "val/loss": loss}
        self.log_dict(metrics, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"test/acc": acc, "test/loss": loss}
        self.log_dict(metrics, on_epoch=True, prog_bar=True, logger=True)
        return metrics

    def _shared_eval_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        acc = accuracy(y_hat, y, task="multiclass", num_classes=self.num_classes)
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
        
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import matplotlib.pyplot as plt

def plot_spectrogram(x):
    n_mels = 49
    fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True, sharey=True)
    cmap=cm.get_cmap('jet')
    normalizer=Normalize(0,1)
    im=cm.ScalarMappable(norm=normalizer, cmap=cmap)
    names = ['horizontal', 'axial', 'vertical', 'tachometer']
    for i in range(4):
        axs[i].imshow(x[i].flip(0), interpolation="bilinear", cmap='jet', aspect='auto', origin='lower', extent=[0, 1, 1, n_mels])
        axs[i].set_title(names[i].capitalize())
        axs[i].set_ylabel('Coeficiente Mel')
        axs[i].yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    plt.xlabel('Time (s)')
    fig.subplots_adjust(right=0.8)
    fig.colorbar(im, ax=axs.ravel().tolist(), label='Amplitud')
    plt.show()

def convertir_txt_a_pt(directorio_base):
    """Convertir archivos .txt a .pt

    Recorre todas las subcarpetas y archivos en el directorio base y convierte

    Args:
        directorio_base (str): Directorio base donde se encuentran los archivos .txt
        Ejemplo: "../data"
    """
    # Recorre todas las subcarpetas y archivos en el directorio base
    for directorio_actual, _, archivos in os.walk(directorio_base):
        print(directorio_actual)
        for archivo in archivos:
            print(archivo)
            if archivo.endswith(".txt"):
                # Construye la ruta completa del archivo .txt
                ruta_txt = os.path.join(directorio_actual, archivo)

                # Procesa y convierte el contenido del archivo .txt en un tensor PyTorch
                # Esto puede variar según la estructura de tus datos de series de tiempo
                tensor_pytorch = torch.from_numpy(
                    np.loadtxt(ruta_txt, dtype=np.float32)
                )

                # Reemplaza la extensión .txt con .pt en el nombre del archivo
                nombre_archivo_sin_extension = os.path.splitext(archivo)[0]
                nombre_archivo_pt = nombre_archivo_sin_extension + ".pt"

                # Construye la ruta completa del archivo .pt
                ruta_pt = os.path.join(directorio_actual, nombre_archivo_pt)

                # Guarda el tensor PyTorch en el archivo .pt
                torch.save(tensor_pytorch, ruta_pt)

                # Elimina el archivo .txt original si ya no es necesario
                os.remove(ruta_txt)