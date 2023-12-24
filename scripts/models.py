import os, re, time, torch
from torch.utils.data import Dataset
import torchaudio.transforms as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl


class AbstractModel(nn.Module):
    def __init__(self, learning_rate=1e-3, batch_size=64, n_epochs=200):
        super(AbstractModel, self).__init__()
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None  # Se debe definir en cada modelo

    def forward(self, x):
        # lo debe implementar cada modelo
        pass

    def fit(self, train_loader, val_loader, name="best_model"):
        best_val_loss = float("inf")
        best_model_state_dict = None
        early_stopping_counter = 0
        training_curves = {
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
        }

        # Tiempo inicial
        t0 = time.perf_counter()

        for epoch in range(self.n_epochs):
            self.train()
            total_loss = 0.0
            correct_train = 0
            total_train = 0

            for x_batch, y_batch in train_loader:
                self.optimizer.zero_grad()

                # Propagación hacia adelante
                outputs = self(x_batch)

                # Cálculo de pérdida
                loss = self.criterion(outputs, y_batch)
                total_loss += loss.item()

                # Retropropagación y actualización de pesos
                loss.backward()
                self.optimizer.step()

                # Cálculo de precisión
                _, predicted = torch.max(outputs.data, 1)
                total_train += y_batch.size(0)
                correct_train += (predicted == y_batch).sum().item()

            train_loss = total_loss / len(train_loader)
            train_accuracy = correct_train / total_train

            # Validación
            self.eval()
            with torch.no_grad():
                total_val_loss = 0.0
                correct_val = 0
                total_val = 0

                for x_val, y_val in val_loader:
                    outputs = self(x_val)
                    val_loss = self.criterion(outputs, y_val)
                    total_val_loss += val_loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    total_val += y_val.size(0)
                    correct_val += (predicted == y_val).sum().item()

                val_loss = total_val_loss / len(val_loader)
                val_accuracy = correct_val / total_val

            training_curves["train_loss"].append(train_loss)
            training_curves["val_loss"].append(val_loss)
            training_curves["train_accuracy"].append(train_accuracy)
            training_curves["val_accuracy"].append(val_accuracy)

            print(
                f"\r Epoch {epoch + 1}/{self.n_epochs} | TL: {train_loss:.4f} | VL: {val_loss:.4f} | TA: {train_accuracy:.4f} | VA: {val_accuracy:.4f} ",
                end="",
            )

            # Guardar el mejor modelo y early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state_dict = self.state_dict()
                torch.save(best_model_state_dict, f"{name}.pt")
                early_stopping_counter = 0
            elif early_stopping_counter >= 10:
                print("Early Stopping")
                print(
                    f"Tiempo total de entrenamiento: {time.perf_counter() - t0:.4f} [s]"
                )
                break
            else:
                early_stopping_counter += 1

        # Cargar el mejor modelo antes de finalizar
        if best_model_state_dict is not None:
            self.load_state_dict(best_model_state_dict)

        # Guardar historial de curvas
        torch.save(training_curves, f"training_curves_{name}.pt")

        return training_curves

    def evaluate(self, x, y):
        self.eval()
        with torch.no_grad():
            x = torch.tensor(x, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.long)
            outputs = self(x)
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == y).sum().item() / y.size(0)
        return accuracy


class AbstractLightningModel(pl.LightningModule):
    def __init__(self, learning_rate, n_epochs):
        super(AbstractLightningModel, self).__init__()
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()

    def configure_optimizers(self):
        self.optimizer = optim.Adam(
            self.parameters(), lr=self.learning_rate, betas=(0.9, 0.999)
        )
        return self.optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = self.criterion(outputs, y)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = self.criterion(outputs, y)
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def fit(self, train_loader, val_loader, name="best_model"):
        trainer = pl.Trainer(
            max_epochs=self.n_epochs,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            callbacks=[
                pl.callbacks.EarlyStopping(monitor="val_loss", patience=10, mode="min"),
                pl.callbacks.ModelCheckpoint(
                    filename=name, monitor="val_loss", save_top_k=1, mode="min"
                ),
                # Tener copia del avance cada 10 epochs
                pl.callbacks.ModelCheckpoint(
                    filename=f"{name}_epoch{self.n_epochs}",
                    every_n_epochs=10,  # Guardar cada 10 epochs
                    monitor="val_loss",
                    mode="min",
                ),
            ],
            logger=pl.loggers.TensorBoardLogger("lightning_logs/", name=name),
            log_every_n_steps=1,
        )

        trainer.fit(self, train_loader, val_loader)


class PHM2023Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_list = self._load_file_list()

    def _load_file_list(self):
        file_list = []
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith(".pt"):
                    file_list.append(os.path.join(root, file))
        return file_list

    # forma generica de extraer un valor de una ruta
    def _extract_value(self, pattern, path):
        match = re.search(pattern, path)
        if match:
            return int(match.group(1))
        return -1

    def _load_file(self, file_path):
        # Serie de tiempo para cada instante de tiempo de la forma "n1 n2 n3 n4"
        # -7.554961414e-002 3.061763576e-001 -2.260650605e-001 0.000000000e+000
        data = torch.load(file_path)

        # Extract degradation level from the file path
        degradation = self._extract_value(r"(\d+)", file_path)

        # Extract velocity and torque from the file path
        velocity = self._extract_value(r"V(\d+)_", file_path)
        torque = self._extract_value(r"_(\d+)N", file_path)

        return data, degradation, velocity, torque, len(data)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_path = self.file_list[idx]
        data, degradation, velocity, torque, length = self._load_file(file_path)

        # Apply MFCC transformation
        if self.transform:
            data = self.transform(data)

        return data, degradation


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Block, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.seq(x)


class GearModel(AbstractLightningModel):
    def __init__(
        self,
        input_shape=[4, 13, 481],  # [channels, dim1, dim2]
        learning_rate=1e-3,
        # batch_size=64,
        n_epochs=200,
        n_classes=7,
    ):
        super(GearModel, self).__init__(learning_rate, n_epochs)

        self.n_classes = n_classes
        self.input_shape = input_shape

        self.seq1 = nn.Sequential(
            Block(input_shape[0], 32, 3, 1, 1),
            Block(32, 128, 3, 1, 1),
            Block(128, 64, 3, 1, 1),
        )

        # Calculate the output shape of seq1
        _, channels, dim1, dim2 = self.seq1(torch.randn(1, *input_shape)).shape
        print(f"Output shape of CNN: [{channels}, {dim1}, {dim2}]")

        self.seq2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * dim1 * dim2, 128),
        )

        self.seq3 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, n_classes),
        )

        self.optimizer = optim.Adam(
            self.parameters(), lr=learning_rate, betas=(0.9, 0.999)
        )

    def forward(self, x):
        x = self.seq1(x)
        x = self.seq2(x)
        x = self.seq3(x)
        return x

    def get_features(self, x):
        x = self.seq1(x)
        x = self.seq2(x).detach().cpu().numpy()
        return x


class VariableLengthMFCC:
    def __init__(self):
        self.freq = 20480
        for k in range(3):
            setattr(
                self,
                f"t{3 * 2**k}s",
                T.MFCC(
                    sample_rate=self.freq,
                    n_mfcc=13,
                    melkwargs={
                        "n_fft": self.freq // int(2 ** (3 - k) * 10),
                        "hop_length": self.freq // int(2 ** (3 - k) * 20),
                        "n_mels": 40,  # mels son los filtros de mel scale que se aplican a la señal
                    },
                ),
            )

    def __call__(self, signal):
        n = signal.shape[0]  # Calcular la duración en muestras
        for k in range(2, -1, -1):
            time = 3 * 2**k
            m = self.freq * time  # Calcular la duración en muestras standard
            if n >= m:
                # print(f"Usando t{time}s con {m} muestras")
                # print(f"n_fft = {self.freq // (2**(3-k) * 10)} y hop_length = {self.freq // (2**(3-k) * 20)}")
                return torch.stack(
                    [
                        getattr(self, f"t{time}s")(signal[:m, i])
                        for i in range(signal.shape[1])
                    ]
                )
        raise ValueError(f"La señal es muy corta para aplicar MFCC: {signal.shape}") 
