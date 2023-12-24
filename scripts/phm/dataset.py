import os
import re
import torch, numpy as np
from torch.utils.data import Dataset
from torch import Generator


class PHM2023Dataset(Dataset):
    def __init__(self, subset="train", transform=None, base_dir="../data/", tranform_dir="../data_mfcc/", force_transform=False, label_transform=None):
        # Force transform: si es True, se aplica la transformación y se guarda en tranform_dir
        # Si es False, se carga la transformación desde tranform_dir, asumiendo que ya existe, sino, se aplica la transformación
        self.subset = subset
        self.transform = transform
        self.label_transform = label_transform
        self.force_transform = force_transform
        
        # Directorios
        self.base_dir = base_dir
        self.root_dir = os.path.join(base_dir, subset)
        self.tranform_dir = tranform_dir
        
        # Cargar lista de archivos
        self.file_list = self._load_file_list()
        print("Path del script: ", os.getcwd(), " Path del root dir:", self.root_dir)
        if subset != "train":
            self.file_list = sorted(self.file_list, key=lambda x: int(re.search(r"^(\d+)_", os.path.split(x)[-1]).group(1)))

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
        degradation = self._extract_value(r"(\d+)", file_path) if "train" in file_path else -1

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

        # Apply transformation
        if self.transform:
            transform_file_path = file_path.replace(self.base_dir, self.tranform_dir)
            if os.path.exists(transform_file_path) and not self.force_transform:
                data = torch.load(transform_file_path)
            else:
                data = self.transform(data)
                torch.save(data, transform_file_path)

        # Apply label transformation 
        if self.label_transform:
            degradation = self.label_transform(degradation)

        return data, degradation
    
    def item(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_path = self.file_list[idx]
        data, degradation, velocity, torque, length = self._load_file(file_path)

        # Apply MFCC transformation
        if self.transform:
            data = self.transform(data)

        return data, degradation, velocity, torque, length, self.file_list[idx]
    
    def __str__(self) -> str:
        dim = self[0][0].shape
        return f"PHM2023Dataset({self.subset}, {len(self.file_list)} files, dim={np.array(dim)})"
    
    
class PHM2023DatasetUnsupervised(Dataset):
    # Obtiene los datasets (train, val, test) y los concatena
    def __init__(self, subsets=["train", "val", "test"], transform=None, base_dir="../data/", tranform_dir="../data_mfcc/", force_transform=False, label_transform=None):
        self.subsets = subsets
        self.transform = transform
        self.label_transform = label_transform
        self.force_transform = force_transform
    
        self.datasets = []
        for subset in subsets:
            dataset_params = {
                "subset": subset,
                "transform": transform,
                "base_dir": base_dir,
                "tranform_dir": tranform_dir,
                "force_transform": force_transform,
                "label_transform": label_transform
            }
            self.datasets.append(PHM2023Dataset(**dataset_params))
        
        self.lengths = [len(dataset) for dataset in self.datasets]
        self.cum_lengths = np.cumsum(self.lengths)
        self.total_length = sum(self.lengths)
        
    def __len__(self):
        return self.total_length
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # Obtener el dataset y el indice dentro del dataset
        dataset_idx = np.searchsorted(self.cum_lengths, idx, side="right")
        if dataset_idx > 0:
            idx = idx - self.cum_lengths[dataset_idx-1]
        return self.datasets[dataset_idx][idx]
        
    
    
# Data Module para PyTorch Lightning (que carga dataloaders de train, val y test)
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

class PHM2023DataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, base_dir="../data/", tranform_dir="../data_mfcc/", force_transform=False, transform=None, label_transform=None, supervised=False, clustering=False):
        super().__init__()
        self.gen = Generator().manual_seed(42)
        self.batch_size = batch_size
        self.base_dir = base_dir
        self.tranform_dir = tranform_dir
        self.force_transform = force_transform
        self.transform = transform
        self.label_transform = label_transform
        self.supervised = supervised
        self.clustering = clustering
        self.dataset_params = {
            "base_dir": self.base_dir,
            "tranform_dir": self.tranform_dir,
            "force_transform": self.force_transform,
            "transform": self.transform,
            "label_transform": self.label_transform
        }

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if self.supervised:
            if self.clustering:
                self.train_dataset = PHM2023Dataset(subset="train", **self.dataset_params)
                self.val_dataset = PHM2023Dataset(subset="val", **self.dataset_params)
                self.test_dataset = PHM2023Dataset(subset="test", **self.dataset_params)
            else:
                all_data = PHM2023Dataset(subset="train", **self.dataset_params)
                tlen, vlen = int(len(all_data)*0.8), len(all_data) - int(len(all_data)*0.8)
                self.train_dataset, self.val_dataset = random_split(all_data, [tlen, vlen])
                self.test_dataset = PHM2023Dataset(subset="val", **self.dataset_params)
        else:
            if stage == "fit" or stage is None:
                all_data = PHM2023DatasetUnsupervised(subsets=["train", "val"], **self.dataset_params)
                tlen, vlen = int(len(all_data)*0.8), len(all_data) - int(len(all_data)*0.8)
                self.train_dataset, self.val_dataset = random_split(all_data, [tlen, vlen])
                self.test_dataset = PHM2023DatasetUnsupervised(subsets=["test"], **self.dataset_params)
        
        print(f"Train: {len(self.train_dataset)} files, Val: {len(self.val_dataset)} files, Test: {len(self.test_dataset)} files")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=4, generator=self.gen, pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4, pin_memory=True, persistent_workers=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4, pin_memory=True, persistent_workers=True)

    def __str__(self) -> str:
        return f"PHM2023DataModule(batch_size={self.batch_size}, {self.dataset_params})"


# Test datamodule
if __name__ == "__main__":
    from transforms import VariableLengthMFCC as MFCC, VariableLengthMelSpectrogram as VMS, LabelTransformer as LT, InverseLabelTransformer as ILT
    # DataModule
    dataset_params = {
            "base_dir": "./data/",
            "tranform_dir": "./data_vms/",
            "force_transform": False,
            "transform": VMS(), # MFCC(), VMS()
            "label_transform": LT(),
            "supervised": False
        }
    dm = PHM2023DataModule(batch_size=32, **dataset_params)
    dm.setup()
    print(dm)
    
    # Mostrar un dato desde un dataloader
    dl = dm.train_dataloader()
    x, y = next(iter(dl))
    x, y = x[0], y[0]
    print(x.shape, y)
    
    # Graficar cada canal
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(4, 1, figsize=(4, 8), gridspec_kw={"height_ratios": [1, 1, 1, 1], "hspace": 0.2, "top": 0.9, "bottom": 0.1, "left": 0.1, "right": 0.9})
    vmin, vmax = torch.min(x), torch.max(x)
    for i in range(4):
        axs[i].imshow(x[i], aspect="auto", extent=[0, x.shape[1], 0, x.shape[2]], vmin=vmin, vmax=vmax, interpolation="bilinear", cmap="jet")
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap="jet", norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axs, shrink=0.5)
    plt.tight_layout()
    plt.show()
