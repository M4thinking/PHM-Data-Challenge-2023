# Graficar PacMap
import os, torch, pytorch_lightning as pl
import numpy as np
from model3 import Autoencoder as AE
from dataset import PHM2023DataModule
from transforms import VariableLengthMFCC as MFCC, VariableLengthMelSpectrogram as VMS, LabelTransformer as LT, InverseLabelTransformer as ILT
from callbacks import StandarCallbacks as SC

def main(version, log_dir, name, device="cpu"):
    # DataModule
    data_params = {
            "batch_size": 32,
            "base_dir": "./data/",
            "tranform_dir": "./data_vms/",
            "force_transform": False,
            "transform": VMS(), # MFCC(), VMS()
            "label_transform": LT(),
            "supervised": True,
            "clustering": True,
        }
    
    # Cargar modelo
    ckpt_path = os.path.join(log_dir, name, f"version_{version}", "checkpoints")
    print(f"Loading model from {ckpt_path}")
    
    if os.path.exists(ckpt_path):
        ckpt_path = os.path.join(ckpt_path, os.listdir(ckpt_path)[0])
        print(f"Loading checkpoint from {ckpt_path}")
        model = AE.load_from_checkpoint(ckpt_path).to(device)
        latent_dim = model.hparams.latent_dim
    else:
        print("No checkpoint found")
        return
    
    # DataModule
    dm = PHM2023DataModule(**data_params)
    dm.setup()
    
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    
    # Get features 
    train_features = np.array([]) # (N, latent_dim)
    train_labels = np.array([]) # (N,)
    val_features = np.array([]) # (N, latent_dim)
    val_labels = np.array([]) # (N,)
     
    model.eval()
    itl = ILT()
    with torch.no_grad():
        for x, y in train_loader:
            y = itl(y).numpy()
            x = x.to(device)
            z = model.get_features(x)
            z = z.cpu().numpy()
            train_features = np.concatenate((train_features, z)) if train_features.size else z
            train_labels = np.concatenate((train_labels, y)) if train_labels.size else y
            
        for x, y in val_loader:
            y = y.numpy()
            x = x.to(device)
            z = model.get_features(x)
            z = z.cpu().numpy()
            val_features = np.concatenate((val_features, z)) if val_features.size else z
            val_labels = np.concatenate((val_labels, y)) if val_labels.size else y # Solo son -1
    # shapes
    print(train_features.shape, train_labels.shape, val_features.shape)    
    
    # # Graficar
    # import matplotlib.pyplot as plt
    # from matplotlib.colors import ListedColormap
    # import pacmap
    # # Setting n_neighbors to "None" leads to a default choice shown below in "parameter" section
    # embedding = pacmap.PaCMAP(n_components=2, n_neighbors=20, MN_ratio=0.5, FP_ratio=2.0) 

    # # fit the data (The index of transformed data corresponds to the index of the original data)
    # X_transformed = embedding.fit_transform(train_features, init="pca")
    # X_val_transformed = embedding.transform(val_features, basis=train_features)

    # # Definir los colores personalizados que deseas usar
    # custom_colors = [
    #     "#ff0000",  # Rojo anaranjado
    #     "#c46f00",  # Verde menta
    #     "#969c00",  # Azul violeta
    #     "#569c00",  # Rosa
    #     "#008c9c",  # Azul cielo
    #     "#004b9c",  # Amarillo limón
    #     "#3129d1",  # Rojo coral
    #     "#336DFF",  # Azul real
    #     "#6DFF33",  # Verde lima
    #     "#FFC633",  # Amarillo oro
    #     "#C633FF"   # Púrpura
    # ]

    # # Crear un nuevo mapa de colores personalizado
    # custom_cmap = ListedColormap(custom_colors)

    # # Armar un solo gráfico
    # fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    # sct = axs[0].scatter(X_transformed[:, 0], X_transformed[:, 1], cmap=custom_cmap, c=train_labels, s=0.6, alpha=0.8)
    # legend = axs[0].legend(*sct.legend_elements(), loc="best", title="Classes")
    # axs[0].set_title("PaCMAP - Training data")
    # axs[0].add_artist(legend)
    # sct = axs[1].scatter(X_val_transformed[:, 0], X_val_transformed[:, 1], cmap="Spectral", s=0.6, alpha=0.8)
    # axs[1].set_title("PaCMAP - Validation data")
    # plt.show()
    
    
    # # Si latent_dim = 2, graficar en 2D
    # if latent_dim == 2:
    #     # Graficar
    #     import matplotlib.pyplot as plt
    #     fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    #     sct = axs[0].scatter(train_features[:, 0], train_features[:, 1], cmap=custom_cmap, c=train_labels, s=0.6, alpha=0.8)
    #     legend = axs[0].legend(*sct.legend_elements(), loc="best", title="Classes")
    #     axs[0].set_title("Training data")
    #     axs[0].add_artist(legend)
    #     sct = axs[1].scatter(val_features[:, 0], val_features[:, 1], cmap="Spectral", s=0.6, alpha=0.8)
    #     axs[1].set_title("Validation data")
    #     plt.show()
    # else:
    #     print("latent_dim != 2, no se puede graficar")
    
    
    # Grafico y : features promedio por clase vs x : clase (para ver si las features son monótonas)
    # son latent_dim curvas
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 1, figsize=(12, 6))
    mean_by_class = np.zeros((7, latent_dim))
    lt = LT()
    for i in range(7):
        if i == 5 or i == 7 or i == 9 or i == 10:
            continue
        mean_by_class[lt(i)] = np.mean(train_features[train_labels == i], axis=0)


    # Eliminar las curvas que no son monótonas de las 64
    only_monotonic = []
    for i in range(latent_dim):
        if np.all(np.abs(np.diff(mean_by_class[:, i])) > 0.01):
            only_monotonic.append(i)
    mean_by_class = mean_by_class[:, only_monotonic]
    latent_dim = len(only_monotonic)

    for i in range(latent_dim):
        axs[0].plot(np.arange(7), mean_by_class[:, i], label=f"z{i}")
    axs[0].set_title("Mean features by class")
    axs[0].legend()
    axs[1].legend()
    plt.show()
    
    
if __name__ == "__main__":
    version = 0
    log_dir = "./test_logs/"
    name = "gear_vae_conv"
    main(version, log_dir, name)