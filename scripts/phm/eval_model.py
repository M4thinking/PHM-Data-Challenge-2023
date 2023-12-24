import argparse, os
import torch
import pytorch_lightning as pl
from callbacks import StandarCallbacks as SC
from model3 import Autoencoder as AE
from dataset import PHM2023DataModule
from transforms import VariableLengthMFCC as MFCC, VariableLengthMelSpectrogram as VMS, LabelTransformer as LT, InverseLabelTransformer as ILT

def main(version, log_dir, name, example, device="cpu"):
    # DataModule
    data_params = {
            "batch_size": 32,
            "base_dir": "./data/",
            "tranform_dir": "./data_vms/",
            "force_transform": False,
            "transform": VMS(), # MFCC(), VMS()
            "label_transform": LT(),
            "supervised": False,
            "clustering": False,
        }
    
    # Cargar modelo
    ckpt_path = os.path.join(log_dir, name, f"version_{version}", "checkpoints")
    print(f"Loading model from {ckpt_path}")
    
    if os.path.exists(ckpt_path):
        ckpt_path = os.path.join(ckpt_path, os.listdir(ckpt_path)[0])
        print(f"Loading checkpoint from {ckpt_path}")
        model = AE.load_from_checkpoint(ckpt_path).to(device)
    else:
        print("No checkpoint found")
        return
    
    # DataModule
    dm = PHM2023DataModule(**data_params)
    dm.setup()
    
    # Graficar
    test = dm.test_dataset[example][0].unsqueeze(0).to(device)
    
    # Reconstrucción
    model.eval()
    with torch.no_grad():
        x_recon = model(test)[0][0]
    
    print(torch.min(x_recon), torch.max(x_recon))
    # Visualizar
    names = ["Horizontal", "Axial", "Vertical", "Tacómetro"]
    
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(4, 2, figsize=(8, 8), gridspec_kw={"height_ratios": [1, 1, 1, 1], "hspace": 0.2, "top": 0.9, "bottom": 0.1, "left": 0.1, "right": 0.9}, sharex=True, sharey=True)
    vmin, vmax = torch.min(test), torch.max(test)
    for i in range(4):
        axs[i][0].imshow(test[0][i], aspect="auto", extent=[0, test.shape[2], 0, test.shape[3]], vmin=vmin, vmax=vmax, interpolation="bilinear", cmap="jet")
        axs[i][1].imshow(x_recon[i], aspect="auto", extent=[0, x_recon.shape[1], 0, x_recon.shape[2]], vmin=vmin, vmax=vmax, interpolation="bilinear", cmap="jet")
        # titulos con .text 1 sola vez en cada fila
        axs[i][0].text(1.1, 1.1, names[i], horizontalalignment='center', verticalalignment='center', transform=axs[i][0].transAxes, fontsize=12)

        # Agregar eje y
        axs[i][0].set_ylabel("Amplitud")
        
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap="jet", norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axs, shrink=0.5, label="Amplitud")
    plt.tight_layout()
    # plt.sup_title(f"Reconstrucción de ejemplo {example}")
    # Suptitle
    plt.suptitle(f"Reconstrucción de ejemplo {example} del conjunto de test", fontsize=14)
    
    axs[3][0].set_xlabel("Tiempo [s]")
    axs[3][1].set_xlabel("Tiempo [s]")
    plt.savefig(f"./recon_example_{example}.png", dpi=300)
    plt.show()
    
    # Calcular MSE, MAE, MAPE en conjuntos de entrenamiento, validación y test
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()
    
    def metrics(y, y_hat):
        mse = torch.mean((y - y_hat)**2)
        mae = torch.mean(torch.abs(y - y_hat))
        mape = torch.mean(torch.abs((y - y_hat) / y))
        return mse, mae, mape
    
    # Entrenamiento
    model.eval()
    y = torch.empty(0)
    y_hat = torch.empty(0)
    with torch.no_grad():
        for x, _ in train_loader:
            x = x.to(device)
            x_hat = model(x)[0]
            y = torch.cat([y, x.flatten()])
            y_hat = torch.cat([y_hat, x_hat.flatten()])
            
    mse, mae, mape = metrics(y, y_hat)
    print(f"Train MSE: {mse}, MAE: {mae}, MAPE: {mape}")
    
    # Validación
    model.eval()
    y = torch.empty(0)  
    y_hat = torch.empty(0)
    with torch.no_grad():
        for x, _ in val_loader:
            x = x.to(device)
            x_hat = model(x)[0]
            y = torch.cat([y, x.flatten()])
            y_hat = torch.cat([y_hat, x_hat.flatten()])
            
    mse, mae, mape = metrics(y, y_hat)
    
    print(f"Val MSE: {mse}, MAE: {mae}, MAPE: {mape}")
    
    # Test
    model.eval()
    y = torch.empty(0)
    y_hat = torch.empty(0)
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            x_hat = model(x)[0]
            y = torch.cat([y, x.flatten()])
            y_hat = torch.cat([y_hat, x_hat.flatten()])
            
    mse, mae, mape = metrics(y, y_hat)
    
    print(f"Test MSE: {mse}, MAE: {mae}, MAPE: {mape}")
    
if __name__ == "__main__":
    version = 1
    log_dir = "./test_logs/"
    name = "gear_vae_conv"
    example = 600
    main(version, log_dir, name, example)    
    