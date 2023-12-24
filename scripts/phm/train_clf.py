# Entrenar el modelo de PHM
import argparse, os, torch
import pytorch_lightning as pl
from callbacks import StandarCallbacks as SC
from model3 import Classifier as CLF, Autoencoder as AE, ClassifierRegressor as CR, ConvEncoder as CE, ConvDecoder as CD
from dataset import PHM2023DataModule
from transforms import VariableLengthMFCC as MFCC, VariableLengthMelSpectrogram as VMS, LabelTransformer as LT, InverseLabelTransformer as ILT

def main(version_vae, version_cr, log_dir, name, ae_name = "gear_vae_conv", device="cuda"):
    # DataModule
    clf_params = {
            "learning_rate": 1e-3,
            "num_classes": 11,
            "output_size": 11,
            "input_size": [4, 64, 64],
            "hidden_size": 64,
            "hidden_sizes": [64, 32, 16],
            "freeze_encoder": False, # Importante
            "dropout": 0.2,
            "wd": 0.0,
        }
    
    # Armar clasificador
    ae_name = "gear_vae_conv"
    
    # DataModule
    data_params = {
            "batch_size": 32,
            "base_dir": "./data/",
            "tranform_dir": "./data_vms/",
            "force_transform": False,
            "transform": VMS(), # MFCC(), VMS()
            "label_transform": None,
            "supervised": False,
            "clustering": False,
        }
    
    # Cargar modelo de encoder con pesos
    ckpt_path = os.path.join(log_dir, ae_name, f"version_{version_vae}", "checkpoints")
    print(f"Loading model from {ckpt_path}")
    
    if os.path.exists(ckpt_path):
        ckpt_path = os.path.join(ckpt_path, os.listdir(ckpt_path)[0])
        print(f"Loading checkpoint from {ckpt_path}")
        vae = AE.load_from_checkpoint(ckpt_path).to("cpu")
    else:
        print("No checkpoint found")
        return

    dm = PHM2023DataModule(**data_params)
    callbacks = SC(monitor="val/loss", patience=50, mode="min", verbose=True).get_callbacks()
    model = CR(encoder=vae.encoder, hparams=clf_params)
    
    
    # Cargar pesos si existen
    ckpt_path = os.path.join(log_dir, name, f"version_{version_cr}", "checkpoints")
    if os.path.exists(ckpt_path):
        ckpt_path = os.path.join(ckpt_path, os.listdir(ckpt_path)[0])
        print(f"Loading checkpoint from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        model = CR(encoder=CE( **{"input_size": [4, 64, 64], "hidden_sizes": [16, 185, 32, 243], "output_size": 64}), hparams = checkpoint["hyper_parameters"] ).to(device) 
        model.load_state_dict(checkpoint["state_dict"])
    else:
        print("No checkpoint found")
        return
    
    
    # Entrenar
    trainer = pl.Trainer(
        max_epochs=200,
        callbacks=callbacks,
        logger=pl.loggers.TensorBoardLogger(log_dir, name=name),
    )
    
    trainer.fit(model, dm)
    
if __name__ == "__main__":
    version_vae = 0
    version_cr = 30
    log_dir = "./test_logs/"
    main(version_vae, version_cr, log_dir, name = "gear_cr_conv", device="cuda")