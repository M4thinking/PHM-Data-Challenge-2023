# Entrenar el modelo de PHM
import argparse, os
import pytorch_lightning as pl
from callbacks import StandarCallbacks as SC
from model3 import Autoencoder as AE
from dataset import PHM2023DataModule
from transforms import VariableLengthMFCC as MFCC, VariableLengthMelSpectrogram as VMS, LabelTransformer as LT, InverseLabelTransformer as ILT

def main(version, log_dir):
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
    model_params = {
            "learning_rate": 1e-3,
            "num_classes": 11,
            "input_size": [4, 64, 64],
            "hidden_size": 2,
            "output_size": [4, 64, 64],
            "encoder_hidden_sizes": [16, 185, 32, 243], # Al menos 4 capas
            "decoder_hidden_sizes": [16, 113, 139, 169],
            "latent_dim": 2,
            "noise": 0.01,
            "vae": True,
            "conv": True,
            "freeze_encoder": False,
            "dropout": 0.2,
            "wd": 0.0,
            "beta": 4.0,
        }
    
    # Conjunción de los true params de model_params
    name = "gear"
    for k, v in model_params.items():
        if v is True:
            name += f"_{k}"
        
    dm = PHM2023DataModule(**data_params)
    callbacks = SC(monitor="val/loss", patience=20, mode="min", verbose=True).get_callbacks()
    model = AE(hparams=model_params)
    # Summary
    model.summary()
    
    # ckpt path:
    ckpt_path = os.path.join(log_dir, f"version_{version}", "checkpoints")
    
    # Si existe un folder checkpoint, cargar archivo de extensión .ckpt del folder
    if os.path.exists(ckpt_path):
        ckpt_path = os.path.join(ckpt_path, os.listdir(ckpt_path)[0])
        print(f"Loading checkpoint from {ckpt_path}")
        model = AE.load_from_checkpoint(ckpt_path, hparams=model_params)
         
    # Trainer
    trainer = pl.Trainer(
        max_epochs=180,
        callbacks=callbacks,
        logger=pl.loggers.TensorBoardLogger(log_dir, name=name),
    )
    trainer.fit(model, dm, ckpt_path = ckpt_path if os.path.exists(ckpt_path) else None)
            
if __name__ == "__main__":
    version = 1
    log_dir = "./test_logs/"
    main(version, log_dir)
    
    # tensorboard --logdir ./test_logs/ --bind_all