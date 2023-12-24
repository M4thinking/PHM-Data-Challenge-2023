# Implementar callbacks de pytorch-lightning
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
# Path: scripts/phm/early_stopping.py


class StandarCallbacks:
    def __init__(
        self,
        monitor="val/loss",
        patience=10,
        mode="min",
        save_top_k=1,
        verbose=False,
    ):
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.save_top_k = save_top_k
        self.verbose = verbose

    def get_callbacks(self):
        # Early stopping
        early_stopping = EarlyStopping(
            monitor=self.monitor, patience=self.patience, mode=self.mode, verbose=self.verbose, check_on_train_epoch_end=True
        )
        # Guardar el mejor modelo
        checkpoint_callback = ModelCheckpoint(
            monitor=self.monitor,
            save_top_k=1,
            mode=self.mode,
            verbose=self.verbose,
            filename="{epoch}-{step}",
        )
        
        return [early_stopping, checkpoint_callback]
