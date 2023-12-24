import torch
import torch.nn as nn
from utils2 import BaseModel

# Modelo del autoencoder
class AutoEncoder(nn.Module):
    def __init__(self, shape, input_size, output_size=[4, 128, 128]):
        super().__init__()
        # self.output_size = output_size
        # Transformacion de la dimension de entrada luego de cada capa
        self.dim = [input_size]
        
        # Generamos los tamaños
        self.deep     = shape[0]
        self.k_size   = shape[1]
        self.stride   = shape[2]
        self.padding  = shape[3] 
        self.pooling  = shape[4]
    
        try: assert(len(self.deep)-1 == len(self.k_size) == len(self.stride) == len(self.padding) == len(self.pooling))
        except: print("[ERROR]: Tamaño de los parametros no coinciden")
    
        # Generamos las capas
        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()
    
        # Generamos las capas de encoder
        for i in range(len(self.deep)-1):
            self.encoder.add_module(
                "conv"+str(i+1),
                nn.Conv2d(
                    self.deep[i],
                    self.deep[i+1],
                    self.k_size[i],
                    self.stride[i],
                    self.padding[i]
                )
            )
            self.encoder.add_module(
                "relu"+str(i+1),
                nn.ReLU()
            )
            self.encoder.add_module(
                  "batch_norm"+str(i+1),
                  nn.BatchNorm2d(self.deep[i+1])
            )
            self.dim.append(
                int(
                    (self.dim[i] - self.k_size[i] + 2*self.padding[i])//self.stride[i] + 1
                )
            )
            if self.pooling[i] != 0:
                self.encoder.add_module(
                    "max_pooling"+str(i+1),
                    nn.MaxPool2d(self.pooling[i], self.pooling[i])
                )
                self.dim[i+1] = self.dim[i]//self.pooling[i]
    
        # Capa densa hecha con capas convolucionales
        self.encoder.add_module(
            "conv"+str(len(self.deep)),
            nn.Conv2d(
                self.deep[-1],
                self.deep[-1],
                self.dim[-1]
            )
        )
        # Activacion
        self.encoder.add_module(
            "reluLinear",
            nn.ReLU()
        )
    
        # Generamos las capas de decoder con indice inverso
        for i in range(len(self.deep)-1, 0, -1):
            if self.pooling[i-1] != 0:
                self.decoder.add_module(
                    "Upsample"+str(i+1),
                    nn.Upsample(scale_factor=self.pooling[i-1])
                )
                  
                # Si la dimension anterior es impar se
                # agrega padding replicando el ultimo pixel
                if self.dim[i-1]%2 != 0:
                    self.decoder.add_module(
                      "padding"+str(i+1),
                      nn.ReplicationPad2d((0,1,0,1))
                    )
            self.decoder.add_module(
                "conv_t"+str(i+1),
                nn.ConvTranspose2d(
                    self.deep[i],
                    self.deep[i-1],
                    self.k_size[i-1],
                    self.stride[i-1],
                    self.padding[i-1],
                )
            )
            self.decoder.add_module(
                "relu"+str(i+1),
                nn.ReLU()
            )
          
        # Con una distribución conveniente
        self.init_weights()
    
    # Inicializar los pesos
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)    
            

    # Summary del modelo
    def summary(self):
        print("Encoder: ", self.encoder)
        print("Decoder: ", self.decoder)

    def get_features(self,x):
        # Obtenter las features de la forma [batch_size, 1, features]
        return self.encoder(x).view(x.shape[0], 1, -1)

    def encode(self,x):
        # Obtener el vector de features
        return self.encoder(x)

    def decode(self,x):
        # Decodificar el vector de features
        return self.decoder(x)

    def forward(self,x):
        x = self.encode(x)
        x = self.decode(x)
        return x

# Clase VAE que recibe un modelo AutoEncoder y aplicamos el reparametrization trick
class VAE(BaseModel):
    def __init__(self, model, latent_dim, hparams={}):
        super().__init__()
        self.hparams.update(hparams)
        self.save_hyperparameters(self.hparams)
        self.encoder = model.encoder
        self.decoder = model.decoder
        self.loss_fn = nn.MSELoss()
        self.hidden_size = hparams["hidden_size"]
        self.fc_mu = nn.Linear(self.hidden_size, latent_dim)
        self.fc_var = nn.Linear(self.hidden_size, latent_dim)
        self.fc_z = nn.Linear(latent_dim, self.hidden_size)
        self.latent_dim = latent_dim
        # Si kl_weight != 1, se aplica beta-VAE
        self.kl_weight = hparams["kl_weight"] if "kl_weight" in hparams else 1
        self.beta = hparams["beta"] if "beta" in hparams else 1

    def loss(self, x, x_hat, mu, log_var, train=True, return_aux=False):
        # Reconstruction loss
        recon_loss = self.loss_fn(x_hat, x)
        # KL divergence
        kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1))
        # Total loss
        loss = recon_loss + self.kl_weight * kl_loss
        
        # Update kl_weight
        self.log("kl_weight", self.kl_weight, on_step=False, on_epoch=True)
        self.kl_weight = self.kl_weights[self.trainer.current_epoch]
        if train:
            self.log("train/loss", loss, prog_bar=True)
            self.log("train/reconstruction_loss", recon_loss,  prog_bar=True)
            self.log("train/kl_loss", kl_loss,  prog_bar=True)
        else:
            self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log("val/reconstruction_loss", recon_loss, on_step=False, on_epoch=True)
            self.log("val/kl_loss", kl_loss, on_step=False, on_epoch=True)
        if return_aux:
            return loss, recon_loss, kl_loss
        return loss
    
    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def encode(self, x):
        x = self.encoder(x) # (B, 15, 1, 1) -> (B, 15)
        x = x.view(x.shape[0], 1, -1)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var
    
    def decode(self, z):
        # (B, 15) -> (B, 15, 1, 1) (No 10, 1, 1, 15)
        z = z.view(z.shape[0], -1, 1, 1)
        x_hat = self.decoder(z)
        return x_hat
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparametrize(mu, log_var)
        z = self.fc_z(z)
        z = torch.relu(z)
        x_hat = self.decode(z)
        # Output shape: (B, 4, 128, 128)
        # x_hat = x_hat.view(x_hat.shape[0], -1, self.output_size[1], self.output_size[2])
        return x_hat, mu, log_var

    def training_step(self, batch, batch_idx):
        """ "
        Called for every batch. Computes the loss and performs backpropagation and optimization automatically.
        """
        x, _ = batch
        x_hat, mu, log_var = self.forward(x)
        loss = self.loss(x, x_hat, mu, log_var, train=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_hat, mu, log_var = self.forward(x)
        loss = self.loss(x, x_hat, mu, log_var, train=False, return_aux=False)
        return loss

    def configure_optimizers(self):
        lr = self.hparams["lr"] if "lr" in self.hparams else 1e-3

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        # use step
        if self.hparams["scheduler"] == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50)
            return [optimizer], [scheduler]
        
        return [optimizer]
    
    def get_features(self, x):
        # Obtenter las features de la forma [batch_size, 1, features]
        return self.encoder(x) 
    
    def sample(self, n):
        """Sample n images from the decoder"""
        z = torch.randn(n, self.latent_dim, device=self.device)
        return self.decode(z)
    
    def summary(self):
        print("Encoder: ", self.encoder)
        print("Decoder: ", self.decoder)
        print("FC_mu: ", self.fc_mu)
        print("FC_var: ", self.fc_var)
        
    def on_fit_start(self) -> None:
        print("Initializing kl_weights...")
        # initialize kl_weights when fit starts and we know the max_epochs
        # you can tweak this if you want
        if self.beta == 1: # kl_weight lineal desde [0, kl_weight]
            self.kl_weights = torch.linspace(
                0, self.kl_weight, self.trainer.estimated_stepping_batches + 1
            )
        else: # kl_weight constante
            self.kl_weights = (
                torch.ones(self.trainer.estimated_stepping_batches) * self.kl_weight
            )

        self.kl_weight = self.kl_weights[0]
         
    

if __name__ == "__main__":
    # Estructura del modelo 128x128
    # Shape: Profundidad (n), kernel size (n-1), stride (n-1), padding (n-1), pooling (n-1)
    latent_dim = 15 # Features
    in_size = 128
    hidden_size = 128
    cin = 4 # Canales de entrada

    shape = [   
            [ cin, 64, 64, 64, 64, 64, 64, hidden_size], # Mapas de activación
                [ 3,  3,  3,  3,  3,  3,  3], # Kernel size
                [ 1,  1,  1,  1,  1,  1,  1], # Stride
                [ 1,  1,  1,  1,  1,  1,  1], # Padding
                [ 2,  2,  2,  2,  2,  2,  2], # Pooling
            ]
     
    # Modelo
    model = AutoEncoder(shape, in_size)
    dummy = torch.rand(10,cin,in_size,in_size)
    # Test
    assert(model(dummy).shape == torch.Size([10,cin,in_size,in_size]))
    # Get features
    assert(model.get_features(dummy).shape == torch.Size([10,1,hidden_size]))
    
    # VAE
    vae = VAE(model, latent_dim, hparams={"hidden_size": hidden_size})
    # vae.summary()
    # Test
    assert(vae(dummy).shape == torch.Size([10,cin,in_size,in_size]))
    # Get features
    print(vae.get_features(dummy).shape)
    assert vae.get_features(dummy).shape == torch.Size([10,1,latent_dim]), vae.get_features(dummy).shape
    
    # Probar una pasada de train y una de validacion
    from dataset import PHM2023Dataset as PHM
    from transforms import VariableLengthMFCC2 as MFCC, LabelTransformer as LT, InverseLabelTransformer as ILT
    from utils import get_data_loader
    from model1 import GearModel

    # generator = torch.Generator().manual_seed(42)

    # Cargar datos
    train_dataset = PHM('train', transform=MFCC(), label_transform=LT(), base_dir='./data/', tranform_dir="./data_mfcc2/", force_transform=True)
    # Printear la transformacion
    print(train_dataset.transform)
    # Ver directorio actual
    import os
    print(os.getcwd())
    
    # Ver dimensiones
    print(train_dataset[0][0].shape)
    
    # Cargar dataloader
    train_loader = get_data_loader(train_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
    
    # Cargar modelo
    model = VAE(model, hidden_size, hparams={"lr": 1e-3, "scheduler": "step"})
    # model.summary()
    model = model.to('cuda')
    
    # Entrenar una epoca con trainer de pytorch lightning
    from pytorch_lightning import Trainer
    
    trainer = Trainer(max_epochs=1)
    trainer.fit(model, train_loader)
    
    
    
