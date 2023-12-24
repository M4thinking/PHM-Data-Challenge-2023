import torch, numpy as np
import torch.nn as nn
import pytorch_lightning as pl
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import BaseModel
from torch.nn import functional as F
from torchmetrics.functional import accuracy
from scipy.interpolate import lagrange

class Encoder(pl.LightningModule):
    # Crear encoder lineal desde cero
    def __init__(self, input_size = [4, 49, 49], output_size = 128, hidden_sizes = [30, 64, 32]):
        super().__init__()
        self.save_hyperparameters()
        self.input_size = input_size
        self.output_size = output_size
        self.dropout = self.hparams.get("dropout", 0.0)
        self.encoder = nn.Sequential()
        for i in range(len(hidden_sizes)):
            if i == 0:
                self.encoder.add_module('linear'+str(i), nn.Linear(np.prod(input_size), hidden_sizes[i]))
                self.encoder.add_module('dropout'+str(i), nn.Dropout(self.dropout))
            else:
                self.encoder.add_module('linear'+str(i), nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            self.encoder.add_module('dropout'+str(i), nn.Dropout(self.dropout))
            self.encoder.add_module('norm'+str(i), nn.BatchNorm1d(hidden_sizes[i]))
            self.encoder.add_module('relu'+str(i), nn.ReLU())
        self.encoder.add_module('linear'+str(len(hidden_sizes)), nn.Linear(hidden_sizes[-1], output_size))
         
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.encoder(x)
        return x
    
    def summary(self):
        print(self.encoder)
        print("Encoder summary")
        print("Input size: ", self.input_size)
        print("Output size: ", self.output_size)
        print("Encoder summary")
    
class Decoder(pl.LightningModule):
    # Crear decoder lineal desde cero
    def __init__(self, input_size, output_size = [4, 49, 49], hidden_sizes = [32, 64, 30]):
        super().__init__()
        self.save_hyperparameters()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.decoder = nn.Sequential()
        self.dropout = self.hparams.get("dropout", 0.0)
        for i in range(len(hidden_sizes)):
            if i == 0:
                self.decoder.add_module('linear'+str(i), nn.Linear(input_size, hidden_sizes[i]))
            else:
                self.decoder.add_module('linear'+str(i), nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            self.decoder.add_module('dropout'+str(i), nn.Dropout(self.dropout))
            self.decoder.add_module('norm'+str(i), nn.BatchNorm1d(hidden_sizes[i]))
            self.decoder.add_module('relu'+str(i), nn.ReLU())
        self.decoder.add_module('linear'+str(len(hidden_sizes)), nn.Linear(hidden_sizes[-1], np.prod(output_size)))
         
    def forward(self, x):
        x = self.decoder(x)
        x = x.view(x.size(0), *self.output_size)
        return x
    
    def summary(self):
        print(self.decoder)
        print("Decoder summary")
        print("Input size: ", self.input_size)
        print("Output size: ", self.output_size)
        print("Hidden sizes: ", self.hidden_sizes)
        print("Decoder summary")
        
class ConvEncoder(pl.LightningModule):
    def __init__(self, input_size = [4, 64, 64], output_size = 128, hidden_sizes = [128, 64, 32, 16]):
        super().__init__()
        self.save_hyperparameters()
        self.input_size = input_size
        self.output_size = output_size
        self.dropout = self.hparams.get("dropout", 0.0)
        self.encoder = nn.Sequential()
        for i in range(len(hidden_sizes)):
            if i == 0:
                self.encoder.add_module('conv'+str(i), nn.Conv2d(input_size[0], hidden_sizes[i], kernel_size=3, stride=1, padding=1))
                self.encoder.add_module('dropout'+str(i), nn.Dropout(self.dropout))
            else:
                self.encoder.add_module('conv'+str(i), nn.Conv2d(hidden_sizes[i-1], hidden_sizes[i], kernel_size=3, stride=1, padding=1))
            self.encoder.add_module('dropout'+str(i), nn.Dropout(self.dropout))
            self.encoder.add_module('norm'+str(i), nn.BatchNorm2d(hidden_sizes[i]))
            self.encoder.add_module('relu'+str(i), nn.ReLU())
            self.encoder.add_module('pool'+str(i), nn.MaxPool2d(kernel_size=2, stride=2))
        
        x = torch.randn(1, *input_size)
        _, c, h, w = self.encoder(x).shape
        self.encoder.add_module('flatten', nn.Flatten())
        self.encoder.add_module('linear'+str(len(hidden_sizes)), nn.Linear(c*h*w, output_size))

        self.prod = c*h*w
        
    def get_prod(self):
        return self.prod
        
    def forward(self, x):
        x = self.encoder(x)
        return x
    
    def summary(self):
        print(self.encoder)
        print("Encoder summary")
        print("Input size: ", self.input_size)
        print("Output size: ", self.output_size)
        print("Encoder shape summary:")
        for layer in self.encoder:
            x = layer(x)
            print(layer.__class__.__name__,'output shape:\t',x.shape)
        
class ConvDecoder(pl.LightningModule):
    def __init__(self, input_size, output_size = [4, 64, 64], hidden_sizes = [16, 32, 64, 128], prod=256):
        super().__init__()
        self.save_hyperparameters()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.decoder = nn.Sequential()
        self.dropout = self.hparams.get("dropout", 0.0)
        # Pasar de vector de 128 a tensor de 4xnp.sqrt(prod/4)xnp.sqrt(prod/4)
        self.decoder.add_module('linear'+str(0), nn.Linear(input_size, prod))
        self.decoder.add_module('dropout'+str(0), nn.Dropout(self.dropout))
        self.decoder.add_module('relu'+str(0), nn.ReLU())
        self.decoder.add_module('reshape'+str(0), nn.Unflatten(1, (hidden_sizes[0], int(np.sqrt(prod/hidden_sizes[0])), int(np.sqrt(prod/hidden_sizes[0])))))
        for i in range(len(hidden_sizes)):
            if i == len(hidden_sizes)-1:
                self.decoder.add_module('conv'+str(i), nn.ConvTranspose2d(hidden_sizes[i], output_size[0], kernel_size=3, stride=1, padding=1))
                self.decoder.add_module('norm'+str(i), nn.BatchNorm2d(output_size[0]))
            else:
                self.decoder.add_module('conv'+str(i), nn.ConvTranspose2d(hidden_sizes[i], hidden_sizes[i+1], kernel_size=3, stride=1, padding=1))
                self.decoder.add_module('norm'+str(i), nn.BatchNorm2d(hidden_sizes[i+1]))
            self.decoder.add_module('relu'+str(i), nn.ReLU())
            self.decoder.add_module('upsample'+str(i), nn.Upsample(scale_factor=2, mode='nearest'))
        self.decoder.add_module('conv'+str(len(hidden_sizes)), nn.ConvTranspose2d(output_size[0], output_size[0], kernel_size=3, stride=1, padding=1))
        
    def forward(self, x):
        x = self.decoder(x)
        x = x.view(x.size(0), *self.output_size)
        return x
    
    def summary(self):
        print(self.decoder)
        print("Decoder summary")
        print("Input size: ", self.input_size)
        print("Output size: ", self.output_size)
        print("Hidden sizes: ", self.hidden_sizes)
        print("Decoder summary")
        x = torch.randn(2, self.input_size)
        for layer in self.decoder:
            x = layer(x)
            print(layer.__class__.__name__,'output shape:\t',x.shape)
    
class Autoencoder(pl.LightningModule):
    # Crear autoencoder
    def __init__(self, input_size = None, hidden_size = None, output_size = None, hparams={}):
        super().__init__()
        # Guardar hiperparámetros
        self.hparams.update(hparams)
        self.save_hyperparameters(hparams)
        self.beta = self.hparams.get("beta", 1.0)
        self.input_size = input_size if input_size else self.hparams.get("input_size", [4, 49, 49])
        self.hidden_size = hidden_size if hidden_size else self.hparams.get("hidden_size", 128)
        self.output_size = output_size if output_size else self.hparams.get("output_size", [4, 49, 49])
        self.encoder_hidden_sizes = self.hparams.get("encoder_hidden_sizes", [20, 20, 20])
        self.decoder_hidden_sizes = self.hparams.get("decoder_hidden_sizes", [20, 20, 20])
        self.latent_dim = self.hparams.get("latent_dim", 128)
        if self.hparams.get("conv", False):
            self.encoder = ConvEncoder(self.input_size, self.hidden_size, self.encoder_hidden_sizes)
            self.decoder = ConvDecoder(self.hidden_size, self.output_size, self.decoder_hidden_sizes)
        else:
            self.encoder = Encoder(self.input_size, self.hidden_size, self.encoder_hidden_sizes)
            self.decoder = Decoder(self.hidden_size, self.output_size, self.decoder_hidden_sizes)
        
        if self.hparams.get("vae", False):
            self.fc_mu = nn.Sequential(
                nn.Linear(self.hidden_size, self.latent_dim),
                nn.BatchNorm1d(self.latent_dim), 
                nn.ReLU()
            )
            self.fc_var = nn.Sequential(
                nn.Linear(self.hidden_size, self.latent_dim),
                nn.BatchNorm1d(self.latent_dim),
                nn.ReLU()
                )
            self.fc_z = nn.Sequential(
                nn.Linear(hparams.get("latent_dim", 128), self.hidden_size),
                nn.BatchNorm1d(self.hidden_size),
                nn.ReLU()
                )
            
    def noise(self, x):
        return x + torch.randn_like(x) * self.hparams.get("noise", 0.01)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
        
    def forward(self, x):
        x = self.encoder(x)
        if self.hparams.get("vae", False):
            mu = self.fc_mu(x)
            logvar = self.fc_var(x)
            z = self.reparameterize(mu, logvar)
            z = self.fc_z(z)
            x = self.decoder(z)
            return x, mu, logvar
        return self.decoder(x), None, None
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        if self.hparams.get("noise", 0.01) > 0.0:
            x_recon, mean, logvar = self(self.noise(x))
        else:
            x_recon, mean, logvar = self(x)
        loss, recon_loss, kl_divergence = self.loss_function(x, x_recon, mean, logvar)
        self.log('train/loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        if self.hparams.get("vae", False):
            self.log('train/recon_loss', recon_loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log('train/kl_divergence', kl_divergence, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_recon, mean, logvar = self(x)
        loss, recon_loss, kl_divergence = self.loss_function(x, x_recon, mean, logvar)
        self.log('val/loss', loss, prog_bar=True)
        if self.hparams.get("vae", False):
            self.log('val/recon_loss', recon_loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log('val/kl_divergence', kl_divergence, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def loss_function(self, x, x_recon, mean, logvar):
        if self.hparams.get("vae", False): 
            recon_loss = F.mse_loss(x_recon, x, reduction='sum')
            kl_divergence = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
            return recon_loss + self.beta * kl_divergence, recon_loss, kl_divergence
        else:
            return F.mse_loss(x_recon, x, reduction='sum'), None, None
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.get("lr", 0.01), weight_decay=self.hparams.get("wd", 0.0))
        return optimizer
    
    def get_features(self, x):
        x = self.encoder(x)
        return x
    
    def summary(self):
        print("Autoencoder summary")
        print("Input size: ", self.input_size)
        print("Output size: ", self.output_size)
        print("Hidden size: ", self.hidden_size)
        print("Encoder hidden sizes: ", self.encoder_hidden_sizes)
        print("Decoder hidden sizes: ", self.decoder_hidden_sizes)
        print("Autoencoder summary")
    
class Classifier(pl.LightningModule):
    # Crear clasificador
    def __init__(self, encoder, hparams={}):
        super().__init__()
        # Guardar hiperparámetros
        self.hparams.update(hparams)
        self.save_hyperparameters(hparams)
        self.hidden_size = self.hparams.get("hidden_size", 128)
        self.hidden_sizes = self.hparams.get("hidden_sizes", [128, 64, 32])
        self.output_size = self.hparams.get("output_size", 11)
        self.encoder = encoder
        self.num_classes = self.hparams.get("num_classes", 11)
        # Ver salida del encoder
        x = torch.randn(2, *self.encoder.input_size)
        b, l = self.encoder(x).shape
        print("Encoder output shape: ", b, l)
        self.fc = nn.Sequential()
        for i in range(len(self.hidden_sizes)):
            if i == 0:
                self.fc.add_module('linear'+str(i), nn.Linear(l, self.hidden_sizes[i]))
                self.fc.add_module('dropout'+str(i), nn.Dropout(self.hparams.get("dropout", 0.2)))
            else:
                self.fc.add_module('linear'+str(i), nn.Linear(self.hidden_sizes[i-1], self.hidden_sizes[i]))
            self.fc.add_module('dropout'+str(i), nn.Dropout(self.hparams.get("dropout", 0.2)))
            self.fc.add_module('norm'+str(i), nn.BatchNorm1d(self.hidden_sizes[i]))
            self.fc.add_module('relu'+str(i), nn.ReLU())
        self.fc.add_module('linear'+str(len(self.hidden_sizes)), nn.Linear(self.hidden_sizes[-1], self.output_size))
        
        if self.hparams.get("freeze_encoder", True):
            for param in self.encoder.parameters():
                param.requires_grad = False
                
        # Incializar pesos de la capa fc
        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
            if isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x)
        return x
    
    def get_features(self, x):
        x = self.encoder(x)
        return x
    
    def loss_function(self, y_hat, y):
        return F.cross_entropy(y_hat, y)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_function(y_hat, y)
        acc = accuracy(y_hat, y, task="multiclass", num_classes=self.num_classes)
        self.log('train/loss', loss, on_step=False, on_epoch=True)
        self.log('train/acc', acc, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        acc = accuracy(y_hat, y, task="multiclass", num_classes=self.num_classes)
        loss = self.loss_function(y_hat, y)
        self.log('val/loss', loss)
        self.log('val/acc', acc)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.get("lr", 0.001), weight_decay=self.hparams.get("wd", 0.0))
        return optimizer
    
    
    
class ClassifierRegressor(pl.LightningModule):
    def __init__(self, encoder, hparams={}):
        super().__init__()
        self.hparams.update(hparams)
        self.save_hyperparameters(hparams)
        self.num_classes = self.hparams.get("num_classes", 11)
        self.hidden_size = self.hparams.get("hidden_size", 128)
        self.hidden_sizes = self.hparams.get("hidden_sizes", [128, 64, 32])
        self.output_size = self.hparams.get("output_size", 11)
        self.encoder = encoder
        self.num_classes = self.hparams.get("num_classes", 11)
        
        self.classification_head = nn.Sequential()
        for i in range(len(self.hidden_sizes)):
            if i == 0:
                self.classification_head.add_module('linear'+str(i), nn.Linear(self.encoder.output_size, self.hidden_sizes[i]))
                self.classification_head.add_module('dropout'+str(i), nn.Dropout(self.hparams.get("dropout", 0.2)))
            else:
                self.classification_head.add_module('linear'+str(i), nn.Linear(self.hidden_sizes[i-1], self.hidden_sizes[i]))
            self.classification_head.add_module('dropout'+str(i), nn.Dropout(self.hparams.get("dropout", 0.2)))
            self.classification_head.add_module('norm'+str(i), nn.BatchNorm1d(self.hidden_sizes[i]))
            self.classification_head.add_module('relu'+str(i), nn.ReLU())
        self.classification_head.add_module('linear'+str(len(self.hidden_sizes)), nn.Linear(self.hidden_sizes[-1], self.output_size))
        
        self.regression_head = nn.Sequential()
        for i in range(len(self.hidden_sizes)):
            if i == 0:
                self.regression_head.add_module('linear'+str(i), nn.Linear(self.encoder.output_size, self.hidden_sizes[i]))
                self.regression_head.add_module('dropout'+str(i), nn.Dropout(self.hparams.get("dropout", 0.2)))
            else:
                self.regression_head.add_module('linear'+str(i), nn.Linear(self.hidden_sizes[i-1], self.hidden_sizes[i]))
            self.regression_head.add_module('dropout'+str(i), nn.Dropout(self.hparams.get("dropout", 0.2)))
            self.regression_head.add_module('norm'+str(i), nn.BatchNorm1d(self.hidden_sizes[i]))
            self.regression_head.add_module('relu'+str(i), nn.ReLU())
        self.regression_head.add_module('linear'+str(len(self.hidden_sizes)), nn.Linear(self.hidden_sizes[-1], 1))        
        
        # Inicializar pesos de las capas
        for m in self.classification_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
        
        for m in self.regression_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
                
                
                
    def on_save_checkpoint(self, checkpoint):
        checkpoint["encoder"] = self.encoder.state_dict()
        checkpoint["encoder_hparams"] = self.encoder.hparams
        checkpoint["classification_head"] = self.classification_head.state_dict()
        checkpoint["regression_head"] = self.regression_head.state_dict()
        checkpoint["hparams"] = self.hparams
        
    # Cargar el estado del encoder cada vez que se carga un checkpoint
    def on_load_checkpoint(self, checkpoint):
        for k in checkpoint.keys():
            print(k)
        self.encoder.load_state_dict(checkpoint["encoder"])
        # self.encoder.hparams = checkpoint["encoder_hparams"]
        # self.hparams = checkpoint["hparams"]
        self.classification_head.load_state_dict(checkpoint["state_dict"])
        self.regression_head.load_state_dict(checkpoint["state_dict"])
        print("Loaded encoder state dict & hparams")

    def forward(self, x):
        x = self.encoder(x)
        y_hat_classification = self.classification_head(x)
        y_hat_regression = self.regression_head(x)
        return y_hat_classification, y_hat_regression
    
    def collapse_to_interval(self, value, lower_bound = 0, upper_bound = 10):
        return torch.round(torch.clamp(value, lower_bound, upper_bound))
    
    def custom_reg_loss(self, y_hat, y):
        device = y_hat.device
        x_data = np.array([0,  1,   2, 3,   4, 5,   6, 7,   8, 9, 10, 11])
        y_data = np.array([-1, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4,  4.5, 5])
        polynomial = lagrange(x_data, y_data)
        coefficients = polynomial.c
        coefficients = torch.tensor(coefficients, dtype=torch.float32, device=device)
        
        def polynomial(x, coeffs = coefficients, device=device):
            pows = torch.arange(len(coeffs)-1, -1, -1, device=device)
            return torch.sum(coeffs * torch.pow(x, pows))
        
        distance = torch.abs(y_hat - y).to(device=device).squeeze()
        costs = list(map(lambda x: polynomial(x, coefficients, device), distance))
        loss = torch.tensor(costs, device=device)
        loss[distance > 10.5] = 4.75
        return loss.mean()

    def loss_function(self, y_hat_classification, y_hat_regression, y):
        device = y_hat_classification.device

        idx_no_label = torch.where(y == -1)[0]
        idx_label = torch.where(y != -1)[0]
        
        # Factor de penalización para regression por estar fuera del dominio de clasificación (-1/2, 10.5) cuando la clase es -1
        regression_penalty = 100
        acc_classification = torch.tensor(0.0)

        
        if y[idx_no_label].shape[0] > 0:
            label_from_reg = self.collapse_to_interval(y_hat_regression[idx_no_label].clone()).to(torch.int64).squeeze().detach()
            label_from_clf = torch.argmax(y_hat_classification[idx_no_label].clone(), dim=1).to(torch.int64).detach()
            
            y_classification = torch.cat((y[idx_label], label_from_reg), dim=0).to(device).long()
            y_regression = torch.cat((y[idx_label], label_from_clf), dim=0).to(device).unsqueeze(1).float()
            
            y_hat_classification = torch.cat((y_hat_classification[idx_label], y_hat_classification[idx_no_label]), dim=0)
            y_hat_regression = torch.cat((y_hat_regression[idx_label], y_hat_regression[idx_no_label]), dim=0)
            
            acc_classification = accuracy(y_hat_classification[idx_label], y_classification[idx_label], task="multiclass", num_classes=self.num_classes)
            
        else:
            y_classification = y.to(device).long()
            y_regression = y.unsqueeze(1).float()
            acc_classification = acc_classification_extended = accuracy(y_hat_classification, y_classification, task="multiclass", num_classes=self.num_classes)
        
        
        max_interval = torch.clamp(y_hat_regression, -3.5, 13.5)
        penalty_factor = torch.mean(torch.abs(max_interval - y_hat_regression)) * regression_penalty
        
        # Calcular accuracy
        acc_classification_extended = accuracy(y_hat_classification, y_classification, task="multiclass", num_classes=self.num_classes) 
        loss_classification = F.cross_entropy(y_hat_classification, y_classification)
        # loss_regression = F.mse_loss(y_hat_regression, y_regression) * 2
        loss_regression = self.custom_reg_loss(y_hat_regression, y_regression)
        
        return loss_classification, loss_regression, penalty_factor, acc_classification, acc_classification_extended

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat_classification, y_hat_regression = self(x)
        
        BCE, MSE, pentaly, acc, acc_ext = self.loss_function(y_hat_classification, y_hat_regression, y)
        loss = BCE + MSE + pentaly
        self.log('train/loss_classification', BCE, on_step=False, on_epoch=True)
        self.log('train/loss_regression', MSE, on_step=False, on_epoch=True)
        self.log('train/loss_penalty', pentaly, on_step=False, on_epoch=True)
        self.log('train/loss',loss, on_step=False, on_epoch=True)
        self.log('train/acc', acc, on_step=False, on_epoch=True)
        self.log('train/acc_ext', acc_ext, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        y_hat_classification, y_hat_regression = self(x)
        
        BCE, MSE, pentaly, acc, acc_ext = self.loss_function(y_hat_classification, y_hat_regression, y)
        loss = BCE + MSE + pentaly
        self.log('val/loss_classification', BCE, on_step=False, on_epoch=True)
        self.log('val/loss_regression', MSE, on_step=False, on_epoch=True)
        self.log('val/loss_penalty', pentaly, on_step=False, on_epoch=True)
        self.log('val/loss', loss, on_step=False, on_epoch=True)
        self.log('val/acc', acc, on_step=False, on_epoch=True)
        self.log('val/acc_ext', acc_ext, on_step=False, on_epoch=True)
        
        return loss  
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.get("lr", 0.001), weight_decay=self.hparams.get("wd", 0.0))
        return optimizer
     

if __name__ == "__main__":
    hparams = dict(
        max_epochs=100,
        lr=0.01,
        callbacks= None,
        num_classes=11,
    )
    # Probar encoder
    encoder = Encoder(input_size=[4, 13, 481], output_size=128)
    y = encoder(torch.randn(30, *encoder.input_size))
    assert y.shape == torch.Size([30, 128]), f"Input shape is {y.shape} instead of [30, 128]"
    # Probar decoder
    decoder = Decoder(input_size=128, output_size=[4, 13, 481])
    y = decoder(torch.randn(30, decoder.input_size))
    assert y.shape == torch.Size([30, 4, 13, 481]), f"Output shape is {y.shape} instead of [30, 4, 13, 481]"
    # Probar autoencoder
    hparams["vae"] = False
    model = Autoencoder(input_size=[4, 13, 481],
                  hidden_size=128,
                  output_size=[4, 13, 481],
                  hparams=hparams)
    y, _, _ = model(torch.randn(30, *model.input_size))
    assert y.shape == torch.Size([30, 4, 13, 481]), f"Output shape is {y.shape} instead of [30, 4, 13, 481]"
    # Probar autoencoder con VAE
    hparams["vae"] = True
    model = Autoencoder(input_size=[4, 13, 481],
                  hidden_size=128,
                  output_size=[4, 13, 481],
                  hparams=hparams)
    y, _, _ = model(torch.randn(30, *model.input_size))
    assert y.shape == torch.Size([30, 4, 13, 481]), f"Output shape is {y.shape} instead of [30, 4, 13, 481]"
    # Probar clasificador
    model = Classifier(encoder, hparams=hparams)
    y = model(torch.randn(30, *model.encoder.input_size))
    assert y.shape == torch.Size([30, model.output_size]), f"Output shape is {y.shape} instead of [30, {model.output_size}]"
    # Probar ConvEncoder
    encoder = ConvEncoder(input_size=[4, 64, 64], output_size=128)
    y = encoder(torch.randn(30, *encoder.input_size))
    assert y.shape == torch.Size([30, 128]), f"Input shape is {y.shape} instead of [30, 128]"
    
    # Probar ConvDecoder
    decoder = ConvDecoder(input_size=128, output_size=[4, 64, 64])
    y = decoder(torch.randn(30, decoder.input_size))
    assert y.shape == torch.Size([30, 4, 64, 64]), f"Output shape is {y.shape} instead of [30, 4, 64, 64]"
    
    # Probar ConvAutoencoder
    hparams["conv"] = True
    hparams["encoder_hidden_sizes"] = [128, 64, 32, 16]
    hparams["decoder_hidden_sizes"] = [16, 32, 64, 128]
    model = Autoencoder(input_size=[4, 64, 64],
                  hidden_size=256,
                  output_size=[4, 64, 64],
                  hparams=hparams)
    y, _, _ = model(torch.randn(30, *model.input_size))
    assert y.shape == torch.Size([30, 4, 64, 64]), f"Output shape is {y.shape} instead of [30, 4, 64, 64]"
    # Probar ConvAutoencoder sin VAE
    hparams["vae"] = False
    model = Autoencoder(input_size=[4, 64, 64],
                  hidden_size=256,
                  output_size=[4, 64, 64],
                  hparams=hparams)
    y, _, _ = model(torch.randn(30, *model.input_size))
    assert y.shape == torch.Size([30, 4, 64, 64]), f"Output shape is {y.shape} instead of [30, 4, 64, 64]"
    # Probar clasificador con ConvEncoder
    hparams["hidden_size"] = 128
    hparams["output_size"] = 7
    hparams["num_classes"] = 7
    model = Classifier(encoder, hparams=hparams)
    y = model(torch.randn(30, *model.encoder.input_size))
    assert y.shape == torch.Size([30, model.output_size]), f"Output shape is {y.shape} instead of [30, {model.output_size}]"
    
    # Probar clasificador-regresor
    hparams["hidden_size"] = 128
    hparams["output_size"] = 7
    hparams["num_classes"] = 7
    model = ClassifierRegressor(encoder, hparams=hparams)
    import matplotlib.pyplot as plt
    x = torch.linspace(-1, 11, 100)
    # Vector de 0s para clasificación
    y = torch.zeros_like(x)
    loss = torch.zeros_like(x)
    # Obtener la loss de regresión
    for i in range(100):
        y_hat = torch.tensor([x[i]])
        y_real = torch.tensor([y[i]])
        print(y_hat.shape, y_real.shape)
        loss[i] = model.custom_reg_loss(y_hat, y_real)
    dist = torch.abs(x - y)
    plt.plot(dist.numpy(), loss.numpy())
         