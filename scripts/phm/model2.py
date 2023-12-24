import torch
import torch.nn as nn
import pytorch_lightning as pl
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import BaseModel


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


class GearModel(BaseModel):
    def __init__(
        self,
        input_shape=[4, 13, 481],  # [channels, dim1, dim2]
        num_classes=7,
        loss_fn=nn.CrossEntropyLoss(),
        hparams=None,
    ):
        super().__init__(
            num_classes=num_classes,
            name="gear_model",
            loss_fn=loss_fn,
            hparams=hparams,
        )
        self.latent_dim = 128 if "latent_dim" not in hparams else hparams["latent_dim"]
        self.num_classes = num_classes
        self.input_shape = input_shape

        self.seq1 = nn.Sequential(
            Block(input_shape[0], 32, 3, 1, 1),
            Block(32, 128, 3, 1, 1),
            Block(128, 64, 3, 1, 1),
        )

        # Calculate the output shape of seq1
        _, channels, dim1, dim2 = self.seq1(torch.randn(1, *input_shape)).shape
        print(f"Input shape of CNN: [{input_shape[0]}, {input_shape[1]}, {input_shape[2]}]")
        print(f"Output shape of CNN: [{channels}, {dim1}, {dim2}]")

        self.seq2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * dim1 * dim2, self.latent_dim),
        )

        self.seq3 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.latent_dim, self.num_classes),
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
    
    def __str__(self) -> str:
        return f"""
        GearModel(
            input_shape={self.input_shape},
            num_classes={self.num_classes},
        )
        """



if __name__ == "__main__":
    hparams = dict(
        max_epochs=100,
        lr=0.01,
        callbacks= None,
        num_classes=11,
    )
    model = GearModel(input_shape=[4, 13, 481],
                  num_classes=hparams["num_classes"],
                  hparams=hparams)
    x = torch.randn(1, *model.input_shape)
    y = model(x)
    print(y.shape)