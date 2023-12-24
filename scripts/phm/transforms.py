import torch
import torchaudio.transforms as T
import torch.nn.functional as F


class VariableLengthMFCC:
    def __init__(self, n_mfcc=13):
        self.freq = 20480
        for k in range(3):
            setattr(
                self,
                f"t{3 * 2**k}s",
                T.MFCC(
                    sample_rate=self.freq,
                    n_mfcc=n_mfcc,
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

class InverseLabelTransformer:
    def __init__(self):
        self.dict = {
            0: 0,
            1: 1,
            2: 2,
            3: 3,
            4: 4, 
            5: 6,
            6: 8,
            -1:-1
        }
    def __call__(self, label):
        if label.shape:
            return torch.tensor([self.dict[l.item()] for l in label])
        return self.dict[label]
    
class LabelTransformer:
    def __init__(self):
        self.dict = {
            0: 0,
            1: 1,
            2: 2,
            3: 3,
            4: 4, 
            6: 5,
            8: 6,
            -1:-1
        }
        
    def __call__(self, label):
        return self.dict[label]    
    
    
class VariableLengthMFCC2:
    def __init__(self, interpolate=True):
        self.interpolate = interpolate
        self.freq = 20480
        for k in range(3):
            setattr(
                self,
                f"t{3 * 2**k}s",
                T.MFCC(
                    sample_rate=self.freq,
                    n_mfcc=97,
                    log_mels = True,
                    melkwargs={
                        "n_fft": self.freq // int(2 ** (4 - k)),
                        "hop_length": self.freq // int(2 ** (5 - k)),
                        "n_mels": 97,  # mels son los filtros de mel scale que se aplican a la señal
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
                mfcc = torch.stack(
                    [
                        getattr(self, f"t{time}s")(signal[:m, i])
                        for i in range(signal.shape[1])
                    ]
                )
                if self.interpolate:
                    if len(mfcc.shape) == 3:
                        return F.interpolate(mfcc.unsqueeze(0), size=128, mode='bilinear').squeeze(0)
                    else:
                        return F.interpolate(mfcc, size=128, mode='bilinear')
                else:
                    return mfcc
                
        raise ValueError(f"La señal es muy corta para aplicar MFCC: {signal.shape}")
    
class VariableLengthMelSpectrogram:
    def __init__(self, n_mels = 49, interpolate=True, normalize=True, size=128):
        self.interpolate = interpolate
        self.n_mels = n_mels
        self.normalize = normalize
        self.size = size
        self.freq = 20480
        for k in range(3):
            setattr(
                self,
                f"t{3 * 2**k}s",
                T.MelSpectrogram(
                    sample_rate=self.freq,
                    n_fft= self.freq // int(2 ** (3 - k)),
                    hop_length = self.freq // int(2 ** (4 - k)),
                    n_mels = self.n_mels,  # mels son los filtros de mel scale que se aplican a la señal
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
                mfcc = torch.stack(
                    [
                        getattr(self, f"t{time}s")(signal[:m, i])
                        for i in range(signal.shape[1])
                    ]
                )
                if self.interpolate:
                    if len(mfcc.shape) == 3:
                        mfcc = F.interpolate(mfcc.unsqueeze(0), size=self.size, mode='bilinear').squeeze(0)
                    else:
                        mfcc = F.interpolate(mfcc, size=self.size, mode='bilinear')
                if self.normalize:
                    # Normalización cada espectro (cada instante de tiempo)
                    for i in range(mfcc.shape[0]):
                        mfcc[i] = (mfcc[i] - mfcc[i].mean()) / mfcc[i].std()
                
                mfcc = 2 * (mfcc - mfcc.min()) / (mfcc.max() - mfcc.min()) - 1
                
                return mfcc
                
        raise ValueError(f"La señal es muy corta para aplicar MFCC: {signal.shape}")
    