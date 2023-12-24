import numpy as np
import torch
import python_speech_features as psf
import torchaudio.transforms as T

# Parámetros comunes
samplerate = 20480  # Frecuencia de muestreo en Hz
overlap = 0.5  # Porcentaje de superposición (50%)
nfft = [1024, 1024, 1024]  # Tamaño de la FFT

# Duración de las señales en segundos
durations = [3, 6, 12]

# Número de coeficientes MFCC
n_mfcc = 13

# Ventanas de análisis en segundos
win_lens = [0.03, 0.06, 0.12]  # 30ms, 60ms, 120ms
k = 0
for nfft, duration, win_len in zip(nfft, durations, win_lens):
    m = samplerate * duration  # Duración en muestras

    # Calcular los coeficientes MFCC
    mfcc = psf.mfcc(
        signal=np.zeros(int(m)),
        samplerate=samplerate,
        winlen=win_len,
        winstep=win_len * overlap,
        numcep=n_mfcc,
        nfft=nfft,
    )

    # Imprimir las dimensiones de la matriz de MFCC
    print(f"Duración: {duration} segundos, Dimensiones de MFCC: {mfcc.shape}")

    mfcc2 = T.MFCC(
        sample_rate=samplerate,
        n_mfcc=13,
        melkwargs={
            "n_fft": samplerate // (2 ** (2 - k) * 10),
            "hop_length": samplerate // (2 ** (2 - k) * 20),
            "n_mels": 40,  # mels son los filtros de mel scale que se aplican a la señal
        },
    )(torch.zeros(int(m)))

    print(f"Duración: {duration} segundos, Dimensiones de MFCC 2: {mfcc2.shape}")
    k += 1

# correr el script: python scripts/mfcc_test.py
