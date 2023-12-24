import argparse, os, numpy as np
import torch, pandas as pd
import pytorch_lightning as pl
from callbacks import StandarCallbacks as SC
from model3 import Autoencoder as AE, Classifier as CLF, ClassifierRegressor as CR, ConvEncoder as CE, ConvDecoder as CD
from dataset import PHM2023DataModule
from transforms import VariableLengthMFCC as MFCC, VariableLengthMelSpectrogram as VMS, LabelTransformer as LT, InverseLabelTransformer as ILT
import matplotlib.pyplot as plt

def base_plot(y_true, y_pred, y_reg, confidence, title, ylab="Clase", xlab="Regresión", tight=True):
    plt.style.use('ggplot')
    fig, axs = plt.subplots(1, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [1], "hspace": 0.2, "top": 0.9, "bottom": 0.1, "left": 0.1, "right": 0.9})
    for i in range(11):
        axs.text(i, 0.2, np.sum(y_pred == i), ha="center", va="center", color="black")
        
    # Linea de 0.5
    axs.axhline(0.5, color="black", linestyle="--", linewidth=1, label=f"50% de confianza")
    plt.xlabel(ylab)
    plt.ylabel(xlab)
    if tight:
        plt.tight_layout()
    plt.ylim(0, 1)
    plt.xlim(-1.5, 10.5)
    plt.title(title, fontsize=16, pad=20, loc="center", y=1.03)
    return fig, axs

def evaluate_model(dataloader, model, device="cpu"):
    y_true = np.array([])
    y_pred = np.array([])
    y_regs = np.array([])
    confidence = np.array([])
    pxs = np.array([])
    with torch.no_grad():
        for x, y in dataloader:
            y = y.numpy()
            x = x.to(device)
            y_clf, y_reg = model(x)
            y_max = torch.argmax(y_clf, dim=1).cpu().numpy()
            y_true = np.concatenate((y_true, y)) if y_true.size else y
            y_pred = np.concatenate((y_pred, y_max)) if y_pred.size else y_max
            y_regs = np.concatenate((y_regs, y_reg)) if y_regs.size else y_reg
            px = torch.softmax(y_clf, dim=1)
            px_round = torch.floor(px * 1000) / 1000.0
            pxs = np.concatenate((pxs, px_round.cpu().numpy())) if pxs.size else px_round.cpu().numpy()
            Hp = torch.sum(-px * torch.log(px), dim=1).cpu().numpy()
            conf = 1 - Hp / np.log(y_clf.shape[1])
            confidence = np.concatenate((confidence, conf)) if confidence.size else conf
            
    sample_numbers = np.arange(1, y_pred.shape[0] + 1)
    data = np.concatenate((sample_numbers.reshape(-1, 1), pxs, confidence.reshape(-1, 1)), axis=1)
    
    # Verificar que todos sean del mismo largo
    print(f"Shapes: {y_true.shape}, {y_pred.shape}, {y_regs.shape}, {confidence.shape}, {pxs.shape}, {sample_numbers.shape}, {data.shape}")
    
    return y_true, y_pred, y_regs, confidence, pxs, sample_numbers, data

def verify_pxs_sum(data):
    bad_rows = 0
    for i in range(len(data)):
        sum_pxs = np.sum(data[i, 1:12])
        if sum_pxs > 1:
            print(f"Error en la fila {i}, suman {sum_pxs}")
            bad_rows += 1
        else:
            print(f"Ok en la fila {i}, suman {sum_pxs}")
    print(f"Total de filas malas: {bad_rows}, de {len(data)}")

def main(version, log_dir, name, example, device="cpu"):
    # DataModule
    data_params = {
            "batch_size": 32,
            "base_dir": "./data/",
            "tranform_dir": "./data_vms/",
            "force_transform": False,
            "transform": VMS(), # MFCC(), VMS()
            "label_transform": None,
            "supervised": True,
            "clustering": True,
        }
    
    # Cargar modelo
    ckpt_path = os.path.join(log_dir, name, f"version_{version}", "checkpoints")
    print(f"Loading model from {ckpt_path}")
    
    if os.path.exists(ckpt_path):
        ckpt_path = os.path.join(ckpt_path, os.listdir(ckpt_path)[0])
        
        print(f"Loading checkpoint from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        # vae = AE(encoder=CE( **{"input_size": [4, 64, 64], "hidden_sizes": [16, 185, 32, 243], "output_size": 64}), hparams = checkpoint["hyper_parameters"] ).to(device)
        
        model = CR(encoder=CE( **{"input_size": [4, 64, 64], "hidden_sizes": [16, 185, 32, 243], "output_size": 64}), hparams = checkpoint["hyper_parameters"] ).to(device) 
        model.load_state_dict(checkpoint["state_dict"])
    else:
        print("No checkpoint found")
        return
    
    # DataModule
    dm = PHM2023DataModule(**data_params)
    dm.setup()
    model.eval()
    
    # Ver salida de red para un batch
    dl = dm.train_dataloader()
    it = 0
    for _x, _y in dl:
        if it == example:
            x = _x.to(device)
            y = _y.unsqueeze(1).to(device)
            break 
        it += 1
    y_clf, y_reg = model(x)
    # y_max = torch.argmax(y_clf, dim=1).cpu().numpy()
    # print(f"Clasificación: {y_max}, Regresión: {y_reg} (real: {y})")
    print(y_reg.shape, y.shape, y_clf.shape)
    

    
    
    # # test vectors de regresión
    # y_reg_test = torch.linspace(-1, 11, 10).to(device)
    # y_test = torch.zeros_like(y_reg_test).to(device)
    # losses = torch.zeros_like(y_reg_test).to(device)
    # distances = torch.zeros_like(y_reg_test).to(device)
    # for i in range(y_reg_test.shape[0]):
    #     a = y_reg_test[i].unsqueeze(0)
    #     b = y_test[i].unsqueeze(0)
    #     distances[i] = torch.abs(a - b).squeeze()
    #     a = a.repeat(32, 1)
    #     b = b.repeat(32, 1)
    #     print(a.shape, b.shape)
    #     losses[i] = model.custom_reg_loss(a, b).squeeze()
    
    # # Graficar loss vs distancia
    # fig, axs = plt.subplots(1, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [1], "hspace": 0.2, "top": 0.9, "bottom": 0.1, "left": 0.1, "right": 0.9})
    # axs.plot(distances.cpu().numpy(), losses.cpu().numpy(), color="blue", label="Loss")
    # plt.xlabel("Distancia")
    # plt.ylabel("Loss")
    # plt.tight_layout()
    # plt.ylim(-2, 5)
    # plt.xlim(-2, 10.5)
    # plt.title("Loss de regresión", fontsize=16, fontweight="bold", pad=20, loc="center", y=1.05)
    # plt.show()

    # print(f"Losses: {losses}")
    # check_loss = model.custom_reg_loss(y_reg, y)
    # print(f"Loss de regresión: {check_loss}")
    
    # -------------------- Ejemplo --------------------
    plots(dm, model, device)

def plots(dm, model, device="cpu"):
    colnames = ["sample_number"] + [f"prob_{i}" for i in range(11)] + ["confidence"]
    # -------------------- Entrenamiento --------------------
    y_true, y_pred, y_reg, confidence, pxs, sample_numbers, data = evaluate_model(dm.train_dataloader(), model)
    print(y_true.shape, y_pred.shape, y_reg.shape, confidence.shape, pxs.shape, sample_numbers.shape, data.shape)
    
    def metics(y_true, y_pred, y_reg, confidence):
        # Obtener métrica de loss de regresión y clasificación
        # Accuracy de clasificación
        clf_acc = np.sum(y_true == y_pred) / y_true.shape[0]
        # Loss de regresión
        reg_loss = np.mean((y_true - y_reg)**2)
        
        return clf_acc, reg_loss
    
    clf_acc, reg_loss = metics(y_true, y_pred, y_reg, confidence)
    print(f"Entrenamiento: Accuracy de clasificación: {clf_acc}, Loss de regresión: {reg_loss}")
     
    
    verify_pxs_sum(data)
    
    # Rango de la regresión
    print(f"Rango de la regresión: {np.min(y_reg)}, {np.max(y_reg)}")
    
    fig, axs = base_plot(y_true, y_pred, y_reg, confidence, "Clasificación de entrenamiento con confianza", ylab="Clase", xlab="Confianza")    
    axs.scatter(y_pred[y_true == y_pred], confidence[y_true == y_pred], color="green", label=f"$ 1 - \mathcal{{H}}(p) / \log(n)$ √ {np.sum(y_true == y_pred)} ({np.floor(np.sum(y_true == y_pred) * 1000. / y_true.shape[0]) / 10.}%)")
    axs.scatter(y_pred[y_true != y_pred], confidence[y_true != y_pred], color="red", label=f"$ 1 - \mathcal{{H}}(p) / \log(n)$ X {np.sum(y_true != y_pred)} ({np.floor(np.sum(y_true != y_pred) * 1000. / y_true.shape[0]) / 10.}%)")
    # Poner un margen entre los labels
    plt.legend( bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", ncol=3, borderaxespad=0.1, fontsize=8)
    plt.savefig("clf_train_conf.png")
    # plt.show()
    
    # Graficar regresiones como scatter, donde la confianza es la distancia al entero mas max(1 - cercano * 2, 0)
    # Acotar a (-1/2, 10.5)
    y_reg_clamp = np.clip(y_reg, -0.49999, 10.5)
    # Calcular confianza
    y_reg_conf = 1 - 4*(y_reg_clamp - np.round(y_reg_clamp))**2
    # axs.scatter(y_reg, y_reg_conf, color="blue", label="$ 1 - 4(r - round(r))^2$")
    # Colapsar regresiones a las clases enteras mas cercanas
    y_reg_round = np.round(y_reg_clamp)
    # Canmbiar a vector
    y_reg_round = y_reg_round.squeeze()
    
    fig, axs = base_plot(y_true, y_reg_round, y_reg, y_reg_conf, "Regresión de entrenamiento", ylab="Clase", xlab="Regresión")
    print(y_reg_round.shape, y_reg_conf.shape, y_true.shape)
    print(f"Regresiones: {y_reg_round}")
    print(f"Confianzas: {y_reg_conf}")
    print(f"Labels: {y_true}")
    # Graficar
    axs.scatter(y_reg[y_true == y_reg_round], y_reg_conf[y_true == y_reg_round], color="green", label=f"$ 1 - 4(r - [r])^2$ √{np.sum(y_true == y_reg_round)} ({np.floor(np.sum(y_true == y_reg_round) * 1000. / y_true.shape[0]) / 10.}%)")
    axs.scatter(y_reg[y_true != y_reg_round], y_reg_conf[y_true != y_reg_round], color="red", label=f"$ 1 - 4(r - [r])^2$ X{np.sum(y_true != y_reg_round)} ({np.floor(np.sum(y_true != y_reg_round) * 1000. / y_true.shape[0]) / 10.}%)")
    # Graficar 2 violin plots
    # axs.violinplot([y_reg_conf[y_true == y_reg_round], y_reg_conf[y_true != y_reg_round]], showmeans=True, showmedians=True)
    
    
    # Graficar la identidad
    plt.legend( bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", ncol=3, borderaxespad=0.1, fontsize=8)
    plt.savefig("reg_train_conf.png")
    # plt.show()    

    # -------------------- Validación --------------------
    y_true, y_pred, y_reg, confidence, pxs, sample_numbers, data = evaluate_model(dm.val_dataloader(), model)

    
    # Accuracy de clasificación y regresión solo
    clf_acc, reg_loss = metics(y_true, y_pred, y_reg, confidence)
    
    
    # Graficar regresiones como scatter, donde la confianza es la distancia al entero mas max(1 - cercano * 2, 0)
    # Acotar a (-1/2, 10.5)
    y_reg_clamp = np.clip(y_reg, -0.49999, 10.5)
    # Calcular confianza
    y_reg_conf = 1 - 4*(y_reg_clamp - np.round(y_reg_clamp))**2
    # axs.scatter(y_reg, y_reg_conf, color="blue", label="$ 1 - 4(r - round(r))^2$")
    # Colapsar regresiones a las clases enteras mas cercanas
    y_reg_round = np.round(y_reg_clamp)
    # Canmbiar a vector
    y_reg_round = y_reg_round.squeeze()
    
    fig, axs = base_plot(y_true, y_reg_round, y_reg, y_reg_conf, "Regresión de validación", ylab="Clase", xlab="Regresión")
    print(y_reg_round.shape, y_reg_conf.shape, y_true.shape)
    print(f"Regresiones: {y_reg_round}")
    print(f"Confianzas: {y_reg_conf}")
    print(f"Labels: {y_true}")
     
    # Graficar un solo scatter
    axs.scatter(y_reg, y_reg_conf, color="blue", label="$ 1 - 4(r - [r])^2$")
    # Graficar la identidad
    plt.legend( bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", ncol=3, borderaxespad=0.1, fontsize=8)
    plt.savefig("reg_val_conf.png")
    # plt.show()
    



    verify_pxs_sum(data)
    
    # Guardar en un csv
    df = pd.DataFrame(data, columns=colnames)
    df.to_csv(f"./validation_submission.csv", index=False)
    
    fig, axs = base_plot(y_true, y_pred, y_reg, confidence, "Clasificación de validación con confianza", ylab="Clase", xlab="Confianza")
    axs.scatter(y_pred, confidence, color="blue", label="$ 1 - \mathcal{H}(p) / \log(n)$")
    # axs.violinplot(confidence, showmeans=True, showmedians=True)
    plt.legend( bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", ncol=3, borderaxespad=0.1, fontsize=8)
    plt.savefig("clf_val_conf.png")

    # plt.show()
        
    # Histograma de clases
    fig, axs = plt.subplots(1, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [1], "hspace": 0.2, "top": 0.9, "bottom": 0.1, "left": 0.1, "right": 0.9})
    bin_count = np.bincount(y_pred)
    # Rellenar con ceros
    bin_count = np.concatenate((bin_count, np.zeros(11 - bin_count.shape[0])))
    plt.bar(np.arange(11), bin_count, color="blue", label="Clases")
    plt.xlabel("Clase")
    plt.ylabel("Cantidad")
    plt.title("Histograma de clases de validación", fontsize=16, pad=10, loc="center", y=1)
    plt.savefig("clf_val_hist.png")
    # plt.show()
    
    # -------------------- Test --------------------
    y_true, y_pred, y_reg, confidence, pxs, sample_numbers, data = evaluate_model(dm.test_dataloader(), model)
    print(y_true.shape, y_pred.shape, y_reg.shape, confidence.shape, pxs.shape, sample_numbers.shape, data.shape)
    
    
    # Graficar regresiones como scatter, donde la confianza es la distancia al entero mas max(1 - cercano * 2, 0)
    # Acotar a (-1/2, 10.5)
    y_reg_clamp = np.clip(y_reg, -0.49999, 10.5)
    # Calcular confianza
    y_reg_conf = 1 - 4*(y_reg_clamp - np.round(y_reg_clamp))**2
    # axs.scatter(y_reg, y_reg_conf, color="blue", label="$ 1 - 4(r - round(r))^2$")
    # Colapsar regresiones a las clases enteras mas cercanas
    y_reg_round = np.round(y_reg_clamp)
    # Canmbiar a vector
    y_reg_round = y_reg_round.squeeze()
    
    fig, axs = base_plot(y_true, y_reg_round, y_reg, y_reg_conf, "Regresión de test", ylab="Clase", xlab="Regresión")
    print(y_reg_round.shape, y_reg_conf.shape, y_true.shape)
    print(f"Regresiones: {y_reg_round}")
    print(f"Confianzas: {y_reg_conf}")
    print(f"Labels: {y_true}")
     
    # Graficar un solo scatter
    axs.scatter(y_reg, y_reg_conf, color="blue", label="$ 1 - 4(r - [r])^2$")
    # Graficar la identidad
    plt.legend( bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", ncol=3, borderaxespad=0.1, fontsize=8)
    plt.savefig("reg_test_conf.png")
    
    verify_pxs_sum(data)
    df = pd.DataFrame(data, columns=colnames)
    df.to_csv(f"./submission.csv", index=False)
        
    fig, axs = base_plot(y_true, y_pred, y_reg, confidence, "Clasificación de test con confianza", ylab="Clase", xlab="Confianza")
    axs.scatter(y_pred, confidence, color="blue", label="$ 1 - \mathcal{H}(p) / \log(n)$")
    # en vez de scatter, hacer violin plot
    # axs.violinplot(confidence, showmeans=True, showmedians=True)
    plt.legend( bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", ncol=3, borderaxespad=0.1, fontsize=8)
    plt.savefig("clf_test_conf.png")
    # plt.show()
    
    # Histograma de clases
    fig, axs = plt.subplots(1, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [1], "hspace": 0.2, "top": 0.9, "bottom": 0.1, "left": 0.1, "right": 0.9})
    bin_count = np.bincount(y_pred)
    bin_count = np.concatenate((bin_count, np.zeros(11 - bin_count.shape[0])))
    plt.bar(np.arange(11), bin_count, color="blue", label="Clases")
    plt.xlabel("Clase")
    plt.ylabel("Cantidad")
    plt.title("Histograma de clases de test", fontsize=16, pad=10, loc="center", y=1)
    plt.savefig("clf_test_hist.png")
    # plt.show()
    
    # # Scatter de regresion usando lexsort (ordenar por clase y luego por confianza), con tamaño de punto proporcional a la confianza
    # fig, axs = base_plot(y_true, y_pred, y_reg, confidence, "Clasificación de test", ylab="Clase", xlab="Regresión")
    # indices = np.lexsort((confidence, y_pred))
    # # axs.scatter(y_pred[indices], y_reg[indices], s=confidence[indices] * 100, color="blue", label="Regresión")
    # # Graficar la identidad
    # plt.legend()
    # plt.show()

    
if __name__ == "__main__":
    version = 9
    # version = 30
    log_dir = "./test_logs/"
    name = "gear_cr_conv"
    example = 30
    main(version, log_dir, name, example)
    