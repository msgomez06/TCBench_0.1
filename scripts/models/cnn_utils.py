import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import time, random

from torch.utils.data import DataLoader
from cnn_loaders import CNN4PP_Dataset

def plot_cnn_pres(model_name, train_seasons, val_seasons, test_seasons, model, criterion, train_losses, val_losses, test_set, test_loader, 
             lr, optim, sched, epochs, device, save_path):
    
    seasons = sorted(train_seasons + val_seasons + [str(test_seasons)])
    test_seasons = [str(test_seasons)] if not isinstance(test_seasons, list) else test_seasons
    
    fig, axs = plt.subplot_mosaic([[".", "a", "a", "."], 
                                   ["b", "b", "c", "c"],
                                   ["d", "d", "e", "e"]], figsize=(10, 10),
                                  gridspec_kw={"wspace": 0.70, "hspace": 0.45})
    
    ax1, ax2, ax3, ax4, ax5 = axs.values()
    
    ax1.plot(train_losses, label="train losses")
    ax1.plot(val_losses, label="val losses")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"Losses for {model_name} model")
    ax1.legend()
    
    test_loss = 0.0
    test_targets = []
    test_preds = []
    print(len(test_loader))
    t = time.time()
    with torch.no_grad():
        for batch_idx, (fields, coords, targets) in enumerate(test_loader):
            fields, coords, targets = fields.float().to(device), coords.float().to(device), targets.float().to(device)
            pred = model(fields, coords)
            test_loss += criterion(pred, targets).cpu().mean()
            
            targets, preds = targets.cpu().numpy(), pred.cpu().numpy()
            test_targets.extend(np.concatenate((targets[:, :-2]*test_set.target_std+test_set.target_mean, targets[:, -2:]*np.array([90, 180])), axis=1))
            test_preds.extend(np.concatenate((preds[:, :-2]*test_set.target_std+test_set.target_mean, preds[:, -2:]*np.array([90, 180])), axis=1))           
            
            if batch_idx%(len(test_loader)//5)==0:
                print(f"Batch {batch_idx}/{len(test_loader)} ({time.time()-t:.2f} s)")
            
    test_loss /= len(test_loader)
    test_targets = np.array(test_targets)
    test_preds = np.array(test_preds)
    
    print(test_preds.shape, test_targets.shape)
    
    ax2.scatter(test_preds[:, 0], test_targets[:, 0], label="Wind", s=0.1, alpha=1)
    ax2.plot(test_targets[:, 0], test_targets[:, 0], c="r", label="y=x")
    ax2.set_xlabel("Prediction")
    ax2.set_ylabel("Target")
    ax2.set_title("Wind")
    ax2.legend()
    
    ax3.scatter(test_preds[:, 1], test_targets[:, 1], label="Pressure", s=0.1, alpha=1)
    ax3.plot(test_targets[:, 1], test_targets[:, 1], c="r", label="y=x")
    ax3.set_xlabel("Prediction")
    ax3.set_ylabel("Target")
    ax3.set_title("Pressure")
    ax3.legend()
    
    ax4.hist(test_preds[:, 0], bins=50, weights=np.ones(test_preds.shape[0])/test_preds.shape[0], label="Histogram of predictions", alpha=0.7)
    ax4.hist(test_targets[:, 0], bins=50, weights=np.ones(test_preds.shape[0])/test_preds.shape[0], label="Histogram of targets", alpha=0.5)
    ax4.set_xlabel("Wind")
    ax4.set_ylabel("Frequency")
    ax4.set_title("Wind Histograms")
    ax4.legend()
    
    ax5.hist(test_preds[:, 1], bins=50, weights=np.ones(test_preds.shape[0])/test_preds.shape[0], label="Histogram of predictions", alpha=0.7)
    ax5.hist(test_targets[:, 1], bins=50, weights=np.ones(test_preds.shape[0])/test_preds.shape[0], label="Histogram of targets", alpha=0.5)
    ax5.set_xlabel("Pressure")
    ax5.set_ylabel("Frequency")
    ax5.set_title("Pressure Histograms")
    ax5.legend()
    
    fig.suptitle(f"Model: {model_name} | Train: {', '.join(train_seasons)}, Val: {', '.join(val_seasons)}, Test: {', '.join(test_seasons)}\nTest Loss: {test_loss:.4f}")
    fig.savefig(f"{save_path}/Figs/{model_name}_season_{'_to_'.join([seasons[0],seasons[-1]])}"+\
                f"_epochs_{epochs}_lr_{lr}_optim_{optim}_sched_{sched}.png", dpi=500)
        
    

def load_and_plot(model_name, pres, epochs, learning_rate, optim, sched, train_losses, val_losses, device, 
                  train_seasons, val_seasons, test_seasons=2008,
                  save_path='/users/lpoulain/louis/plots/cnn',
                  data_path="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/",
                  df_path="/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/ERA5/TC_track_filtered_1980_00_06_12_18.csv"):
    
    model = torch.load(f"{save_path}/Models/{model_name}_pres_{pres}_epochs_{epochs}_lr_{learning_rate}_optim_{optim}"\
                     + f"_sched_{sched}_{'_'.join(train_seasons)}.pt", map_location=device)
    
    test_set = CNN4PP_Dataset(data_path, model_name, df_path, seasons=test_seasons, pres=pres, train_seasons=train_seasons)
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False, num_workers=1)
    #test_loader = DataLoader(test_set, batch_size=1024, shuffle=False, num_workers=1)
    
    plot_cnn_pres(model_name=model_name, train_seasons=train_seasons, val_seasons=val_seasons, test_seasons=test_seasons, model=model, 
                  criterion=nn.MSELoss(reduction='none'), train_losses=train_losses, val_losses=val_losses, test_set=test_set, test_loader=test_loader, 
                  lr=learning_rate, optim=optim, sched=sched, epochs=epochs, device=device, save_path=save_path)
    
    
    
if __name__ == "__main__":
    
    model_name = 'graphcast'
    pres = True
    epochs = 20
    learning_rate = 0.01
    optim = 'adam'
    sched = 'cosine_annealing'
    train_losses = [0.6227650001231168, 0.6119483037334349, 0.6139808170083496, 0.6140421433374286, 0.6099564190747009,
                    0.60572314719773, 0.6097113668608168, 0.5883500267234113, 0.5747776502329442, 0.5652339003152318,
                    0.5573669509341319, 0.5496919985239704, 0.5425492350839907, 0.535276373123957, 0.5290123479440808,
                    0.5237273731993304, 0.5192654200519125, 0.5147068410077029, 0.512261855105559, 0.5106430954817268]

    val_losses = [0.6237815673986491, 0.5847966070592838, 0.578215099218553, 0.5772222013477861, 0.577008346952226,
                  0.5817092912693094, 0.5552693184492362, 0.5574375193049438, 0.5551653945968099, 0.5474963659135095,
                  0.5452220662884468, 0.5403911870000137, 0.5326276294832683, 0.5251873648405945, 0.517332472940431,
                  0.5112758376193742, 0.507685202914868, 0.5062574728453246, 0.50578330005825, 0.5058453223348534]

    
    device = "cpu"
    
    train_seasons = sorted(['2002', '2006', '2001', '2003', '2007', '2005'])
    val_seasons = ['2004', '2000']
    test_seasons = 2008

    load_and_plot(model_name, pres, epochs, learning_rate, optim, sched, train_losses, val_losses, device, train_seasons, val_seasons, test_seasons)