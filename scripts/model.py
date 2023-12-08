import torch
import torch.nn as nn
from tqdm.notebook import tqdm_notebook as tqdm, trange
import time

from IPython.display import display
from matplotlib import pyplot as plt

import numpy as np


class CNN_TC(nn.Module):
    def __init__(self, in_channels, out_channels, input_size=(241,241), embedding_size=30):
        
        super(CNN_TC, self).__init__()
        size_after_3pools = (input_size[0]//8, input_size[1]//8)
        size_after_2pools = (input_size[0]//4, input_size[1]//4)
        size_after_1pools = (input_size[0]//2, input_size[1]//2)
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(out_channels, 1, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=1)
        
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.bn4 = nn.BatchNorm2d(1)
        
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        self.embedding1 = nn.Embedding(721, embedding_size) # Lat
        self.embedding2 = nn.Embedding(1440, embedding_size) # Lon
        self.embedding3 = nn.Embedding(28, embedding_size) # Ldt
        
        # define parameters for the skip connections
        ksize1 = (3+input_size[0]%2, 3+input_size[1]%2)
        ksize2 = (3+size_after_1pools[0]%2, 3+size_after_1pools[1]%2)
        ksize3 = (3+size_after_2pools[0]%2, 3+size_after_2pools[1]%2)
        pad1 = (input_size[0]%2, input_size[1]%2)
        pad2 = (size_after_1pools[0]%2, size_after_1pools[1]%2)
        pad3 = (size_after_2pools[0]%2, size_after_2pools[1]%2)
        div1 = (2 + input_size[0]%2, 2 + input_size[1]%2)
        div2 = (2 + size_after_1pools[0]%2, 2 + size_after_1pools[1]%2)
        div3 = (2 + size_after_2pools[0]%2, 2 + size_after_2pools[1]%2)
        dil1 = ((input_size[0]-size_after_3pools[0]+2*pad1[0])//div1[0], (input_size[1]-size_after_3pools[1]+2*pad1[1])//div1[1])
        dil2 = ((size_after_1pools[0]-size_after_3pools[0]+2*pad2[0])//div2[0], (size_after_1pools[1]-size_after_3pools[1]+2*pad2[1])//div2[1])
        dil3 = ((size_after_2pools[0]-size_after_3pools[0]+2*pad3[0])//div3[0], (size_after_2pools[1]-size_after_3pools[1]+2*pad3[1])//div3[1])
        
        self.resconv1 = nn.Conv2d(in_channels, 1, kernel_size=ksize1, stride=1, padding=pad1, dilation=dil1)
        self.resconv2 = nn.Conv2d(self.conv1.weight.shape[0], 1, kernel_size=ksize2, stride=1, padding=pad2, dilation=dil2)
        self.resconv3 = nn.Conv2d(self.conv2.weight.shape[0], 1, kernel_size=ksize3, stride=1, padding=pad3, dilation=dil3)
        self.reshape = nn.Flatten()
        
        self.final_conv = nn.Conv2d(1, 4, kernel_size=(embedding_size+3, embedding_size), stride=1, padding=0)
        self.view = self.add_dim
        
        
        
    def forward(self, fields, lat, lon, ldt):
        lat_idx, lon_idx, ldt_idx = torch.LongTensor([((lat[i]+90.0)//0.25).item() for i in range(lat.shape[0])]), \
                                    torch.LongTensor([(lon[i]//0.25).item() for i in range(lon.shape[0])]), \
                                    torch.LongTensor([(ldt[i]//6-1).item() for i in range(ldt.shape[0])])
        res1 = self.resconv1(fields)
        fields = self.pool1(self.bn1(self.conv1(fields)))
        #print(fields.shape, self.resconv2(fields).shape)
        res2 = self.resconv2(fields)
        
        fields = self.pool2(self.bn2(self.conv2(fields)))
        #print(fields.shape, self.resconv3(fields).shape)
        res3 = self.resconv3(fields)
        
        fields = self.pool3(self.bn3(self.conv3(fields)))
        #print(fields.shape, self.resconv4(fields).shape)
        
        fields = self.bn4(self.conv4(fields))
        
        lat_embed, lon_embed, ldt_embed = self.embedding1(lat_idx), self.embedding2(lon_idx), self.embedding3(ldt_idx)
        
        cat_embeddings = torch.cat([self.view(lat_embed), self.view(lon_embed), self.view(ldt_embed)], dim=-2)
        cat_fields_res = torch.cat([fields, res1, res2, res3], dim=1)
        fields = self.conv5(cat_fields_res)
        final_map = torch.cat([fields, cat_embeddings], dim=-2)
        
        final = self.reshape(self.final_conv(final_map))
        return final
    
    def add_dim(self, x):
        return x.unsqueeze(1).unsqueeze(2)
        

def train(model, train_loader, val_loader, optimizer, criterion, device, epochs=10):
    hdisplay_img = display(display_id=True)
    hdisplay_txt = display(display_id=True)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    line1, = ax.plot(0,0, label='Train loss')
    line2, = ax.plot(0,0, label='Val loss')
    plt.legend()
    plt.close()

    def update(x, y_train, y_val, epoch):
        line1.set_xdata(x)
        line1.set_ydata(y_train)
        line2.set_xdata(x)
        line2.set_ydata(y_val)
        #ax.set_yscale('log')
        fig.canvas.draw()
        hdisplay_img.update(fig)
        hdisplay_txt.update(f"Epoch {epoch}")
        
    model.train()
    train_losses = []
    val_losses = []
    with trange(1, epochs + 1, desc='Training', unit='epoch') as t:
        for epoch in t:
            train_loss = 0
            val_loss = 0
            #start_time = time.time()
            with tqdm(train_loader, desc=f'Train epoch {epoch}',
              unit='batch', leave=False) as t1:
                for batch_idx, (fields, lat, lon, ldt, targets) in enumerate(t1):
                    fields, lat, lon, ldt, targets = fields.to(device), lat.to(device), lon.to(device),\
                                                            ldt.to(device), targets.to(device)
                    optimizer.zero_grad()
                    outputs = model(fields, lat, lon, ldt)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            with torch.no_grad():
                with tqdm(val_loader, desc=f'Val. epoch {epoch}',
                unit='batch', leave=False) as t2:
                    for batch_idx, (fields, lat, lon, ldt, targets) in enumerate(t2):
                        fields, lat, lon, ldt, targets = fields.to(device), lat.to(device), lon.to(device),\
                                                                ldt.to(device), targets.to(device)
                        outputs = model(fields, lat, lon, ldt)
                        loss = criterion(outputs, targets)
                        val_loss += loss.item()
            avg_val_loss = val_loss / len(val_loader)
            if epoch == 1:
                ax.set_xlim(0,epochs)
                ax.set_ylim(0,avg_val_loss*1.1)
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            update(range(1, len(train_losses)+1), train_losses, val_losses, epoch)
    return np.array(train_losses), np.array(val_losses)


def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        with tqdm(test_loader, desc='Test', unit='batch', leave=False) as t:
            for batch_idx, (fields, lat, lon, ldt, targets) in enumerate(t):
                fields, lat, lon, ldt, targets = fields.to(device), lat.to(device), lon.to(device),\
                                                        ldt.to(device), targets.to(device)
                outputs = model(fields, lat, lon, ldt)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
    avg_test_loss = test_loss / len(test_loader)
    return np.array(avg_test_loss)

