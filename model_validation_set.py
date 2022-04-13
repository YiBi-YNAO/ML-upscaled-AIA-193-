#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 15:54:04 2019

@author: bi
"""

from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import glob,os,random,pickle,sim

import numpy as np
from torch.utils import data
from scipy.stats import pearsonr as pr
import FLAN_pho_sim_lib as jlib
from wdsr_b_wn import WDSR_B as NET

class FitsSet(data.Dataset): 
    def __init__(self,file):

        self.imgs=glob.glob(file)
    def __getitem__(self, index):
        img_path = self.imgs[index-1]
        dat = pickle.load(open(img_path,'rb'))
        high=dat[1].astype('float32') #label image
        low=dat[0].astype('float32') #input image
        #low=sim.imresize(low,1/scale)


        high=np.ascontiguousarray(high, dtype=np.float32)
        low=np.ascontiguousarray(low, dtype=np.float32)
        
        high=torch.from_numpy(high).unsqueeze(0)
        low=torch.from_numpy(low).unsqueeze(0)

        return low, high,img_path

    def __len__(self):
        return len(self.imgs)
    
# Device configuration
device = 'cpu'#torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
scale=4
n_resblocks=16
n_feats=64
n_colors=1
block_feats=512
res_scale=1
batchsize=10
netfile='../ML-upscaled-AIA-193--main/net_temporal_run4/30000_wd_sdo2hic1.0_run4.mod' #name of model 
parafile=netfile+'.train'  #training record, including the value of loss function
Vfile='noresize_test_prep_HiC_L1.5_v3_18.55.30.15.fits.dat4'
train_dataset=FitsSet(Vfile) 
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batchsize, 
                                           shuffle=True)

model_state_dict = torch.load(netfile,map_location='cpu').state_dict()

model=NET(scale,n_resblocks,n_feats,n_colors,res_scale).to(device)   
model.load_state_dict(model_state_dict)
model.eval() 
torch.cuda.empty_cache() 

plt.close('all')
dat = pickle.load(open(Vfile,'rb'))
#low=data[0]
wd=5
wd=1
for i, (images, labels, name) in enumerate(train_loader):
      j=i
      #images=sim.imresize(low,1/scale)
images = images.to(device)
outputs = model(images)
pre = outputs.cpu().detach().numpy()[:,:,wd:-wd,wd:-wd] #
predicted=pre[0,0] #
plt.close('all')
plt.figure(1)
jlib.showim(predicted,K=4)

img2Out=jlib.imresize(dat[0],4)
high=dat[1]
im1=img2Out[1:-1,1:-1]

im2=predicted
#im[im<10]=10
#im2=np.log(im)

im3=high[1:-1,1:-1]
#im[im<10]=10
#im3=np.log(im)
frame=np.dstack((im1,im2,im3))

filesub=os.path.basename(Vfile)

sim.create_gif(frame, filesub+'.gif')

sav=[]
sav.append(img2Out[wd:-wd-1,wd:-wd-1])
sav.append(high[wd:-wd-1,wd:-wd-1])
sav.append(predicted[1:,1:])
sav=np.array(sav)
#pickle.dump(sav,open(filesub+'_L1.dat','wb'))
jlib.fitswrite(filesub+'_run4.fits', sav,header=None)