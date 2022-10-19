# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 08:43:35 2018

@author: jkf
"""
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import glob,os,random,pickle,sim
import numpy as np
from torch.utils import data
from scipy.stats import pearsonr as pr
from wdsr_b_wn import WDSR_B as NET

class FitsSet(data.Dataset): #daa for training
    def __init__(self,file):
        # 
        self.imgs=glob.glob(file)
    def __getitem__(self, index):
        img_path = self.imgs[index-1]
        dat = pickle.load(open(img_path,'rb')) #
        high=dat[1].astype('float32') #label image
        low=dat[0].astype('float32') #input image
        

       #Data Augmentation
        flip=random.randint(0,1)
        if flip == 1:
            high=high[:,::-1]
            low=low[:,::-1]

        flip=random.randint(0,3) #rotate the image with an angle of 90*n degree.
        high=np.rot90(high,flip)
        low=np.rot90(low,flip)
       
        h,w=low.shape
        #Randomly cut image blocks with imsize
        no0=0
        while no0 == 0:
         starth=0
         endh=h-imsize-1
         h0=random.randint(starth,endh)
         startw=0
         endw=w-imsize-1
         w0=random.randint(startw,endw)
         high0=high[h0*scale:(h0+imsize)*scale,w0*scale:(w0+imsize)*scale].astype('float32')
         mm=high0.min() 
         if mm>0:
             no0=1
             high=high0
             low=low[h0:h0+imsize,w0:w0+imsize].astype('float32')
             low_1=sim.imresize(low,scale)
             [dx,dy,cor]=sim.xcorrcenter(low_1,high)
             high=sim.immove(high,dx,dy)       
       
        it=0
        while abs(dx)>1 or abs(dy)>1:
            it+=1
            dx=abs(int(dx))
            dy=abs(int(dy))
            [dx,dy,cor]=sim.xcorrcenter(low_1[dx:-dx-1,dy:-dy-1],high[dx:-dx-1,dy:-dy-1])
            high=sim.immove(high,dx,dy)
            if it>10:
                break
#
            
        high=np.ascontiguousarray(high, dtype=np.float32) 
        low=np.ascontiguousarray(low, dtype=np.float32)

        high=torch.from_numpy(high).unsqueeze(0) 
        low=torch.from_numpy(low).unsqueeze(0)

        return low, high,img_path

    def __len__(self):
        return len(self.imgs)
    
class validFitsSet(data.Dataset): #data for validation set
    def __init__(self,file):
        self.imgs=glob.glob(file)
    def __getitem__(self, index):
        img_path = self.imgs[index-1]
        dat = pickle.load(open(img_path,'rb'))
        high=dat[1].astype('float32')
        low=dat[0].astype('float32')

        low_1=sim.imresize(low,scale)
        [dx,dy,cor]=sim.xcorrcenter(low_1,high)
        high=sim.immove(high,dx,dy)       
        it=0
        while abs(dx)>1 or abs(dy)>1:
            it+=1
            dx=abs(int(dx))
            dy=abs(int(dy))
            [dx,dy,cor]=sim.xcorrcenter(low_1[dx:-dx-1,dy:-dy-1],high[dx:-dx-1,dy:-dy-1])
            high=sim.immove(high,dx,dy)
            if it>10:
                break

        high=np.ascontiguousarray(high, dtype=np.float32)
        low=np.ascontiguousarray(low, dtype=np.float32)

        high=torch.from_numpy(high).unsqueeze(0)
        low=torch.from_numpy(low).unsqueeze(0)

        return low, high,img_path

    def __len__(self):
        return len(self.imgs)
    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

scale=4
# Hyper-parameters
n_resblocks=16
n_feats=64
n_colors=1
res_scale=1
batchsize=16
curr_lr=1.2e-3
learning_rate = 1.2e-3 
num_epochs=100000
imsize=48
K=0
train_dataset=FitsSet('./hicmatch_4x_aia_box4/train_match_x4_box4_*.dat4') #training set
valid_dataset=validFitsSet('./hicmatch_4x_aia_box4/valid_match_x4_box4_*.dat4') #validation set

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batchsize, 
                                           shuffle=True)
valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                           batch_size=batchsize, 
                                           shuffle=True)

netfile='wd_sdo2hic1.0_run7.mod' #name of model
parafile=netfile+'.train' #training record, including the value of loss function



try:
    para = pickle.load(open(parafile,'rb'))   
    epoch0=para[0]
    totloss=para[1]
    curr_lr=para[2]
except: 
    epoch0=0
    totloss=[]
    para=[epoch0, totloss, curr_lr]




model=NET(scale,n_resblocks,n_feats,n_colors,res_scale).to(device)   
model.eval() 

# loss function
criterion = nn.L1Loss() #
#criterion= nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def update_lr(optimizer, lr):  
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Train the model
total_step = len(train_loader) #batch for training
valid_step=len(valid_loader) #batch for validation

for epoch in range(epoch0,num_epochs):
    eploss=0
    torch.cuda.empty_cache() 
    
    for i, (images, labels, name) in enumerate(train_loader): 

        images = images.to(device) 
        labels = labels.to(device)
        

        outputs = model(images) 
        wd=5
        loss= criterion(outputs[:,:,wd:-wd,wd:-wd].squeeze(), labels[:,:,wd:-wd,wd:-wd].squeeze()) #计算损失函数，由于卷积的边界问题，所以扣除了边上的数据，但这个边界宽度我也没有谱
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        lossvalue=loss.item() #obtain the value of loss
        eploss+=lossvalue 

        if (i+1) % 1 == 0:    
                print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f} {:}" .format(epoch+1, num_epochs, i+1, total_step,lossvalue,name[0] ))

    Loss=[epoch,eploss/total_step]                                         

    if (epoch+1) % 100== 0: #for showing

        pre = outputs.cpu().detach().numpy()[:,:,wd:-wd,wd:-wd] #
        predicted=pre[0,0] #
        y_t=labels.cpu().detach().numpy()[:,:,wd:-wd,wd:-wd] #
        y_train=y_t[0,0]
        
        name= name[0]#


        VDI=images.cpu().detach().numpy()[0,0]
        VDI=sim.imresize(VDI,scale)
        VDI=VDI[wd:-wd,wd:-wd]

        mi=y_t.min() 
        mx=y_t.max()
        
      
        arr=(np.abs(y_train)<10000)*(np.abs(y_train)>0)
        S1=np.polyfit(y_train[arr].flatten(),predicted[arr].flatten(),1)
        P1=np.poly1d(S1) #
        pr1=pr(y_train.flatten(),predicted.flatten()) #
        

        
        delta=(predicted-y_train) #
        re=delta.std() #

        mtmp=np.hstack(((predicted),(y_train)))     
        mtmp=np.hstack((sim.zscore2(VDI),sim.zscore2(mtmp),sim.zscore2(delta))) #
        
    
        plt.figure(1)
        ax=plt.subplot(211)
        if K==0:
            dis=ax.imshow(mtmp,vmax=5,vmin=-3,cmap='gray', interpolation='bicubic')      
            plt.pause(0.1)
            plt.draw()
        else:
            dis.set_data(mtmp)
        K=1
        namesub=os.path.basename(name)

        plt.draw()

    
        ax2=plt.subplot(223)
        plt.cla()
        ax2.plot(y_t.flatten(),pre.flatten(),'.r',MarkerSize=1)
        
        plt.xlim((mi,mx))
        plt.ylim((mi,mx))
        
        
        plt.plot([mi,mx],[mi,mx],'-g')
        title=str(P1[1])[:5]+' '+str(pr1[0])[:5]
        plt.title(title)

        torch.cuda.empty_cache()   

     
        for i, (images, labels,name) in enumerate(valid_loader):
            vloss=0
            images = images.to(device)
            labels = labels.to(device)
              
            outputs = model(images)
    
            loss_valid= criterion(outputs[:,:,wd:-wd,wd:-wd].squeeze(), labels[:,:,wd:-wd,wd:-wd].squeeze())
            vloss+=loss_valid.item()
            images=images.cpu() 
            labels=labels.cpu()
            outputs=outputs.cpu()
            
        vloss/=valid_step  
        Loss.append(vloss)       
        totloss.append(Loss) 

     
        ax3=plt.subplot(224)
        plt.cla()

        tloss=np.array(totloss)
        plt.plot(tloss[0:,0],tloss[0:,1],'r',MarkerSize=1) 
        plt.plot(tloss[0:,0],tloss[0:,2],'b',MarkerSize=1) 

        plt.title(str(Loss[1])[:7]+'/'+str(Loss[2])[:7])
        plt.ylim((0,100)) 
        plt.pause(0.1)
        plt.draw() 
        
        #save the model
        torch.save(model, 'net_temporal_run7/'+ str(int(epoch+1)).zfill(6)+'_'+netfile)
        torch.save(model, netfile)
        para[0]=epoch
        para[1]=(totloss)
        para[2]=(curr_lr)
        pickle.dump(para, open(parafile, 'wb'))
    
    # change learning rate
    if (epoch+1) % 2000 == 0:
        curr_lr /= 1.1
        update_lr(optimizer, curr_lr)
