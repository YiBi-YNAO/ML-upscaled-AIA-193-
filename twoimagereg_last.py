#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 02:54:35 2021

@author: hong
"""



import numpy as np
from matplotlib import pyplot as plt
import FLAN_pho_sim_lib as sim
import os
import pickle
import math
plt.close('all')

#file = open('datalist.txt', 'r')

if 1==1:
           # text_line = file.readline()
        #if text_line:
            #  HiC_L1.5_v3_18.54.17.79.fits  AIA20120711_185420_0193.fits
            #  HiC_L1.5_v3_18.53.44.41.fits  AIA20120711_185344_0193.fits
            #  HiC_L1.5_v3_18.53.05.43.fits  AIA20120711_185308_0193.fits
            #HiC_L1.5_v3_18.55.30.15.fits  AIA20120711_185508_0193.fits
            
            # For train
            Hfile = '../4K-Level1.5/prep/prep_HiC_L1.5_v3_18.52.54.29.fits'
            filein = '../aia_193_contemporaneous/AIA20120711_185256_0193.fits'
            
            Hfile = '../4K-Level1.5/prep/prep_HiC_L1.5_v3_18.53.05.43.fits'
            filein = '../aia_193_contemporaneous/AIA20120711_185308_0193.fits'
            
            Hfile = '../4K-Level1.5/prep/prep_HiC_L1.5_v3_18.53.44.41.fits'
            filein = '../aia_193_contemporaneous/AIA20120711_185344_0193.fits'

            Hfile = '../4K-Level1.5/prep/prep_HiC_L1.5_v3_18.54.17.79.fits'
            filein = '../aia_193_contemporaneous/AIA20120711_185420_0193.fits'


            Hfile = '../4K-Level1.5/prep/prep_HiC_L1.5_v3_18.55.30.15.fits'
            filein = '../aia_193_contemporaneous/AIA20120711_185532_0193.fits'      
            #x1,x2,y1,y2=965,1690,1525,2215
            
            # Hi-c2.1
            
            #Hfile = '../hic2/hic2_iris/HiC2.1_L1.5_19.01.34.80_good.fits'
            #filein = '../hic2/AIA/aia.lev1.2018-05-29T190134Z.172930046.image_lev1.fits'
#            plt.close('all')
            #filein='AIA20120711_185320_0193.fits'
            x1,x2,y1,y2=900,1650,1400,2100
            SdoOrg,hS=sim.fitsread(filein)
            SdoOrg=SdoOrg[x1:x2,y1:y2]
            SdoOrg=sim.removenan(SdoOrg)
            
            low=SdoOrg.copy()
            Sdo=SdoOrg
            lr,ud=0,0
        
            #Highscal= 0.12899999
            K=0.1
            KG=0.8
            
            W=1
            
            #Hfile='HiC_L1.5_v3_18.53.22.13.fits'
            ##########################################3
            HighOrg,hH=sim.fitsread(Hfile)
            Highscal=hH['CDELT1']
            
            
            #################################################333
            Mask=np.zeros((HighOrg.shape[0],HighOrg.shape[1]))
            Mask[W:-W,W:-W]=1
            
            
            High=HighOrg.copy()
            
            if ud==1:
                High=High[::-1,:]
                HighOrg=HighOrg[::-1,:]
            if lr==1:
                High=High[:,::-1]
                HighOrg=HighOrg[:,::-1]
                
            #try:
             #   SDOscal=hS['CDELT1']
            #except:
          
            SDOscal=0.6  
            Sdo=sim.imresize(Sdo,SDOscal/0.15)
            SDOscal=0.15
            #Sdo=sim.imresize(Sdo,SDOscal/(Highscal*2))
           # SdoOrg=sim.imresize(SdoOrg,SDOscal/(Highscal*2))
            #SDOscal=Highscal*2
        
            #xc = hS['CRPIX1']
            #yc = hS['CRPIX2'] 
            #RSUN=hS['RSUN_OBS']/SDOscal
            #cen=[xc,yc]
            sc0= Highscal/SDOscal
            Sdo0=Sdo
            Sdo=High
            SdoOrg=High
            High=Sdo0
            #Sdo=sim.removenan(np.log(Sdo))
            #High=sim.removenan(np.log(High))
            
            
          
            Sdo,sHigh,sc= sim.per_data(Sdo,High,maskS=Sdo>0,maskH=(High>0),sc0=sc0,lr=lr,ud=ud,K=K)
         
            img1Out,img2Out,img3Out,H,status, src, dst=sim.siftImageAlignment(sHigh,Sdo,High.shape,debug=1
                                                                              ,mask1=np.int8(sHigh>0),mask2=Sdo>0,KG=KG,img2Org=SdoOrg)
            
            
            ######################################################33
            print(Highscal*H.scale,H.rotation,H.translation)
            Err=(H.residuals(status[0],status[1])**2)
            Perr=np.sqrt(Err.sum()/(len(Err)-3))*SDOscal
            print(Perr,len(Err))
            ##############################Create Calibrated Fits
            
            # ########################save GIF
#            frame=np.dstack((High,img2Out))
#            sim.create_gif(frame, filein+'.submap.gif')
#            frame=np.dstack((Sdo,img1Out))
#            sim.create_gif(frame, filein+'.map.gif')
            
#            plt.figure()
#            plt.imshow(img1Out)
#            plt.figure()
#            plt.imshow(Sdo)
            High0=High
            High=img2Out#[1:2000,1:1900]
            img2Out=High0#[1:2000,1:1900]  
            #igh[High <2]=2
            #High=np.log(High)
            #img2Out[img2Out <2]=2
            #img2Out=np.log(img2Out)
            plt.figure()
            #  for train  [100:1300,250:1300]
            #  for hi-c 2.1 [100:1300,:1300]
            
            sim.showim(img2Out,K=4)#  for train  [100:1300,250:1300]
            plt.figure()
            sim.showim(High,K=4)
            from skimage.transform import resize
            
            #img2Out=img2Out[200:1500,250:1550]
            #img3Out=img3Out[200:1500,250:1550]
            #High=High[200:1500,250:1550]
            high0=High
            im0=img2Out
            
            am1=np.mean(High,axis=0)
            a1=np.where(am1 !=0)
            #a1=np.where(am1 > High.min()+1)       
            am2=np.mean(High,axis=1)
            a2=np.where(am2 !=0)
            #a2=np.where(am2 > High.min()+1)
            a1=a1[0]
            a2=a2[0]
            print('here',High.shape)
            im02=High*1.0
            im01=img2Out*1.0
            #w1,w2,v1,v2=1467*2,1491*2,1908*2,1940*2

            sav=[]
            sav.append(low)
            sav.append(High)
            filesub=os.path.basename(Hfile)
            pickle.dump(sav, open('full_match_x4_box4_'+filesub+'.dat4', 'wb'))            

            #testing
            w1,w2,v1,v2=1052*2,1118*2,608*2,658*2
            lw1,lw2,lv1,lv2=int(w1/4),int(w2/4),int(v1/4),int(v2/4)
            im2=High[w1:w2,v1:v2]*1.0
            im1=low[lw1:lw2,lv1:lv2]*1.0
            High[w1:w2,v1:v2]=0
            low[lw1:lw2,lv1:lv2]=0

            #validation
            w1,w2,v1,v2=2004,2104,1068,1200
            lw1,lw2,lv1,lv2=int(w1/4),int(w2/4),int(v1/4),int(v2/4)
            im2v=High[w1:w2,v1:v2]*1.0
            im1v=low[lw1:lw2,lv1:lv2]*1.0
            High[w1:w2,v1:v2]=0
            low[lw1:lw2,lv1:lv2]=0
           
            sav=[]
            sav.append(low)
            sav.append(High)
            

            filesub=os.path.basename(Hfile)
            pickle.dump(sav, open('train_match_x4_box4_'+filesub+'.dat4', 'wb'))
            frame=np.dstack((sim.imresize(low,4,order=0),High))
            sim.create_gif(frame, 'train_match_x4_box4_'+filesub+'.gif')

            sav=[]
            sav.append(im1)
            sav.append(im2)
            pickle.dump(sav, open('test_match_x4_box4_'+filesub+'.dat4', 'wb'))
            frame=np.dstack((sim.imresize(im1,4,order=0),im2))
            sim.create_gif(frame, 'test_match_x4_box4_'+filesub+'.gif')
            
            sav=[]
            sav.append(im1v)
            sav.append(im2v)
            pickle.dump(sav, open('valid_match_x4_box4_'+filesub+'.dat4', 'wb'))
            frame=np.dstack((sim.imresize(im1v,4,order=0),im2v))
            sim.create_gif(frame, 'valid_match_x4_box4_'+filesub+'.gif')
