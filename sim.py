# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 16:05:28 2018

@author: jkf
pip install opencv-contrib-python==3.4.2.16
"""

#import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import warp, EuclideanTransform,SimilarityTransform, rotate,rescale
from skimage import transform
import scipy.fftpack as fft

def per_data(Sdo, High, maskH,maskS,sc0, lr, ud,K=0.5,sigma=[0,0,0,0]):
    from skimage import transform, filters
    import skimage.morphology as sm
#    High = filters.gaussian(imnorm(High), sigma=sc0*K)*maskH
    
    def mask3sig(im,mask,sig=[0,0]):
        
        med=np.median(im[mask])
        t=im[mask].std()
        maskH2=(im<(med+sig[0]*t)) | (im>(med+sig[1]*t))
#        maskH2=(im>(med+sig*t))
    
        mask=mask & maskH2
    
        return mask
    H=High[maskH]
    High=imnorm(High,mx=H.max(),mi=H.min())*maskH
    M = High.shape[0] // sc0
    N = High.shape[1] // sc0
    sHigh = transform.resize(High, (M, N),mode='reflect')
    
    smaskH= transform.resize(maskH*1.0, (M, N),mode='reflect')>0.9
    smaskH=sm.erosion(smaskH,sm.square(5)) 

    sHigh = filters.gaussian(sHigh, K)*smaskH
    
    maskS=mask3sig(Sdo,maskS,sigma[0:2])
    smaskH=mask3sig(sHigh,smaskH,sigma[2:4])

    H=sHigh[smaskH]
    sHigh = imnorm(sHigh,mx=H.max(),mi=H.min()) 

    S = imnorm(removenan(Sdo))
    
    S = filters.gaussian(S, 0.5)*maskS
    
    S = imnorm(S,mx=S[maskS].max(),mi=S[maskS].min())
    
    sHigh=sHigh*255
    S = S * 255
    sHigh = np.uint8(sHigh)
    S = np.uint8(S)
    
    sc0=0.5*(High.shape[0]*1.0/M+High.shape[1]*1.0/N)
    return S, sHigh,sc0


def siftImageAlignment(img1, img2, Hsize, debug=0, mask1=None, mask2=None,KG=0.75,scale=None,san=0.8,img2Org=None):
    from skimage.feature import plot_matches
    import sys
    from skimage.measure import ransac
    if scale is None:
        func=SimilarityTransform
    else:
        func=EuclideanTransform
    
    if img2Org is None: 
        img2Org=img2
    else:
        img2Org=imnorm(img2Org)
       
    def sift_kp(image, mask=None):
        if mask is not None:
            mask = np.uint8(mask * 255)
        sift = cv2.xfeatures2d_SIFT.create()
#        sift = cv2.xfeatures2d_SURF.create()
    
        kp, des = sift.detectAndCompute(image, mask)
        kp_image = cv2.drawKeypoints(image, kp, None)
        if debug==1:
            print(len(kp))
            plt.figure()
            plt.imshow(kp_image)
        return  kp, des
    
    
    def get_good_match(des1, des2,KG=0.75):
#        bf = cv2.BFMatcher()
#        matches = bf.knnMatch(des1, des2, k=2)
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=100)# or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(des1,des2,k=2)
        good = []
        for m, n in matches:
            if m.distance < KG * n.distance:
                good.append(m)
        return good

    kp1, des1 = sift_kp(img1, mask1)
    kp2, des2 = sift_kp(img2, mask2)

    goodMatch = get_good_match(des1, des2,KG=KG)
    if len(goodMatch) > 1:
        ptsA = np.float32([kp1[m.queryIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
        ptsB = np.float32([kp2[m.trainIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
    else:
        print('SORRY ! I cannot do it')
        sys.exit(0)

    src = np.squeeze(ptsA)
    dst = np.squeeze(ptsB)
    
    import pandas as pd
    tmp=np.hstack((src,dst))
    newdata=pd.DataFrame(tmp,columns=['A','B','C','D'])
    s=newdata.drop_duplicates(subset=['A','B','C','D'],keep='first')
    tmp=np.array(s)
    src=tmp[:,:2]
    dst=tmp[:,2:]
    
    src2=src[:,::-1]
    dst2=dst[:,::-1]
#    model_robust =func()
#    model_robust.estimate(src, dst)
#    inlier_idxs=range(len(src))
    # robustly estimate affine transform model with RANSAC
    model_robust, inliers = ransac((src, dst), func, min_samples=2,
                                   residual_threshold=san, max_trials=500)
    outliers = inliers == False

    # visualize correspondence
    inlier_idxs = np.nonzero(inliers)[0]
    outlier_idxs = np.nonzero(outliers)[0]
    
    if debug == 1:
        print(len(inliers))
        fig, ax = plt.subplots(nrows=2, ncols=1)

        plt.gray()

        plot_matches(ax[0], img1, img2, src2, dst2,
                     np.column_stack((inlier_idxs, inlier_idxs)), matches_color='b')
        ax[0].axis('off')
        ax[0].set_title('Correct correspondences')

        plot_matches(ax[1], img1, img2, src2, dst2,
                     np.column_stack((outlier_idxs, outlier_idxs)), matches_color='r')
        ax[1].axis('off')
        ax[1].set_title('Faulty correspondences')

        plt.show()
    
    if scale is None:
        tform = SimilarityTransform(scale=model_robust.scale, rotation=model_robust.rotation,
                                    translation=model_robust.translation)
    else:    
        tform = SimilarityTransform(scale=1,rotation=model_robust.rotation,translation=model_robust.translation)
        
    img1Out = warp(img1, tform.inverse, output_shape=(img2.shape[0], img2.shape[1]))
    img2Out = warp(img2Org, tform, output_shape=(img1.shape[0], img1.shape[1]))
#
    img2Out = transform.resize(img2Out, (Hsize[0], Hsize[1]),mode='reflect')


#    img2Out = transform.resize(img2Out, (Hsize[0], Hsize[1]),mode='reflect')
    status = (src[inlier_idxs], dst[inlier_idxs])

    return img1Out, img2Out, tform, status,src,dst


def fitswrite(fileout, im, header=None):
    from astropy.io import fits
    import os
    if os.path.exists(fileout):
        os.remove(fileout)
    if header is None:
        fits.writeto(fileout, im, output_verify='fix', overwrite=True, checksum=False)
    else:        
        fits.writeto(fileout, im, header, output_verify='fix', overwrite=True, checksum=False)


def fitsread(filein):
    from astropy.io import fits
    head = '  '
    hdul = fits.open(filein)

    try:
        data0 = hdul[0].data.astype(np.float32)
        head = hdul[0].header
    except:
        hdul.verify('silentfix')
        data0 = hdul[1].data
        head = hdul[1].header

    return data0, head

def removelimb(im, center=None, RSUN=None):
  #  pip install polarTransform
    import polarTransform as pT
    from scipy import signal

    radiusSize, angleSize = 1024, 1800
    im = removenan(im)
    im2=im.copy()
    if center is None:
        T = (im.max() - im.min()) * 0.2 + im.min()
        arr = (im > T)
        import scipy.ndimage.morphology as snm
        arr=snm.binary_fill_holes(arr)
#        im2=(im-T)*arr
        Y, X = np.mgrid[:im.shape[0], :im.shape[1]]
        xc = (X * arr).astype(float).sum() / (arr*1).sum()
        yc = (Y * arr).astype(float).sum() / (arr*1).sum()
        center = (xc, yc)
        RSUN = np.sqrt(arr.sum() / np.pi)

    Disk = np.int8(disk(im.shape[0], im.shape[1], RSUN * 0.95))
    impolar, Ptsetting = pT.convertToPolarImage(im, center, radiusSize=radiusSize, angleSize=angleSize)
    profile = np.median(impolar, axis=1)
    profile = signal.savgol_filter(profile, 11, 3)
    Z = profile.reshape(-1, 1).repeat(impolar.shape[1], axis=1)
#    im2 = removenan(im / Ptsetting.convertToCartesianImage(Z))-1
#    im2 = im2 * Disk
    im = removenan(im /Ptsetting.convertToCartesianImage(Z))
    im= im*Disk
    return im, center, RSUN, Disk,im


def imnorm(im, mx=0, mi=0):
    #   图像最大最小归一化 0-1
    if mx != 0 and mi != 0:
        pass
    else:
        mi, mx = np.min(im), np.max(im)

    im2 = removenan((im - mi) / (mx - mi))

    arr1 = (im2 > 1)
    im2[arr1] = 1
    arr0 = (im2 < 0)
    im2[arr0] = 0

    return im2


def removenan(im, key=0):
    """
    remove NAN and INF in an image
    """
    im2 = np.copy(im)
    arr = np.isnan(im2)
    im2[arr] = key
    arr2 = np.isinf(im2)
    im2[arr2] = key

    return im2


def showim(im):
    mi = np.max([im.min(), im.mean() - 3 * im.std()])
    mx = np.min([im.max(), im.mean() + 3 * im.std()])
    if len(im.shape) == 3:
        plt.imshow(im, vmin=mi, vmax=mx)
    else:
        plt.imshow(im, vmin=mi, vmax=mx, cmap='gray')

    return


def zscore2(im):
    im = (im - np.median(im)) / im.std()
    return im


def disk(M, N, r0):
    X, Y = np.meshgrid(np.arange(int(-(N / 2)), int(N / 2)), np.linspace(-int(M / 2), int(M / 2) - 1, M))
    r = (X) ** 2 + (Y) ** 2
    r = (r ** 0.5)
    im = r < r0
    return im


#def fgauss(M, N, I, x0, y0, r):
#    # 产生高斯图像
#
#    r = r * r * 2
#    x = np.arange(0, M)
#    x = x - M / 2 + x0 - 1
#    y = np.arange(0, N)
#    y = y - N / 2 + y0 - 1
#    w1 = np.exp(-x ** 2 / r)
#    w2 = np.exp(-y ** 2 / r)
#    w2 = np.reshape(w2, (-1, 1))
#    f = I * w1 * w2
#    return f
#
#
#def showmesh(im):
#    X, Y = np.mgrid[:im.shape[0], :im.shape[1]]
#    from mpl_toolkits.mplot3d import Axes3D
#    figure = plt.figure('mesh')
#    axes = Axes3D(figure)
#
#    axes.plot_surface(X, Y, im, cmap='rainbow')
#    return


def create_gif(images, gif_name, duration=1):
    import imageio
    frames = []
    # Read
    T = images.shape[2]
    for i in range(T):
        frames.append(np.uint8(imnorm(images[:, :, i]) * 255))
    #    # Save
    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)

    return


#def immove(image, dx, dy):
#    """
#    image shift by subpix
#    """
#    # The shift corresponds to the pixel offset relative to the reference image
#    from scipy.ndimage import fourier_shift
#    if dx == 0 and dy == 0:
#        offset_image = image
#    else:
#        shift = (dx, dy)
#        offset_image = fourier_shift(fft.fft2(image), shift)
#        offset_image = np.real(fft.ifft2(offset_image))
#
#    return offset_image

def fulldisk2(im, scal,rot,center=None,size=[4096,4096]):

    im=removenan(im)
    im2=im.copy()
    cen=np.array(imcenterpix(im)) 
    if center is None:
        T = (im.max() - im.min()) * 0.2 + im.min()
        arr = (im > T)
#        im2=(im-T)*arr
        Y, X = np.mgrid[:im.shape[0], :im.shape[1]]
        xc = (X * arr).astype(float).sum() / (arr*1).sum()
        yc = (Y * arr).astype(float).sum() / (arr*1).sum()
        center = [xc, yc]
        RSUN = np.sqrt(arr.sum() / np.pi)

    xc=center[0]
    yc=center[1]

#    im2=immove2(im2,-xc+cen[0],-yc+cen[1])
#    im2=imrotate(im2,rot)
#    im2 = imresize(im2, scal)
          

    shift=[-xc+cen[0],-yc+cen[1]]
    im2=imtransform4096(im2,scale=scal,rot=rot,translation=shift)
    
    cen=imcenterpix(im2)

    size2=(np.array(size)+1)//2
    im2=im2[cen[1]-size2[0]:cen[1]+size2[0],cen[0]-size2[1]:cen[0]+size2[1]]
    

    return im2,center

def fixSDO(Sdo, hS,Disk=None):
    if Disk is None: Disk=1
    M,N=Sdo.shape
    xc = hS['CRPIX1']
    yc = hS['CRPIX2']

    rot = hS['CROTA2']
    shift = [M//2 - xc, N//2 - yc]

    tform = SimilarityTransform(translation=shift)
#    Sdo = imnorm(removenan(Sdo)*Disk)
#    Sdo2 = warp(Sdo, tform, output_shape=(Sdo.shape[0], Sdo.shape[1]))
#    Sdo2 = rotate(Sdo2, -rot,mode='reflect')
    Sdo2=immove2(Sdo,shift[0],shift[1])
    Sdo2=imrotate(Sdo2,-rot)
    return Sdo2



def imrotate(im,rot):
    im2,para=array2img(im)
    im2=rotate(im2,rot,mode='reflect')
    im2=img2array(im2,para)
    return im2

def immove2(im,dx=0,dy=0):
    im2,para=array2img(im)
    tform = SimilarityTransform(translation=(dx,dy))
    im2 = warp(im2, tform.inverse, output_shape=(im2.shape[0], im2.shape[1]),mode='reflect')
    im2=img2array(im2,para)
    return im2

def imresize(im,scale):
    im2,para=array2img(im)
    im2=rescale(im2,scale,mode='reflect',order=3)
    im2=img2array(im2,para)
    return im2

def array2img(im):
    Bzero=im.min()
    mx=im.max()
    Bscale=mx-Bzero
    im2=(im-Bzero)/Bscale
    para=(Bzero,Bscale)
    return im2,para

def img2array(im,para):
    im2=im*para[1]+para[0]
    return im2

def imcenterpix(im):
    X0=(im.shape[0]+1)//2
    Y0=(im.shape[1]+1)//2
    cen=(X0,Y0)
    return cen

def imtransform(im,scale=1,rot=0,translation=[0,0]):
    im2=im.copy()
    im2,para=array2img(im2)
    tform = SimilarityTransform(translation=translation)
    im2 = warp(im2, tform.inverse, output_shape=(im2.shape[0], im2.shape[1]),mode='reflect')

    im2=rotate(im2,rot,mode='reflect')
    im2=rescale(im2,scale,mode='reflect')


    im2=img2array(im2,para)
    return im2   

def imtransform4096(im,scale=1,rot=0,translation=[0,0]):
    im2=im.copy()
    im2,para=array2img(im2)
    tform = SimilarityTransform(translation=translation)
    im2 = warp(im2, tform.inverse, output_shape=(im2.shape[0], im2.shape[1]),mode='reflect')

    im2=rotate(im2,rot,mode='reflect')
    im2=resize(im2,[4096,4096],mode='reflect')


    im2=img2array(im2,para)
    return im2               
def fulldisk(im, scal,rot,center=None,size=[4096,4096],Disk=None):
    if Disk is None: Disk=np.ones(size)>0
    im0=np.zeros(size)
    im=removenan(im)
    im2=im.copy()
    cen=np.array(imcenterpix(im)) 
    if center is None:
        T = (im.max() - im.min()) * 0.2 + im.min()
        arr = (im > T)
#        im2=(im-T)*arr
        Y, X = np.mgrid[:im.shape[0], :im.shape[1]]
        xc = (X * arr).astype(float).sum() / (arr*1).sum()
        yc = (Y * arr).astype(float).sum() / (arr*1).sum()
        center = [xc, yc]
        RSUN = np.sqrt(arr.sum() / np.pi)

    xc=center[0]
    yc=center[1]

#    im2=immove2(im2,-xc+cen[0],-yc+cen[1])
#    im2=imrotate(im2,rot)
#    im2 = imresize(im2, scal)
          

    
    shift=[-xc+cen[0],-yc+cen[1]]
    im2=imtransform(im2,scale=scal,rot=rot,translation=shift)
    
    cen=imcenterpix(im2)

    size2=(np.array(size)+1)//2

    if im2.shape[0]<size[0]:
        im0[size2[0]-cen[1]:size2[0]-cen[1]+im2.shape[0],-cen[0]+size2[1]:-cen[0]+size2[1]+im2.shape[1]]=im2
        im2=im0
    else:

        im2=im2[cen[1]-size2[0]:cen[1]+size2[0],cen[0]-size2[1]:cen[0]+size2[1]]

    

#    im2=removenan(np.log(im2))
#    mx=im2[Disk].max()
#    mi=im2[Disk].min()
#    im2 = imnorm(removenan(im2),mx=mx,mi=mi)
    return im2,center

def xcorrcenter(standimage, compimage, R0=2, flag=0):
    # if flag==1,standimage 是FFT以后的图像，这是为了简化整数象元迭代的运算量。直接输入FFT以后的结果，不用每次都重复计算
    try:
        M, N = standimage.shape

        standimage = zscore2(standimage)
        s = fft.fft2(standimage)

        compimage = zscore2(compimage)
        c = np.fft.ifft2(compimage)

        sc = s * c
        im = np.abs(fft.fftshift(fft.ifft2(sc)))  # /(M*N-1);%./(1+w1.^2);
        cor = im.max()
        if cor == 0:
            return 0, 0, 0

        M0, N0 = np.where(im == cor)
        m, n = M0[0], N0[0]

        if flag:
            m -= M / 2
            n -= N / 2
            # 判断图像尺寸的奇偶
            if np.mod(M, 2): m += 0.5
            if np.mod(N, 2): n += 0.5

            return m, n, cor
        # 求顶点周围区域的最小值
        immin = im[(m - R0):(m + R0 + 1), (n - R0):(n + R0 + 1)].min()
        # 减去最小值
        im = np.maximum(im - immin, 0)
        # 计算重心
        x, y = np.mgrid[:M, :N]
        area = im.sum()
        m = (np.double(im) * x).sum() / area
        n = (np.double(im) * y).sum() / area
        # 归算到原始图像
        m -= M / 2
        n -= N / 2
        # 判断图像尺寸的奇偶
        if np.mod(M, 2): m += 0.5
        if np.mod(N, 2): n += 0.5
    except:
        print('Err in align_Subpix routine!')
        m, n, cor = 0, 0, 0
    return m, n, cor

def immove(image, dx, dy):
    """
    image shift by subpix
    """
    # The shift corresponds to the pixel offset relative to the reference image
    from scipy.ndimage import fourier_shift
    if dx == 0 and dy == 0:
        offset_image = image
    else:
        shift = (dx, dy)
        offset_image = fourier_shift(fft.fft2(image), shift)
        offset_image = np.real(fft.ifft2(offset_image))

    return offset_image