#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 14:42:40 2019

@author: rain
"""

import os
from scipy import io
import numpy as np
from numpy import dot as dot
from numpy import mod as mod
from numpy import cross as cross
from numpy.linalg import norm as norm
from numpy import linalg as LA
from numpy import sqrt as sqrt
import math
import copy
import mayavi.mlab as mlab
from sklearn.preprocessing import normalize
from time import time, sleep
import cupy as cp

from Voxel import *

degree2radian =  math.pi / 180
NumChannels = 5
Channels4AE = [0,1,2]

# specifications of Velodyne-64
nLines = 64
AzimuthView = 360 * degree2radian
AzimuthResolution = 0.20 * degree2radian  # the original resolution is 0.18
VerticalViewDown = -24.8 * degree2radian
VerticalViewUp = 2.0 * degree2radian
VerticalResolution = (VerticalViewUp - VerticalViewDown) / (nLines - 1)
VerticalPixelsOffset = -VerticalViewDown / VerticalResolution
VisibleRange = 100  # manually set (m)


# parameters for projection image
SafeEdgeWidth4Top = 5
ImgH = nLines + SafeEdgeWidth4Top
ImgW = int(AzimuthView / AzimuthResolution)
ImgBottomLine = ImgH - VerticalPixelsOffset
CropWidth_SphericalRing = 8


nLeastPtsInOnePatch = 1

AllPixelIndexList = []
for iX in range(ImgH):
    for iY in range(ImgW):
        AllPixelIndexList.append([iX,iY])
AllPixelIndexes_WithoutWindowEdge = []
Size4FilterTopEdge = 8
for iX in range(Size4FilterTopEdge, nLines-Size4FilterTopEdge,1):
    for iY in range(Size4FilterTopEdge, ImgW-CropWidth_SphericalRing-Size4FilterTopEdge, 1):
        AllPixelIndexes_WithoutWindowEdge.append([iX,iY])
AllPixelIndexes_WithoutWindowEdge = np.array(AllPixelIndexes_WithoutWindowEdge, dtype=np.int32)


lPixelIndex3_WithoutEdge = []
IMGH3 = 16
IMGW3 = 448
Edge3 = 1
for iX in range(Edge3, IMGH3-Edge3,1):
    for iY in range(Edge3, IMGW3-Edge3, 1):
        lPixelIndex3_WithoutEdge.append([iX,iY])
aPixelIndex3_WithoutEdge = np.array(lPixelIndex3_WithoutEdge, dtype=np.int32)


def ProjectPC2SphericalRing(PC):
    assert PC.shape[0] > 3 and PC.shape[1] == 4 
    Image_float = np.zeros((ImgH, ImgW, NumChannels), dtype=np.float32)
    GridCounter = np.zeros((ImgH,ImgW),dtype=np.int32)
    
    rs = LA.norm(PC[:,0:3], axis=1)
    if min(rs) == 0:
        PC = PC[rs>0, :]
        rs = rs[rs>0]
    for iPt in range(PC.shape[0]):
        x = PC[iPt,0]
        y = PC[iPt,1]
        z = PC[iPt,2]
        r = rs[iPt]
        iCol = int((math.pi - math.atan2(y,x)) / AzimuthResolution)     # alpha
        beta = math.asin(z/r)                                           # beta
        iRow = ImgH - int(beta / VerticalResolution + VerticalPixelsOffset)
        if iRow < 0 or iRow >= ImgH:
            continue
        Image_float[iRow, iCol, 0:4] = PC[iPt, 0:4]
        Image_float[iRow, iCol, 4] = r
        GridCounter[iRow, iCol] += 1
    return Image_float, GridCounter


def LocateKeyPixels(NormImage, WindowRadius, AvlPixelList):
    WindowRadius_ = WindowRadius + 1
    
    KeyPixelList = []
    
    for iPixel in range(AvlPixelList.shape[0]):
        iX = AvlPixelList[iPixel,0]
        iY = AvlPixelList[iPixel,1]
    
        patch = NormImage[iX-WindowRadius:iX+WindowRadius_, iY-WindowRadius:iY+WindowRadius_]
        
        maxVal = np.max(patch)
        
        if NormImage[iX,iY] >= maxVal:
            KeyPixelList.append([iX,iY])
        
    return KeyPixelList
    
    
    

def GetKeyPtsByAE(SphericalRing, GridCounter, RespondImg):
    WindowSize = 5
    WindowRadius = int(WindowSize/2)
    WindowRadius_ = WindowRadius + 1
    radiusX = 2
    radiusY = 2
    
    t0=time()
    DiffImg = np.zeros((SphericalRing.shape[0],SphericalRing.shape[1]),dtype=np.float32)
    # DiffImg = cp.zeros((SphericalRing.shape[0],SphericalRing.shape[1]),dtype=np.float32)
    
    # nFixedKeyPts = 1024
    nFixedKeyPts = 1024
    
    NormDiffThreshold = 0.2
#    NormDiffThreshold = 0.5
#    NormDiffThreshold = 20
    PlanarThreshold = 0.4
    
#    # Normalize the respond image
#    RespondImgNorm = LA.norm(RespondImg, axis=2).reshape(RespondImg.shape[0],RespondImg.shape[1],1)
#    RespondImgNorm_ = np.tile(RespondImgNorm,(1,1,RespondImg.shape[2]))
#    RespondImg = RespondImg/RespondImgNorm_
    
    # prepare for the differ images (it is faster than process on each pixel; like down 3.5s to 1.8s)
    RespondImg = cp.array(RespondImg)
    GridMask = cp.array(GridCounter>0, dtype=cp.int32)
    ImgH_RespondImg = RespondImg.shape[0]
    ImgW_RespondImg = RespondImg.shape[1]
    CropedRespondImg = RespondImg[WindowRadius:ImgH_RespondImg-WindowRadius, WindowRadius:ImgW_RespondImg-WindowRadius]
    nWindowArea = WindowSize*WindowSize
    # NeighborFeatureDiffs = np.zeros((RespondImg.shape[0], RespondImg.shape[1], nWindowArea, RespondImg.shape[2]), dtype=np.float32)
    NeighborFeatureDiffs = cp.zeros((RespondImg.shape[0], RespondImg.shape[1], nWindowArea, RespondImg.shape[2]), dtype=np.float32)
    NeighborMaskMap = cp.zeros((RespondImg.shape[0], RespondImg.shape[1], nWindowArea), dtype=np.int32)
    # get diffs map
    # windowModel = np.zeros((WindowSize, WindowSize), dtype=np.int32)
    windowModel = cp.zeros((WindowSize, WindowSize), dtype=np.int32)
    for iNeighbor in range(nWindowArea):
        iOffsetX, iOffsetY = np.unravel_index(iNeighbor, windowModel.shape)
        iOffsetX_ = iOffsetX - WindowRadius
        iOffsetY_ = iOffsetY - WindowRadius
        diffImg = RespondImg[WindowRadius+iOffsetX_:ImgH_RespondImg-WindowRadius+iOffsetX_, 
                             WindowRadius+iOffsetY_:ImgW_RespondImg-WindowRadius+iOffsetY_] - CropedRespondImg
        NeighborFeatureDiffs[WindowRadius:RespondImg.shape[0]-WindowRadius, WindowRadius:RespondImg.shape[1]-WindowRadius, iNeighbor, :] = diffImg                
        cropedMask = GridMask[WindowRadius+iOffsetX_:ImgH_RespondImg-WindowRadius+iOffsetX_, 
                             WindowRadius+iOffsetY_:ImgW_RespondImg-WindowRadius+iOffsetY_]
        NeighborMaskMap[WindowRadius:RespondImg.shape[0]-WindowRadius, WindowRadius:RespondImg.shape[1]-WindowRadius, iNeighbor] = cropedMask
    NeighborDiffs = LA.norm(NeighborFeatureDiffs, axis=-1)
    
    # copy self mask out before set to zero
    selfIndex = int((windowModel.shape[0]*windowModel.shape[1]-1)/2)
    SelfMask = copy.deepcopy(cp.squeeze(NeighborMaskMap[:,:,selfIndex]))
    SelfMask[0:AllPixelIndexes_WithoutWindowEdge[:,0].min(),:] = 0
    SelfMask[AllPixelIndexes_WithoutWindowEdge[:,0].max()+1:SelfMask.shape[0],:] = 0
    SelfMask[:,0:AllPixelIndexes_WithoutWindowEdge[:,0].min()] = 0
    SelfMask[:,AllPixelIndexes_WithoutWindowEdge[:,0].max()+1:SelfMask.shape[0]] = 0
    
    # set self mask as 0 first
    NeighborMaskMap[:,:,selfIndex] = 0
    
    # get the valid neighbordiffs
    NeighborDiffs = NeighborDiffs + ((1-NeighborMaskMap)*1e10)
    
    # # set pixels themselfs as a big number
    # NeighborDiffs[:,:,selfIndex] = 1e10
    
    # get the minDiff with their neighbors
    MinDiffMap = cp.min(NeighborDiffs, axis=-1)
    
    # count the number of valid neighbors
    CountNeighbors = cp.sum(NeighborMaskMap, axis=-1)
    CountNeighbors = SelfMask*CountNeighbors
    
    # mask out the grids that have a few neighbors
    Mask4SparseGrids = cp.array(CountNeighbors >= 5, dtype=cp.int32)
    
    # remove the masked grids out
    MinDiffMap_ = MinDiffMap*Mask4SparseGrids
    
    # sorting
    MinDiffMap_1D = MinDiffMap_.flatten()
    # MinDiffMap_1D = MinDiffMap_1D * cp.array(MinDiffMap_1D>NormDiffThreshold, dtype=cp.int32)
    candidates_indices = cp.argsort(MinDiffMap_1D)
    
    
    distances = LA.norm(SphericalRing[0:nLines,0:ImgW-CropWidth_SphericalRing], axis=-1)
    distanceMask = cp.array(distances >= VisibleBottom, dtype=cp.int32)    
    MinDiffMap_Mask = cp.array(MinDiffMap_ > NormDiffThreshold, dtype=cp.int32)
    finalMask = distanceMask*MinDiffMap_Mask
    
    candidates_indices_2D = np.unravel_index(cp.asnumpy(candidates_indices), MinDiffMap_.shape)
    finalMask_1D = cp.array(finalMask[candidates_indices_2D], dtype=cp.bool)
    
    candidates_indices_ = candidates_indices[finalMask_1D]
    candidates_indices_2D_ = np.unravel_index(cp.asnumpy(candidates_indices_), MinDiffMap_.shape)
    
    
    KeyPts = SphericalRing[candidates_indices_2D_[0],candidates_indices_2D_[1],:]
    KeyPts = KeyPts[-nFixedKeyPts-1:-1,0:3]
    
    KeyPixels = []
    PlanarPts = []
    
    t1=time()
    
    # KeyPts = []
    # for i in range(candidates_indices.shape[0]-1,-1,-1):
    #     if finalMask_1D[i] < 1:
    #         continue
    #     x = candidates_indices_2D[0][i]
    #     y = candidates_indices_2D[1][i]
    #     KeyPts.append(SphericalRing[x,y,0:3])
    #     if len(KeyPts) >= nFixedKeyPts:
    #         break
    #     # KeyPixels.append([x,y])    
    # KeyPts = np.array(KeyPts, dtype=np.float32)
    
    
#     # get the key points
#     aAllDiffs = np.zeros((AllPixelIndexes_WithoutWindowEdge.shape[0], 6), dtype=np.float32)
#     # aAllDiffs = cp.zeros((AllPixelIndexes_WithoutWindowEdge.shape[0], 6), dtype=np.float32)
#     cntKeyPts = 0
#     for iPixel in range(AllPixelIndexes_WithoutWindowEdge.shape[0]):
#         iX = AllPixelIndexes_WithoutWindowEdge[iPixel,0]
#         iY = AllPixelIndexes_WithoutWindowEdge[iPixel,1]
#         if GridCounter[iX,iY] < 1:
#             continue
#         oneDiffs = NeighborDiffs[iX, iY, :]
#         oneMask = copy.deepcopy(GridCounter[iX-WindowRadius:iX+WindowRadius_, iY-WindowRadius:iY+WindowRadius_])
#         oneMask[WindowRadius, WindowRadius] = 0
        
#         diffs = oneDiffs[(oneMask>0).flatten()]
#         if diffs.shape[0] < 5:
# #        if diffs.shape[0] < 25:
#             continue
#         minDiff = min(diffs)
#         maxDiff = max(diffs)
        
#         DiffImg[iX,iY] = minDiff        
# #        minDiff = minDiff*diffs.shape[0]

#         if  minDiff > NormDiffThreshold:
#             pt = SphericalRing[iX, iY, 0:3]
#             if LA.norm(pt) < VisibleBottom:
#                 continue
#             aAllDiffs[cntKeyPts, 0] = minDiff
#             aAllDiffs[cntKeyPts, 1:4] = pt
#             aAllDiffs[cntKeyPts, 4:6] = [iX,iY]
#             cntKeyPts += 1            
            
#         # if minDiff < PlanarThreshold:
#         #     oneWindow = SphericalRing[iX-radiusX:iX+radiusX+1, iY-radiusY:iY+radiusY+1, 0:3]
#         #     pts = oneWindow[oneMask>0, :]
#         #     covMat = np.cov(pts, rowvar=0)
#         #     eigVals, eigVector = np.linalg.eig(covMat)
#         #     sortIdx = np.argsort(eigVals)            
#         #     vNorm = eigVector[:,sortIdx[0]]
#         #     if abs(vNorm[2]) > 0.9:
#         #         PlanarPts.append((np.c_[SphericalRing[iX, iY, 0:3].reshape(1,3), vNorm.reshape(1,3)]).flatten())
                
    
#     nFinalKeyPts = min(cntKeyPts, nFixedKeyPts)    
#     aAllDiffs = aAllDiffs[aAllDiffs[:,0].argsort()]
#     KeyPts = aAllDiffs[aAllDiffs.shape[0]-nFinalKeyPts:aAllDiffs.shape[0], 1:4]
#     KeyPixels = np.array(aAllDiffs[aAllDiffs.shape[0]-nFinalKeyPts:aAllDiffs.shape[0], 4:6], dtype=np.int32)
    
    
    PlanarPts = np.array(PlanarPts, dtype=np.float32)
    assert KeyPts.shape[0] > 50 #and PlanarPts.shape[0] > 1000
    
    t2=time()
    print(round(t1-t0, 4), 's, cupy time')
    print(round(t2-t1, 4), 's, for loop time')
    return KeyPts, KeyPixels, PlanarPts#, DiffImg


def ExtendKeyPtsInShpericalRing(SphericalRing, GridCounter, KeyPixels):
    nNeighborRadius = 6
    nNeighborRadius_ = nNeighborRadius + 1
    distThreshold = 5.0  # meter
    
    ExtendedKeyPts = np.zeros((1,3), dtype=np.float32)
    for iPixel in range(KeyPixels.shape[0]):
        iX = KeyPixels[iPixel,0]
        iY = KeyPixels[iPixel,1]
        # extract neighbors
        oneNeighbors = SphericalRing[iX-nNeighborRadius:iX+nNeighborRadius_, iY-nNeighborRadius:iY+nNeighborRadius_, 0:3]
        oneMask = GridCounter[iX-nNeighborRadius:iX+nNeighborRadius_, iY-nNeighborRadius:iY+nNeighborRadius_]
        pts = oneNeighbors[oneMask>0]
        oneMask[:] = 0  # set zeros using address
        
#        # filter out the far pts
#        pt = SphericalRing[iX,iY,:]
#        dists = LA.norm((pts-pt), axis=1)
#        pts = pts[dists<distThreshold, :]
        
        ExtendedKeyPts = np.r_[ExtendedKeyPts, pts]
        
    ExtendedKeyPts = np.delete(ExtendedKeyPts, 0, axis=0)
    return ExtendedKeyPts




def ProjectPC2RangeImage(PC):
    assert PC.shape[0] > 3 and PC.shape[1] == 3    
    Image_float = np.zeros((ImgH,ImgW),dtype=np.float32)
    for iPt in range(PC.shape[0]):
        x = PC[iPt,0]
        y = PC[iPt,1]
        z = PC[iPt,2]
        r = math.sqrt(x*x + y*y +z*z)
        iCol = int((math.pi - math.atan2(y,x)) / AzimuthResolution)  # alpha
        iRow = ImgH - int(math.asin(z/r) / VerticalResolution + VerticalPixelsOffset)  # beta
        if iRow < 0 or iRow >= ImgH:
            continue
        Image_float[iRow, iCol] = r
    return Image_float


def ProjectPixel2Pt(iRow, iCol, value):
    beta = (ImgBottomLine - iRow)*VerticalResolution
    z = value*math.sin(beta)    
    alpha = math.pi - iCol*AzimuthResolution
    r_ = value*math.cos(beta)
    x = r_*math.cos(alpha)
    y = r_*math.sin(alpha)    
    return x, y, z


def ProjectImage2PC(Image):
    PC = []    
    for iRow in range(ImgH):
        for iCol in range(ImgW):
            if sum(Image[iRow,iCol,:]) == 0:
                continue
            pt = Image[iRow, iCol, :]
            PC.append(pt)    
    PC = np.array(PC, dtype=np.float32)
    return PC

def SphericalRing2PCWithScoreColor(Image, ScoreMap):
    PC = []   
    colors = []
    for iRow in range(ImgH):
        for iCol in range(ImgW):
            if sum(Image[iRow,iCol,:]) == 0:
                continue
            pt = Image[iRow, iCol, :]
            PC.append(pt)
            colors.append(ScoreMap[iRow, iCol])
    PC = np.array(PC, dtype=np.float32)
    colors = np.array(colors, dtype=np.float32)
    colors = colors/np.max(colors)
    return PC, colors

def SphericalRing2PCWithNorm(Image, NormMap):
    PC = []   
    Norms = []
    for iRow in range(ImgH):
        for iCol in range(ImgW):
            if sum(Image[iRow,iCol,:]) == 0:
                continue
            pt = Image[iRow, iCol, :]
            PC.append(pt)
            Norms.append(NormMap[iRow, iCol, :])
    PC = np.array(PC, dtype=np.float32)
    Norms = np.array(Norms, dtype=np.float32)
    return PC, Norms


def GetKeyPtsFromRawFileName(rawFileFullPath, RespondLayer):    
    DataFolderName = 'SphericalRing'
    
    baseDir=os.path.dirname(os.path.dirname(rawFileFullPath))
    
    SphericalRingDir=os.path.join(baseDir,DataFolderName)
    DataFileName = os.path.basename(rawFileFullPath)+'.mat'
    DataFullPath = os.path.join(SphericalRingDir, DataFileName)
    
    t0=time()
    mat = io.loadmat(DataFullPath)
    SphericalRing = mat['SphericalRing']
    GridCounter = mat['GridCounter']
    
    # process
    t1=time()
    SphericalRing_ = SphericalRing[0:nLines, 0:ImgW-CropWidth_SphericalRing,Channels4AE]
    SphericalRing_ = SphericalRing_.reshape(1, SphericalRing_.shape[0], SphericalRing_.shape[1], SphericalRing_.shape[2])
    RespondImg = RespondLayer.predict(SphericalRing_)
    RespondImg = np.squeeze(RespondImg)
    t2=time()
    print(round(t1-t0, 4), 's, data loading time')
    print(round(t2-t1, 4), 's, predicting time')
    
    
    KeyPts, KeyPixels, PlanarPts = GetKeyPtsByAE(SphericalRing, GridCounter, RespondImg)
    
    return KeyPts, KeyPixels, PlanarPts

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    