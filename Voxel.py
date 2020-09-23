#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 11:29:00 2019

@author: rain
"""

import numpy as np
from numpy import linalg as LA
import copy
from sklearn.neighbors import NearestNeighbors

# parameters for voxel models
VoxelSize = 0.02 # meter
PatchSize = 16
BlockRealSize = 1.28  # meter, VoxelSize*64

VisibleLength = 100
VisibleWidth = 100
VisibleHeight = 15

Scales = int(3)
ScaleRatios = [1, 8, 32]

nLeastVoxelsInOneBlock = 5
BlockEdgeWidth = 1
nNeighborBlocks = 1


VoxelSizes = [VoxelSize, VoxelSize*ScaleRatios[1], VoxelSize*ScaleRatios[2]]
HalfVoxelSizes = [i/2 for i in VoxelSizes]

# parameters for voxel patch
PatchRadius = int(PatchSize/2)
nLeastPtsInOnePatch = 1


# parameters for block
BlockSize = int(BlockRealSize/VoxelSize)
CropBlocks = int(ScaleRatios[2]*PatchRadius/BlockSize)  # to ensure the crop safty of VoxelModel2 in sacle 2
nBlocksL = int(2*VisibleLength/BlockRealSize)
nBlocksW = int(2*VisibleWidth/BlockRealSize)
nBlocksH = int(2*VisibleHeight/BlockRealSize)
nBlocksL_half = int(nBlocksL/2)
NeighborOffset = nNeighborBlocks*BlockSize
BigBlockSize = BlockSize*(2*nNeighborBlocks+1)

# get the exactly visible region according to the blocksize and nBlocks
VisibleLength = nBlocksL/2 * BlockRealSize
VisibleWidth = nBlocksW/2 * BlockRealSize
VisibleHeight = nBlocksH/2 * BlockRealSize



# ----- to faster compute
ZeroBlock = np.zeros((BlockSize,BlockSize,BlockSize), dtype=np.int16) 
# BlockIndexList
BlockIndexList = []
for iBlockX in range(nBlocksL):
    for iBlockY in range(nBlocksW):
        for iBlockZ in range(nBlocksH):
            BlockIndexList.append([iBlockX,iBlockY,iBlockZ])
# null blocks
NullBlocks = []
for iBlockX in range(nBlocksL):
    NullBlocks.append([])
    for iBlockY in range(nBlocksW):
        NullBlocks[iBlockX].append([])
        for iBlockZ in range(nBlocksH):
            NullBlocks[iBlockX][iBlockY].append([])
            NullBlocks[iBlockX][iBlockY][iBlockZ].append(False)
# VoxelIndexList
VoxelIndexList = []
for iVoxelX in range(BlockSize):
    for iVoxelY in range(BlockSize):
        for iVoxelZ in range(BlockSize):
            VoxelIndexList.append([iVoxelX,iVoxelY,iVoxelZ])
# neighbor index list, for blocks
NeighborOffsetIdxList1 = []
for iX in range(-1,2,1):
    for iY in range(-1,2,1):
        for iZ in range(-1,2,1):
            voxelOffset = [iX*BlockSize,iY*BlockSize,iZ*BlockSize]
            voxelOffset = np.array(voxelOffset, dtype=np.int32).reshape([1,3])
            NeighborOffsetIdxList1.append([iX,iY,iZ,voxelOffset])


def FilterOutTooFarPts(PC):
    PC_abs = np.abs(PC)
    idx0 = PC_abs[:,0] > VisibleLength
    idx1 = PC_abs[:,1] > VisibleWidth
    idx2 = PC_abs[:,2] > VisibleHeight    
    idx = idx0 + idx1 + idx2
    if sum(idx) > 0:
        PC = PC[idx==False,:]
    return PC
    

def Voxelization(PC):    
    Blocks = copy.deepcopy(NullBlocks)
    avlBlocksList = []
    cntVoxelsLength = []
    cntVoxelsLength.append(0)  # reserve 1 for the starting
    AllVoxels = []
    VoxelModel1 = np.zeros((int(nBlocksL*BlockSize/ScaleRatios[1]),int(nBlocksW*BlockSize/ScaleRatios[1]),int(nBlocksH*BlockSize/ScaleRatios[1])), dtype=np.int8)
    VoxelModel2 = np.zeros((int(nBlocksL*BlockSize/ScaleRatios[2]),int(nBlocksW*BlockSize/ScaleRatios[2]),int(nBlocksH*BlockSize/ScaleRatios[2])), dtype=np.int8)
    assert max(VoxelModel1.shape[0],VoxelModel1.shape[1]) < 30000  # to make sure the size is smaller than int16 
    
    AllVoxels0 = []
    AllVoxels1 = []
    AllVoxels2 = []    
                
    # layers
    PC = FilterOutTooFarPts(PC)    
    for iPt in range(PC.shape[0]):
        pt=PC[iPt,:]
        x_ = pt[0]+VisibleLength
        y_ = pt[1]+VisibleWidth
        z_ = pt[2]+VisibleHeight
                
        iBlockX = int(x_/BlockRealSize)
        iBlockY = int(y_/BlockRealSize)
        iBlockZ = int(z_/BlockRealSize)
        
        if Blocks[iBlockX][iBlockY][iBlockZ][0] == False:
            avlBlocksList.append([iBlockX,iBlockY,iBlockZ])
            cntVoxelsLength.append(0)
            Blocks[iBlockX][iBlockY][iBlockZ][0] = True
            Blocks[iBlockX][iBlockY][iBlockZ].append(np.zeros((BlockSize,BlockSize,BlockSize), dtype=np.int8))
            Blocks[iBlockX][iBlockY][iBlockZ].append([])  # available voxel list
            Blocks[iBlockX][iBlockY][iBlockZ].append([])  # available voxel list with global index in scale 0 within this block
            
        
        # layer 0
        iVoxelX = np.int32((x_-iBlockX*BlockRealSize)/VoxelSize)
        iVoxelY = np.int32((y_-iBlockY*BlockRealSize)/VoxelSize)
        iVoxelZ = np.int32((z_-iBlockZ*BlockRealSize)/VoxelSize)
        if Blocks[iBlockX][iBlockY][iBlockZ][1][iVoxelX,iVoxelY,iVoxelZ] > 0:
            continue
        Blocks[iBlockX][iBlockY][iBlockZ][1][iVoxelX,iVoxelY,iVoxelZ] = 1
        Blocks[iBlockX][iBlockY][iBlockZ][2].append([iVoxelX,iVoxelY,iVoxelZ])
        Blocks[iBlockX][iBlockY][iBlockZ][3].append([iVoxelX+iBlockX*BlockSize, iVoxelY+iBlockY*BlockSize, iVoxelZ+iBlockZ*BlockSize])
        
        
        # layer0, layer1, and layer2
        iX1 = int(x_/VoxelSizes[1])
        iY1 = int(y_/VoxelSizes[1])
        iZ1 = int(z_/VoxelSizes[1])        
        iX2 = int(x_/VoxelSizes[2])
        iY2 = int(y_/VoxelSizes[2])
        iZ2 = int(z_/VoxelSizes[2])
        if VoxelModel1[iX1,iY1,iZ1] == 0:
            VoxelModel1[iX1,iY1,iZ1] = 1
            AllVoxels1.append([iX1,iY1,iZ1])
        if VoxelModel2[iX2,iY2,iZ2] == 0:
            VoxelModel2[iX2,iY2,iZ2] = 1
            AllVoxels2.append([iX2,iY2,iZ2])        
    
    # collect cntVoxelsLength and AllVoxels
    for iAvlBlock in range(len(avlBlocksList)):
        iBlockX, iBlockY, iBlockZ = avlBlocksList[iAvlBlock]
        cntVoxelsLength[iAvlBlock+1] = len(Blocks[iBlockX][iBlockY][iBlockZ][2]) + cntVoxelsLength[iAvlBlock]
        AllVoxels += Blocks[iBlockX][iBlockY][iBlockZ][2]
        AllVoxels0 += Blocks[iBlockX][iBlockY][iBlockZ][3]
        
    avlBlocksList = np.array(avlBlocksList, dtype=np.int16)
    cntVoxelsLength = np.array(cntVoxelsLength, dtype=np.int32)
    AllVoxels = np.array(AllVoxels, dtype=np.int16)
    AllVoxels0 = np.array(AllVoxels0, dtype=np.int16)
    AllVoxels1 = np.array(AllVoxels1, dtype=np.int16)
    AllVoxels2 = np.array(AllVoxels2, dtype=np.int16)
    return Blocks, VoxelModel1, VoxelModel2, avlBlocksList, cntVoxelsLength, AllVoxels, AllVoxels0, AllVoxels1, AllVoxels2



def GetPatchesList(Pts, AllVoxels0, AllVoxels1, AllVoxels2):    
    PatchesList = []
    for iScale in range(Scales):
        PatchesList.append(np.zeros((Pts.shape[0],PatchSize,PatchSize,PatchSize,1), dtype=np.float32))
    
    n_neighbors = 496  # 16*16 + 15*16
    radius = 14  # 8*8*8=192, 14^2=196
    
    Pts_ = Pts + [VisibleLength, VisibleWidth, VisibleHeight]
    
    AllVoxelsList = []
    AllVoxelsList.append(AllVoxels0)
    AllVoxelsList.append(AllVoxels1)
    AllVoxelsList.append(AllVoxels2)
    
    for iScale in range(Scales):
        KeyVoxels = np.array(Pts_/VoxelSizes[iScale], dtype=np.int32)
    
        nbrs = NearestNeighbors(n_neighbors=n_neighbors,radius=radius, algorithm='auto').fit(AllVoxelsList[iScale])
        indices = nbrs.kneighbors(KeyVoxels, return_distance=False)
    
        patchVoxels = AllVoxelsList[iScale][indices,:]
        
        KeyVoxels_ = np.expand_dims(KeyVoxels, axis=1)
        KeyVoxels_ = np.tile(KeyVoxels_, (1,n_neighbors,1))
        nbrVoxels = patchVoxels - KeyVoxels_
        
        idxX_ = (nbrVoxels[:,:,0] >= -PatchRadius)
        idxX = (nbrVoxels[:,:,0] < PatchRadius)
        idxY_ = (nbrVoxels[:,:,1] >= -PatchRadius)
        idxY = (nbrVoxels[:,:,1] < PatchRadius)
        idxZ_ = (nbrVoxels[:,:,2] >= -PatchRadius)
        idxZ = (nbrVoxels[:,:,2] < PatchRadius)
        avlIdx = idxX_*idxX*idxY_*idxY*idxZ_*idxZ
        
        for iVoxel in range(KeyVoxels.shape[0]):
            aVoxels_ = nbrVoxels[iVoxel,avlIdx[iVoxel,:],:]
            PatchesList[iScale][iVoxel,aVoxels_[:,0],aVoxels_[:,1],aVoxels_[:,2],0] = 1
   
    return Pts, PatchesList



def RebuildBlocksWithVoxelList(avlBlocksList, cntVoxelsLength, AllVoxels):
    Blocks = copy.deepcopy(NullBlocks)  # initiate blocks

    # rebuild blocks
    for iAvlBlock in range(avlBlocksList.shape[0]):
        iBlockX = avlBlocksList[iAvlBlock,0]
        iBlockY = avlBlocksList[iAvlBlock,1]
        iBlockZ = avlBlocksList[iAvlBlock,2]
        
        # 0: append the flag of isAvailable
        Blocks[iBlockX][iBlockY][iBlockZ][0] = True
        
        # 1: append current available block (voxel model)
        Blocks[iBlockX][iBlockY][iBlockZ].append(np.zeros((BlockSize,BlockSize,BlockSize), dtype=np.int16))
        # 1.1 and set the available voxels
        for iVoxel in range(cntVoxelsLength[iAvlBlock],cntVoxelsLength[iAvlBlock+1],1):
            Blocks[iBlockX][iBlockY][iBlockZ][1][AllVoxels[iVoxel,0],AllVoxels[iVoxel,1],AllVoxels[iVoxel,2]] = 1
                    
        # 2: append current available voxel index list
        Blocks[iBlockX][iBlockY][iBlockZ].append(list(AllVoxels[cntVoxelsLength[iAvlBlock]:cntVoxelsLength[iAvlBlock+1],:]))
                    
    return Blocks


def RebuildBlocksWithout3DVoxelArray(avlBlocksList, cntVoxelsLength, AllVoxels, AllVoxels1, AllVoxels2):
    # initiate blocks
    BlockMask = np.zeros((nBlocksL, nBlocksW, nBlocksH), dtype=np.int32)
    Blocks = copy.deepcopy(NullBlocks)
    VoxelModel1 = np.zeros((int(nBlocksL*BlockSize/ScaleRatios[1]),int(nBlocksW*BlockSize/ScaleRatios[1]),int(nBlocksH*BlockSize/ScaleRatios[1])), dtype=np.int16)
    VoxelModel2 = np.zeros((int(nBlocksL*BlockSize/ScaleRatios[2]),int(nBlocksW*BlockSize/ScaleRatios[2]),int(nBlocksH*BlockSize/ScaleRatios[2])), dtype=np.int16)
    
    # set block mask
    for iAvlBlock in range(avlBlocksList.shape[0]):
        iBlockX = avlBlocksList[iAvlBlock,0]
        iBlockY = avlBlocksList[iAvlBlock,1]
        iBlockZ = avlBlocksList[iAvlBlock,2]
        
        # set mask as 1
        BlockMask[iBlockX, iBlockY, iBlockZ] = 1

        # 0: append the flag of isAvailable
        Blocks[iBlockX][iBlockY][iBlockZ][0] = True
        
        # 1: append current available block (voxel model)
        Blocks[iBlockX][iBlockY][iBlockZ].append([])  # without 3D voxel array
        
        # 2: and rebuild avlVoxelList
        aAvlVoxelList = AllVoxels[cntVoxelsLength[iAvlBlock]:cntVoxelsLength[iAvlBlock+1],:]
        Blocks[iBlockX][iBlockY][iBlockZ].append(aAvlVoxelList)    

    # rebuild voxelmodel1 and voxelmodel2
    for iVoxel in range(AllVoxels1.shape[0]):
        VoxelModel1[AllVoxels1[iVoxel,0],AllVoxels1[iVoxel,1],AllVoxels1[iVoxel,2]] = 1
    for iVoxel in range(AllVoxels2.shape[0]):
        VoxelModel2[AllVoxels2[iVoxel,0],AllVoxels2[iVoxel,1],AllVoxels2[iVoxel,2]] = 1
        
    return BlockMask, Blocks, VoxelModel1, VoxelModel2




def RebuildVoxelModel(avlBlocksList, cntVoxelsLength, AllVoxels, AllVoxels1, AllVoxels2):
    Blocks = copy.deepcopy(NullBlocks)  # initiate blocks
    VoxelModel1 = np.zeros((int(nBlocksL*BlockSize/ScaleRatios[1]),int(nBlocksW*BlockSize/ScaleRatios[1]),int(nBlocksH*BlockSize/ScaleRatios[1])), dtype=np.int16)
    VoxelModel2 = np.zeros((int(nBlocksL*BlockSize/ScaleRatios[2]),int(nBlocksW*BlockSize/ScaleRatios[2]),int(nBlocksH*BlockSize/ScaleRatios[2])), dtype=np.int16)
    
    # rebuild blocks
    for iAvlBlock in range(avlBlocksList.shape[0]):
        iBlockX = avlBlocksList[iAvlBlock,0]
        iBlockY = avlBlocksList[iAvlBlock,1]
        iBlockZ = avlBlocksList[iAvlBlock,2]
        
        # 0: append the flag of isAvailable
        Blocks[iBlockX][iBlockY][iBlockZ][0] = True
        
        # 1: append current available block (voxel model)
        Blocks[iBlockX][iBlockY][iBlockZ].append(np.zeros((BlockSize,BlockSize,BlockSize), dtype=np.int16))
        # 1.1 and set the available voxels
        for iVoxel in range(cntVoxelsLength[iAvlBlock],cntVoxelsLength[iAvlBlock+1],1):
            Blocks[iBlockX][iBlockY][iBlockZ][1][AllVoxels[iVoxel,0],AllVoxels[iVoxel,1],AllVoxels[iVoxel,2]] = 1
                    
    # rebuild voxelmodel1 and voxelmodel2
    for iVoxel in range(AllVoxels1.shape[0]):
        VoxelModel1[AllVoxels1[iVoxel,0],AllVoxels1[iVoxel,1],AllVoxels1[iVoxel,2]] = 1
    for iVoxel in range(AllVoxels2.shape[0]):
        VoxelModel2[AllVoxels2[iVoxel,0],AllVoxels2[iVoxel,1],AllVoxels2[iVoxel,2]] = 1
        
    return Blocks, VoxelModel1, VoxelModel2


def RebuildVoxelModelWithVoxelList(avlBlocksList, cntVoxelsLength, AllVoxels, AllVoxels1, AllVoxels2):
    Blocks = copy.deepcopy(NullBlocks)  # initiate blocks
    VoxelModel1 = np.zeros((int(nBlocksL*BlockSize/ScaleRatios[1]),int(nBlocksW*BlockSize/ScaleRatios[1]),int(nBlocksH*BlockSize/ScaleRatios[1])), dtype=np.int16)
    VoxelModel2 = np.zeros((int(nBlocksL*BlockSize/ScaleRatios[2]),int(nBlocksW*BlockSize/ScaleRatios[2]),int(nBlocksH*BlockSize/ScaleRatios[2])), dtype=np.int16)
    
    # rebuild blocks
    for iAvlBlock in range(avlBlocksList.shape[0]):
        iBlockX = avlBlocksList[iAvlBlock,0]
        iBlockY = avlBlocksList[iAvlBlock,1]
        iBlockZ = avlBlocksList[iAvlBlock,2]
        
        # 0: append the flag of isAvailable
        Blocks[iBlockX][iBlockY][iBlockZ][0] = True
        
        # 1: append current available block (voxel model)
        Blocks[iBlockX][iBlockY][iBlockZ].append(np.zeros((BlockSize,BlockSize,BlockSize), dtype=np.int16))
        # 1.1 and set the available voxels
        for iVoxel in range(cntVoxelsLength[iAvlBlock],cntVoxelsLength[iAvlBlock+1],1):
            Blocks[iBlockX][iBlockY][iBlockZ][1][AllVoxels[iVoxel,0],AllVoxels[iVoxel,1],AllVoxels[iVoxel,2]] = 1
            
        # 2: append current available voxel index list
        Blocks[iBlockX][iBlockY][iBlockZ].append(list(AllVoxels[cntVoxelsLength[iAvlBlock]:cntVoxelsLength[iAvlBlock+1],:]))
                    
    # rebuild voxelmodel1 and voxelmodel2 
    for iVoxel in range(AllVoxels1.shape[0]):
        VoxelModel1[AllVoxels1[iVoxel,0],AllVoxels1[iVoxel,1],AllVoxels1[iVoxel,2]] = 1
    for iVoxel in range(AllVoxels2.shape[0]):
        VoxelModel2[AllVoxels2[iVoxel,0],AllVoxels2[iVoxel,1],AllVoxels2[iVoxel,2]] = 1
        
    return Blocks, VoxelModel1, VoxelModel2


def GetIndexByAllVoxelListIndex(avlBlocksList, cntVoxelsLength, AllVoxels, iListIndex):
    iBlock = np.where(cntVoxelsLength >= iListIndex)[0][0]
    iBlock -= 1
    
    iBlockIndex =  avlBlocksList[iBlock,:].flatten()
    iVoxel = AllVoxels[iListIndex,:].flatten()
    
    return iBlockIndex[0], iBlockIndex[1], iBlockIndex[2], iVoxel[0], iVoxel[1], iVoxel[2]
    
    



##--------------------- Voxel 2 PC --------------------------------------------------------------

def VoxelModel2PCUsingBlocks(Blocks, avlBlocksList):
    PC = []
    for iAvlBlock in range(avlBlocksList.shape[0]):
        iBlockX = avlBlocksList[iAvlBlock,0]
        iBlockY = avlBlocksList[iAvlBlock,1]
        iBlockZ = avlBlocksList[iAvlBlock,2]
        offsetX = iBlockX*BlockRealSize-VisibleLength
        offsetY = iBlockY*BlockRealSize-VisibleWidth
        offsetZ = iBlockZ*BlockRealSize-VisibleHeight
        for iVoxelX, iVoxelY, iVoxelZ in Blocks[iBlockX][iBlockY][iBlockZ][2]:
            x = iVoxelX*VoxelSize + offsetX
            y = iVoxelY*VoxelSize + offsetY
            z = iVoxelZ*VoxelSize + offsetZ
            PC.append([x,y,z])
    PC = np.array(PC, dtype=np.float32)
    return PC        
    

def VoxelModel2PC_3Scales(avlBlocksList, cntVoxelsLength, AllVoxels, AllVoxels1, AllVoxels2):
    # for PC from blocks
    PC = []
    VisibleLength_ = VisibleLength - HalfVoxelSizes[0]  # for the correction of the shift by voxelization
    VisibleWidth_ = VisibleWidth - HalfVoxelSizes[0]
    VisibleHeight_ = VisibleHeight - HalfVoxelSizes[0]
    for iAvlBlock in range(avlBlocksList.shape[0]):
        iBlockX = avlBlocksList[iAvlBlock,0]
        iBlockY = avlBlocksList[iAvlBlock,1]
        iBlockZ = avlBlocksList[iAvlBlock,2]
        offsetX = iBlockX*BlockRealSize-VisibleLength_
        offsetY = iBlockY*BlockRealSize-VisibleWidth_
        offsetZ = iBlockZ*BlockRealSize-VisibleHeight_
        if cntVoxelsLength[iAvlBlock+1] - cntVoxelsLength[iAvlBlock] < nLeastVoxelsInOneBlock:
            continue
        for iVoxel in range(cntVoxelsLength[iAvlBlock],cntVoxelsLength[iAvlBlock+1],1):
            x = AllVoxels[iVoxel,0]*VoxelSize + offsetX
            y = AllVoxels[iVoxel,1]*VoxelSize + offsetY
            z = AllVoxels[iVoxel,2]*VoxelSize + offsetZ
            PC.append([x,y,z])
    PC = np.array(PC, dtype=np.float32)
    
    # for PC1 and PC2 from voxelmodel1 and voxelmodel2
    PC1 = np.zeros((AllVoxels1.shape[0],3), dtype=np.float32)
    VisibleLength_ = VisibleLength - HalfVoxelSizes[1]  # for the correction of the shift by voxelization
    VisibleWidth_ = VisibleWidth - HalfVoxelSizes[1]
    VisibleHeight_ = VisibleHeight - HalfVoxelSizes[1]
    for iPt in range(AllVoxels1.shape[0]):
        PC1[iPt,0] = AllVoxels1[iPt,0]*VoxelSizes[1] - VisibleLength_
        PC1[iPt,1] = AllVoxels1[iPt,1]*VoxelSizes[1] - VisibleWidth_
        PC1[iPt,2] = AllVoxels1[iPt,2]*VoxelSizes[1] - VisibleHeight_
        
    PC2 = np.zeros((AllVoxels2.shape[0],3), dtype=np.float32)
    VisibleLength_ = VisibleLength - HalfVoxelSizes[2]  # for the correction of the shift by voxelization
    VisibleWidth_ = VisibleWidth - HalfVoxelSizes[2]
    VisibleHeight_ = VisibleHeight - HalfVoxelSizes[2]
    for iPt in range(AllVoxels2.shape[0]):
        PC2[iPt,0] = AllVoxels2[iPt,0]*VoxelSizes[2] - VisibleLength_
        PC2[iPt,1] = AllVoxels2[iPt,1]*VoxelSizes[2] - VisibleWidth_
        PC2[iPt,2] = AllVoxels2[iPt,2]*VoxelSizes[2] - VisibleHeight_
        
    return PC, PC1, PC2

def VoxelModel2PCUsingVoxelList(KeyVoxelList):
    PC = []
    for iBlockX, iBlockY, iBlockZ, iVoxelX, iVoxelY, iVoxelZ in KeyVoxelList:
        offsetX = iBlockX*BlockRealSize-VisibleLength
        offsetY = iBlockY*BlockRealSize-VisibleWidth
        offsetZ = iBlockZ*BlockRealSize-VisibleHeight
        x = iVoxelX*VoxelSize + offsetX
        y = iVoxelY*VoxelSize + offsetY
        z = iVoxelZ*VoxelSize + offsetZ
        PC.append([x,y,z])
    PC = np.array(PC, dtype=np.float32)
    return PC
        


def VoxelModel2PC(voxelModel):
    nLeastPtInVoxel=0.1
#    nLeastPtInVoxel=0.001
    voxelPC=[]
    for iX in range(voxelModel.shape[0]):
        for iY in range(voxelModel.shape[1]):
            for iZ in range(voxelModel.shape[2]):
                if voxelModel[iX,iY,iZ]>nLeastPtInVoxel:
                    voxelPC.append([iX*VoxelSize-VisibleLength,iY*VoxelSize-VisibleWidth,iZ*VoxelSize-VisibleHeight])
    voxelPC=np.array(voxelPC, dtype=np.float32)
    return voxelPC


def VoxelModel2ColofulPC(voxelModel_4D):
    voxelPC=np.zeros((voxelModel_4D.shape[0]*voxelModel_4D.shape[1]*voxelModel_4D.shape[2],3), dtype=np.float32)
    colors=np.zeros((voxelModel_4D.shape[0]*voxelModel_4D.shape[1]*voxelModel_4D.shape[2],1), dtype=np.float32)
    cntVoxelsInPC=0
    for iX in range(voxelModel_4D.shape[0]):
        for iY in range(voxelModel_4D.shape[1]):
            for iZ in range(voxelModel_4D.shape[2]):
                voxelPC[cntVoxelsInPC,:]=[iX*VoxelSize-VisibleLength,iY*VoxelSize-VisibleWidth,iZ*VoxelSize-VisibleHeight]
#                color=0
#                for iValue in range(voxelModel_4D.shape[3]):
#                    color=color+voxelModel_4D[iX,iY,iZ,iValue]*math.pow(2,iValue)
#                colors[cntVoxelsInPC]=color/(math.pow(2,voxelModel_4D.shape[3])-1)  #scale to [0,1]
                colors[cntVoxelsInPC] = LA.norm(voxelModel_4D[iX,iY,iZ,:])
                cntVoxelsInPC=cntVoxelsInPC+1
    colors=colors/np.max(colors)
    deleteList=[]
    for i in range(len(colors)):
        if colors[i]<0.95:
            deleteList.append(i)    
    voxelPC=np.delete(voxelPC, deleteList, 0)
    colors=np.delete(colors, deleteList, 0)
    colors = colors - min(colors)
    colors = colors / max(colors)
    return voxelPC, colors





