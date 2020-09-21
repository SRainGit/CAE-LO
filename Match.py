#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 19:30:26 2019

@author: rain
"""

import os
from scipy import io
from scipy.spatial.distance import cdist
import numpy as np
import mayavi.mlab as mlab
from numpy import linalg as LA
from numpy.linalg import det
from numpy import dot as dot
import matplotlib.pyplot as plt
from time import time, sleep

from Dirs import *
from Voxel import *
from SphericalRing import *
from Transformations import *
from sklearn.preprocessing import normalize

            

def LoadVoxelModel(RawFileName):
    baseDir = os.path.dirname(os.path.dirname(RawFileName))
    voxelFile = os.path.join(baseDir,'VoxelModel',RawFileName.split("/")[-1]+'.mat')
    
    # load voxel data
    mat = io.loadmat(voxelFile)
    avlBlocksList = mat['avlBlocksList']
    cntVoxelsLength = mat['cntVoxelsLength'].flatten()
    AllVoxels = mat['AllVoxels']
    AllVoxels1 = mat['AllVoxels1']
    AllVoxels2 = mat['AllVoxels2']    
    
    # rebuild the voxel models
    Blocks, VoxelModel1, VoxelModel2 = RebuildVoxelModel(avlBlocksList, cntVoxelsLength, AllVoxels, AllVoxels1, AllVoxels2)    
    
    return Blocks, VoxelModel1, VoxelModel2


def LoadVoxelModelAndKeyPts(RawFileName):
    baseDir = os.path.dirname(os.path.dirname(RawFileName))
    voxelFile = os.path.join(baseDir,'VoxelModel',RawFileName.split("/")[-1]+'.mat')
    KeyPtsFile = os.path.join(baseDir,'KeyPts',RawFileName.split("/")[-1]+'.mat')
    
    # load voxel data
    mat = io.loadmat(voxelFile)
    AllVoxels0 = mat['AllVoxels0']
    AllVoxels1 = mat['AllVoxels1']
    AllVoxels2 = mat['AllVoxels2']
        
    # load key pts data
    mat = io.loadmat(KeyPtsFile)
    KeyPts = mat['KeyPts']
    
    return KeyPts, AllVoxels0, AllVoxels1, AllVoxels2



def LoadKeyPtsAndFeatures(RawFileName):
    baseDir = os.path.dirname(os.path.dirname(RawFileName))
    FeaturesFile = os.path.join(baseDir,'Features',RawFileName.split("/")[-1]+'.mat')    
    mat = io.loadmat(FeaturesFile)
    KeyPts = mat['KeyPts']
    Features = mat['Features']
    Weights = mat['Weights']    
    return KeyPts, Features, Weights

    
def GetKeyVoxelsAroundKeyPts(Blocks, KeyPts):
    nBlocksL_ = nBlocksL - CropBlocks
    nBlocksW_ = nBlocksW - CropBlocks
    nBlocksH_ = nBlocksH - CropBlocks    
    
    arrAllExtendedVoxels = np.zeros((1,6), dtype=np.int32)
    # convert KeyPts into KeyVoxels
    for iPt in range(KeyPts.shape[0]):
        pt = KeyPts[iPt,:]
        
        x_ = pt[0]+VisibleLength
        y_ = pt[1]+VisibleWidth
        z_ = pt[2]+VisibleHeight
        
        iBlockX = int(x_/BlockRealSize)
        iBlockY = int(y_/BlockRealSize)
        iBlockZ = int(z_/BlockRealSize)        
        
        # not using the boundary blocks for simplify
        if iBlockX < CropBlocks or iBlockX >= nBlocksL_ or\
           iBlockY < CropBlocks or iBlockY >= nBlocksW_ or\
            iBlockZ < CropBlocks or iBlockZ >= nBlocksH_:
            continue
        if Blocks[iBlockX][iBlockY][iBlockZ][0] == False:
            continue
        
        arrCurBlockIdx = np.array([iBlockX,iBlockY,iBlockZ], dtype=np.int32).reshape(1,3)
        arrCurBlockIdx = np.tile(arrCurBlockIdx, (len(Blocks[iBlockX][iBlockY][iBlockZ][2]),1))
        arrVoxelList = np.array(Blocks[iBlockX][iBlockY][iBlockZ][2], dtype=np.int32)
        arrVoxelList = np.c_[arrCurBlockIdx, arrVoxelList]        
        arrAllExtendedVoxels = np.r_[arrAllExtendedVoxels, arrVoxelList]
        
    
    arrAllExtendedVoxels = np.delete(arrAllExtendedVoxels, [0], axis=0)
    return arrAllExtendedVoxels

    
def GetKeyPtsFromKeyVoxels(KeyVoxels):
    VisibleLength_ = VisibleLength - HalfVoxelSizes[0]  # for the correction of the shift by voxelization
    VisibleWidth_ = VisibleWidth - HalfVoxelSizes[0]
    VisibleHeight_ = VisibleHeight - HalfVoxelSizes[0]    
    KeyPts = np.zeros((KeyVoxels.shape[0],3), dtype=np.float32)
    iPt = 0
    for iVoxel in range(KeyVoxels.shape[0]):
        # compute the corresponding real 3d point
        offsetX = KeyVoxels[iVoxel,0]*BlockRealSize-VisibleLength_
        offsetY = KeyVoxels[iVoxel,1]*BlockRealSize-VisibleWidth_
        offsetZ = KeyVoxels[iVoxel,2]*BlockRealSize-VisibleHeight_
        KeyPts[iPt,0] = KeyVoxels[iVoxel,3]*VoxelSize + offsetX
        KeyPts[iPt,1] = KeyVoxels[iVoxel,4]*VoxelSize + offsetY
        KeyPts[iPt,2] = KeyVoxels[iVoxel,5]*VoxelSize + offsetZ             
        iPt += 1         
    return KeyPts


def GetFeaturesFromPatches(PatchEncoder, PatchesList):
    Features0 = PatchEncoder.predict(PatchesList[0])
    Features1 = PatchEncoder.predict(PatchesList[1])
    Features2 = PatchEncoder.predict(PatchesList[2])
    Features = np.c_[Features0,Features1,Features2]
    return Features
 

def SolveRT(Pairs0, Pairs1):
    isCredible=1
    
    mean0=np.mean(Pairs0, axis=0).reshape(1,3)
    mean1=np.mean(Pairs1, axis=0).reshape(1,3)
    Pairs0_=Pairs0-mean0
    Pairs1_=Pairs1-mean1
    
    H = np.dot(Pairs1_.T, Pairs0_)
        
    U, Sigma, V = LA.svd(H)
    R = np.dot(V.T, U.T)
    
    if det(R) < 0:
        isCredible = -1
#        print('Reflection detected')
        V[:,2] = V[:,2]*(-1)
        R = np.dot(V.T, U.T)
    
    T = mean0.T - np.dot(R, mean1.T)
    return R, T, isCredible
        


def RANSAC4RT(Pairs0, Pairs1, Weights0, Weights1):    
#    nRandSamples = 3
    nRandSamples = 4
    
    leastInliers = min(100, int(0.2*Pairs0.shape[0]))
    minSuccessInliers = 0.25*Pairs0.shape[0]
    minTrails = 100
    maxTrails = 500
#    maxTrails = 1000
    residualThreshold = 0.4
#    residualThreshold = 1.0
    
    isSuccess = False
    cntIters = 0
    curNumInliers = 0
    R_star = np.eye(3, dtype=np.float64)
    T_star = np.zeros((3,1), dtype=np.float64)
    inlierIdx_star = np.zeros((Pairs0.shape[0],), dtype=np.bool)
    while True:
        while (cntIters < minTrails) or (cntIters >= minTrails and cntIters < maxTrails and  curNumInliers < minSuccessInliers):        
            RandIdxes = np.random.random((nRandSamples,))
            RandIdxes = RandIdxes*(Pairs0.shape[0])
            RandIdxes = np.array(RandIdxes, dtype=np.int32)
            
            samples0 = Pairs0[RandIdxes,:]
            samples1 = Pairs1[RandIdxes,:]
            
            R, T, isCredible = SolveRT(samples0, samples1)
            
            Pairs1_ = (dot(R, Pairs1.T) + T).T
            dists = LA.norm(Pairs0 - Pairs1_, axis=1)
            inlierIdx = np.array(dists<residualThreshold, dtype=np.bool)
            nInliers = sum(inlierIdx)
            if nInliers < leastInliers:
                cntIters += 1
                continue
            inlierIdx = dists < residualThreshold
            if  nInliers > curNumInliers:
                curNumInliers = nInliers
                inlierIdx_star = inlierIdx
                R_star = R
                T_star = T
                
            cntIters += 1
            isSuccess = True
        if isSuccess == True:
            break
        cntIters = 0
        residualThreshold = 2*residualThreshold
        if residualThreshold > 2.0:
            print('failed when residual =', residualThreshold)
            residualThreshold = residualThreshold/2
            break
    print('cntItersRANSAC =', cntIters)
    print('residualThreshold =', residualThreshold)
    print('nInliers/nFilteredKeyPts1 =', curNumInliers, '/', Pairs0.shape[0], '=', round(curNumInliers/Pairs0.shape[0], 3))
    return  R_star, T_star, isSuccess, inlierIdx_star, residualThreshold


def FilterOutBadKeyPts(PC, Codes, nMaximumPts):
    nMinimumPts = 200
    SelectionRatio = 0.9
    
    distMatrix_PC = cdist(PC, PC, metric='euclidean')
    distMatrix_Code = cdist(Codes, Codes, metric='euclidean')
    scoreMatix = distMatrix_PC*distMatrix_Code
    scores = np.sum(scoreMatix, axis=1)
    sortIdx = scores.argsort()
    
    if nMaximumPts < 0 or PC.shape[0] < nMinimumPts:
        NumOfSelected = int(PC.shape[0]*SelectionRatio)
    else:
        NumOfSelected = int(min(PC.shape[0]*SelectionRatio, nMaximumPts))
        
    idx = sortIdx[PC.shape[0]-NumOfSelected : PC.shape[0]]
    return idx



def SolveRelativePose(OriPC0, OriCodes0, Weights0, OriPC1, OriCodes1, Weights1):
    # filter out the keyPts with low distinctiveness
#    idx0 = FilterOutBadKeyPts(OriPC0, OriCodes0, -1)
#    idx1 = FilterOutBadKeyPts(OriPC1, OriCodes1, -1)
#    PC0 = OriPC0[idx0,:]
#    Codes0 = OriCodes0[idx0,:]
#    PC1 = OriPC1[idx1,:]
#    Codes1 = OriCodes1[idx1,:]
    idx0 = np.arange(OriPC0.shape[0])
    idx1 = np.arange(OriPC1.shape[0])
    PC0 = OriPC0
    Codes0 = OriCodes0
    PC1 = OriPC1
    Codes1 = OriCodes1
    
    # the original pairs
    distMatrix = cdist(Codes0, Codes1, metric='euclidean')
    pairIdx = np.argmin(distMatrix, axis=0)
    
    Pairs0 = PC0[pairIdx,:]
    Pairs1 = PC1
    idxPairs0 = idx0[pairIdx]
    idxPairs1 = idx1
    
    Weights0 = np.ones(Pairs0.shape[0],dtype=np.float32)
    Weights1 = np.ones(Pairs1.shape[0],dtype=np.float32)
    
    # solve RT using RANSAC
    R, T, isSuccess, inlierIdx, residualThreshold = RANSAC4RT(Pairs0, Pairs1, Weights0, Weights1)
    
#    inlierIdx = range(0,Pairs0.shape[0])
    
    inliersIdx0 =  idxPairs0[inlierIdx]
    inliersIdx1 =  idxPairs1[inlierIdx]

    
    if inliersIdx0.shape[0] == 0:
        return R, T, isSuccess, inliersIdx0, inliersIdx1, residualThreshold
    
    inliers0 = OriPC0[inliersIdx0,:]
    inliers1 = OriPC1[inliersIdx1,:]
    R, T, isCredible = SolveRT(inliers0, inliers1)    
    return R, T, isSuccess, inliersIdx0, inliersIdx1, residualThreshold
   
   
if __name__ == "__main__": 
    
    bLoadKeyPtsFromFile = False
    # bLoadKeyPtsFromFile = True
    
    bLoadFeaturesFromFile = False
    # bLoadFeaturesFromFile = True
    

    strSequence = '01'
    iFrame0 = 498
    iFrameStep = 1
    iFrame1 = iFrame0 + iFrameStep
    DataDir = strDataBaseDir + strSequence + '/velodyne/'
    FileName0 = DataDir + str(iFrame0).zfill(6)+'.bin'
    FileName1 = DataDir + str(iFrame1).zfill(6)+'.bin'
    
    PC0 = np.fromfile(FileName0, dtype=np.float32, count=-1).reshape([-1,4])[:,0:3]
    PC1 = np.fromfile(FileName1, dtype=np.float32, count=-1).reshape([-1,4])[:,0:3]
    
    
    import os
    import tensorflow as tf
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    gpus = tf.config.experimental.list_physical_devices('GPU')    
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
    import keras
    from keras.models import load_model
    PatchEncoder = load_model('EncoderModel4VoxelPatch.h5')
    
    t0=time()
    
    if bLoadFeaturesFromFile == False:
        KeyPts0, AllVoxels00, AllVoxels01, AllVoxels02 = LoadVoxelModelAndKeyPts(FileName0)
        KeyPts1, AllVoxels10, AllVoxels11, AllVoxels12 = LoadVoxelModelAndKeyPts(FileName1)
        t1=time()
        print(round(t1-t0, 2), 's, Loading Data')
        
        if bLoadKeyPtsFromFile == False:        
            RespondLayer = load_model('SphericalRingPCRespondLayer.h5')
            KeyPts0, KeyPixels0, PlanarPts0 = GetKeyPtsFromRawFileName(FileName0, RespondLayer)
            KeyPts1, KeyPixels1, PlanarPts1 = GetKeyPtsFromRawFileName(FileName1, RespondLayer)
            
        KeyPts0, PatchesList0 = GetPatchesList(KeyPts0, AllVoxels00, AllVoxels01, AllVoxels02)
        KeyPts1, PatchesList1 = GetPatchesList(KeyPts1, AllVoxels10, AllVoxels11, AllVoxels12)            
        print('nKeyPts0 =', KeyPts0.shape[0])    
        print('nKeyPts1 =', KeyPts1.shape[0])    
        Features0 = GetFeaturesFromPatches(PatchEncoder, PatchesList0)
        Features1 = GetFeaturesFromPatches(PatchEncoder, PatchesList1)
        Weights0 = np.ones((KeyPts0.shape[0],1),dtype=np.float32)
        Weights1 = np.ones((KeyPts1.shape[0],1),dtype=np.float32)    
        
    else:        
        KeyPts0, Features0, Weights0 = LoadKeyPtsAndFeatures(FileName0)
        KeyPts1, Features1, Weights1 = LoadKeyPtsAndFeatures(FileName1)
        
    t2=time()
    print(round(t2-t0, 2), 's, Geting KeyPts')
    
    R, T, score, inliersIdx0, inliersIdx1, residualThreshold = SolveRelativePose(KeyPts0, Features0, Weights0, KeyPts1, Features1, Weights1)
    pairs0 = KeyPts0[inliersIdx0,:]
    pairs1 = KeyPts1[inliersIdx1,:]
    
    t3=time()
#    print(R, '\n', T)
    print(round(t3-t2, 2), 's, Solving Pose')
    print('total time =', round(t3-t0, 2))
    
    iSequence = int(strSequence)
    if iSequence < 12:
        # get errors    
        poses = np.loadtxt(strGroundTruthPosesDir+strSequence+'.txt')
        calibFileFullPath = str(strCalibDataDir + strSequence + '/calib_.txt')
        calib=np.loadtxt(calibFileFullPath)
        Tr=np.array(calib[4,:].reshape(3,4),dtype=np.float32)
        R_Tr=Tr[:,0:3]
        R_Tr_inv=np.linalg.inv(R_Tr)
        T_Tr=Tr[:,3].reshape(3,1)
        T_Tr_inv = -np.dot(R_Tr_inv, T_Tr)
        R_GT, T_GT =  GetLidarRelRtBetween2Poses(poses[iFrame0,:], poses[iFrame1,:], R_Tr, T_Tr, R_Tr_inv, T_Tr_inv)
        errorR = dot(np.linalg.inv(R), R_GT)
        errorEulers = RotateMat2EulerAngle_XYZ(errorR)
        errorT = T - T_GT
        print(errorEulers, errorT.T)
    
        
    PC1_ = (np.dot(R, PC1.T) + T.reshape(3,1)).T
    FusedPC = np.r_[PC0, PC1_]
    KeyPts1_ = (np.dot(R, KeyPts1.T) + T.reshape(3,1)).T
    pairs1_ = (np.dot(R, pairs1.T) + T.reshape(3,1)).T
    
    
    Colors0=np.ones((PC0.shape[0],1), dtype=np.float32)*1.0
    KeyColors0=np.ones((KeyPts0.shape[0],1), dtype=np.float32)*0.5
    KeyColors0_=np.ones((KeyPts0.shape[0],1), dtype=np.float32)*0.8
    
    Colors1=np.ones((PC1.shape[0],1), dtype=np.float32)*0.0
    KeyColors1=np.ones((KeyPts1.shape[0],1), dtype=np.float32)*0.5
    KeyColors1_=np.ones((KeyPts1.shape[0],1), dtype=np.float32)*0.2
    
    Colors4FusedPC=np.r_[Colors0, Colors1]
    #    Colors4FusedPC=np.r_[SingleColor0, SingleColor1]
    Colors4FusedPC=Colors4FusedPC.reshape(Colors4FusedPC.shape[0],)
    
    
    shift4Show = 12
    fig = mlab.figure(bgcolor=(0, 0, 0), size=(1500, 900))
    PtSize  = 0.4
    
    
    node = mlab.points3d(PC0[:,0], PC0[:,1], PC0[:,2], mode="point", figure=fig)
    node.glyph.scale_mode = 'scale_by_vector'
    node.mlab_source.dataset.point_data.scalars = Colors0
    #node.mlab_source.dataset.point_data.scalars = SingleColor0
    
    node = mlab.points3d(PC1[:,0], PC1[:,1], PC1[:,2]+shift4Show, mode="point", figure=fig)
    node.glyph.scale_mode = 'scale_by_vector'
    node.mlab_source.dataset.point_data.scalars = Colors1
    #node.mlab_source.dataset.point_data.scalars = SingleColor1
    
    
    node = mlab.points3d(KeyPts0[:,0], KeyPts0[:,1], KeyPts0[:,2], scale_factor=PtSize, figure=fig)
    node.glyph.scale_mode = 'scale_by_vector'
    node.mlab_source.dataset.point_data.scalars = KeyColors0
    
    
    node = mlab.points3d(KeyPts1[:,0], KeyPts1[:,1], KeyPts1[:,2]+shift4Show, scale_factor=PtSize, figure=fig)
    node.glyph.scale_mode = 'scale_by_vector'
    node.mlab_source.dataset.point_data.scalars = KeyColors1
    
        
    mlab.quiver3d(pairs1[:,0], pairs1[:,1], pairs1[:,2]+shift4Show, \
                         pairs0[:,0]-pairs1[:,0], pairs0[:,1]-pairs1[:,1], pairs0[:,2]-pairs1[:,2]-shift4Show, \
                         figure=fig, line_width=0.5, scale_factor=1)
    
    mlab.title('Feature Matching')
#    mlab.axes(xlabel='X', ylabel='Y', zlabel='Z')
    
    
    
    PtSize = 0.05
    fig = mlab.figure(bgcolor=(0, 0, 0), size=(1200, 800))
    mlab.points3d(FusedPC[:,0], FusedPC[:,1], FusedPC[:,2],
                         Colors4FusedPC, mode="point", figure=fig)
    
    node = mlab.points3d(KeyPts0[:,0], KeyPts0[:,1], KeyPts0[:,2], scale_factor=PtSize, figure=fig)
    node.glyph.scale_mode = 'scale_by_vector'
    node.mlab_source.dataset.point_data.scalars = KeyColors0_
    
    
    node = mlab.points3d(KeyPts1_[:,0], KeyPts1_[:,1], KeyPts1_[:,2], scale_factor=PtSize, figure=fig)
    node.glyph.scale_mode = 'scale_by_vector'
    node.mlab_source.dataset.point_data.scalars = KeyColors1_    
    
    mlab.quiver3d(pairs1_[:,0], pairs1_[:,1], pairs1_[:,2], \
                         pairs0[:,0]-pairs1_[:,0], pairs0[:,1]-pairs1_[:,1], pairs0[:,2]-pairs1_[:,2], \
                         figure=fig, line_width=0.5, scale_factor=1)
    
    mlab.title('Fused PC')
    mlab.axes(xlabel='X', ylabel='Y', zlabel='Z')   
    
    
    
    mlab.show()
















    
    
    