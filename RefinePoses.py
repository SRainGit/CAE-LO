#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 10:07:40 2019

@author: rain
"""

import os
import sys
import numpy as np
from numpy import dot
from numpy import linalg as LA
import mayavi.mlab as mlab
from scipy import io
from threading import Thread
from time import time, sleep
from scipy.spatial.distance import cdist
import copy
from MyICP import *
import math

from Dirs import *
from Visualization import *
from Voxel import *
from Match import *
from Transformations import *
     

def LoadPC(strSequence, iFrame):
    RawDataDir = str(strDataBaseDir+strSequence+'/velodyne/')
    PC = np.fromfile(str(RawDataDir+str(iFrame).zfill(6)+'.bin'), dtype=np.float32, count=-1).reshape([-1,4])[:,0:3]
    return PC

def LoadKeyPts(strSequence, iFrame):
    KeyPtsDir = str(strDataBaseDir+strSequence+'/KeyPts/')
    keyPtsData = io.loadmat(str(KeyPtsDir+str(iFrame).zfill(6)+'.bin.mat'))
    KeyPts = keyPtsData['KeyPts']
    return KeyPts

def LoadFeaturesData(strSequence, iFrame):
    FeaturesDir = str(strDataBaseDir+strSequence+'/Features/')
    FeaturesData = io.loadmat(str(FeaturesDir+str(iFrame).zfill(6)+'.bin.mat'))
    KeyPts = FeaturesData['KeyPts']
    Features = FeaturesData['Features']
    Weights = FeaturesData['Weights']
    return KeyPts, Features, Weights

def LoadExtendedKeyPts(strSequence, iFrame):
    KeyPtsDir = str(strDataBaseDir+strSequence+'/KeyPts/')
    keyPtsData = io.loadmat(str(KeyPtsDir+str(iFrame).zfill(6)+'.bin.mat'))
    ExtendedKeyPts = keyPtsData['ExtendedKeyPts']
#    ExtendedKeyPts = keyPtsData['CorrectedExtendedKeyPts']
    PlanarPts = keyPtsData['PlanarPts']
    return ExtendedKeyPts, PlanarPts


def ExtendTheKeyPts(strSequence, iFrame, keyPtsIdx):
    # firstly, get the key points from file
    keyPtsData = io.loadmat(str(KeyPtsDir+str(iFrame).zfill(6)+'.bin.mat'))
    KeyPts = keyPtsData['KeyPts']
    
    voxelFile = strDataBaseDir + strSequence + '/VoxelModel/' + str(iFrame).zfill(6) + '.bin.mat'
    mat = io.loadmat(voxelFile)
    avlBlocksList = mat['avlBlocksList']
    cntVoxelsLength = mat['cntVoxelsLength'].flatten()
    AllVoxels = mat['AllVoxels']
    
    # rebuild blocks model; get the arounded voxels and the corresponding patches
    Blocks = RebuildBlocksWithVoxelList(avlBlocksList, cntVoxelsLength, AllVoxels)
    ExtendedKeyVoxels = GetKeyVoxelsAroundKeyPts(Blocks, KeyPts)
    ExtendedKeyPts = GetKeyPtsFromKeyVoxels(ExtendedKeyVoxels)    
    return True, ExtendedKeyPts


def ReExtractFeatures(strSequence, iFrame, iniR, iniT):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    import keras
    from keras.models import Model, load_model
    PatchEncoder = load_model('EncoderModel4VoxelPatch.h5')
    
    # load raw PC data and corresponding key points data
    PC = LoadPC(strSequence, iFrame)
    oriKeyPts = LoadKeyPts(strSequence, iFrame)
    
    # translate the PC and KeyPts into the relative pose estimated by the previous odometry
    # onte: only ROTATION at first
    PC_ = np.dot(iniR, PC.T).T
    KeyPts_ = np.dot(iniR, oriKeyPts.T).T
    
    Blocks, VoxelModel1, VoxelModel2, avlBlocksList, cntVoxelsLength, AllVoxels, AllVoxels1, AllVoxels2 = Voxelization(PC_)
    KeyPts__, PatchesList = GetKeyPtsAndPatchesFromKeyPts(Blocks,  KeyPts_, VoxelModel1, VoxelModel2)
    Features = GetFeaturesFromPatches(PatchEncoder, PatchesList)
    Weights = np.ones((KeyPts__.shape[0],1),dtype=np.float32)
    
    # translation at last
    KeyPts___ = (KeyPts__.T + iniT).T
    
    return oriKeyPts, KeyPts___, Features, Weights


def GetTransferPairIdx(idx0, idx1):
    transferedIdxes = []
    if idx0.shape[0] < 1 or idx1.shape[0] < 1:
        return transferedIdxes
    idx0 = np.c_[idx0,idx0]
    idx1 = np.c_[idx1,idx1]
    distMatrix = cdist(idx0, idx1, metric='euclidean')
    possibleIdx = np.argmin(distMatrix, axis=1)
    transferedIdxes = []
    for i in range(distMatrix.shape[0]):
        if distMatrix[i,possibleIdx[i]] == 0:
            transferedIdxes.append([i,possibleIdx[i]])
    return transferedIdxes



# the frame of frarmeNum is the start frame to be updated
# the frameNum is based on the index in poses, not in relRTs
def ForwardUpdatePoses(poses, frameNum, newPose, relRs, relTs):
    poses_ = copy.deepcopy(poses)
    relRs_ = copy.deepcopy(relRs)
    relTs_ = copy.deepcopy(relTs)
    
    # 1. update the cur pose and cur RelRT
    poses_[frameNum,:] = newPose
    relR, relT = GetRelRtBetween2Poses(poses_[frameNum-1,:], newPose)
    relRs_[frameNum-1,:,:] = relR
    relTs_[frameNum-1,:] = relT.reshape(3,)
    
    # 2. forward update from the next frame
    for iFrame in range(frameNum+1, poses_.shape[0], 1):
        # pose0
        pose0 = poses_[iFrame-1]
        R0, T0 = GetRtFromOnePose(pose0)   
        # get the cur relRT
        relativeR = relRs_[iFrame-1,:,:]
        relativeT = relTs_[iFrame-1,:].reshape(3,1)                
        # compute pose1 using the previous relRT
        R = np.dot(R0, relativeR)
        T = np.dot(R0, relativeT) + T0        
        RT = np.c_[R,T]
        pose1 = RT.reshape((1,12))
        poses_[iFrame,:] = pose1    
    return poses_, relRs_, relTs_


# update the poses between frame0 and frame1 (nFrame0 < nFram1)
def BackwardUpdatePoses(poses, iFrame0, iFrame1, newPose, relRs, relTs):
    poses_ = copy.deepcopy(poses)
    relRs_ = copy.deepcopy(relRs)
    relTs_ = copy.deepcopy(relTs)
    
    # original pose differ between iFrame1 and iFrame0
    oriDiffR, oriDiffT = GetRelRtBetween2Poses(poses_[iFrame0,:], poses_[iFrame1,:])    
    
    # original pose data
    oriDeltaR, oriDeltaT = GetRelRtBetween2Poses(poses_[iFrame1,:], newPose)
    oriDeltaEulers = RotateMat2EulerAngle_XYZ(oriDeltaR)
    
    # average the diff to each frame between iFrame0 and iFrame1
    nDeltas = iFrame1 - iFrame0
    avgDeltaEulers = oriDeltaEulers/nDeltas*math.pi/180
    avgDeltaT = oriDeltaT/nDeltas
    
    oriPose1 = poses_[iFrame1,:]
    oriR1, oriT1 = GetRtFromOnePose(oriPose1)
    Rs = np.eye(3, dtype=np.float32)
    Ts = np.zeros((3,1), dtype=np.float32)
    for iFrame in range(iFrame0+1, iFrame1+1, 1):        
        # compute Rp and Tp (previous)
        Rp = np.eye(3, dtype=np.float32)
        Tp = np.zeros((3,1), dtype=np.float32)
        for iFramePrev in range(iFrame, iFrame0, -1):
            relR = relRs_[iFramePrev-1,:,:]
            relT = relTs_[iFramePrev-1,:].reshape(3,1)
            Rp = dot(relR, Rp)
            Tp = dot(relR, Tp) + relT
        # compute Rn and Tn (next)
        Rn = np.eye(3, dtype=np.float32)
        Tn = np.zeros((3,1), dtype=np.float32)
        for iFrameNext in range(iFrame1, iFrame, -1):
            relR = relRs_[iFrameNext-1,:,:]
            relT = relTs_[iFrameNext-1,:].reshape(3,1)
            Rn = dot(relR, Rn)
            Tn = dot(relR, Tn) + relT
                    
        # compute Rs and Ts (star); using the original relRT
        # compute current deltaR and deltaT at first
        curDeltaEulers = avgDeltaEulers*(iFrame-iFrame0)
        curDeltaR = EulerAngle2RotateMat(curDeltaEulers[0],curDeltaEulers[1],curDeltaEulers[2],'xyz')
        curDeltaT = avgDeltaT*(iFrame-iFrame0)
        Rs = dot(oriDiffR, curDeltaR)
        Ts = dot(oriDiffR, curDeltaT) + oriDiffT
        
        # sovle current deltaRelRT
        Rp_inv = np.linalg.inv(Rp)
        Rn_inv = np.linalg.inv(Rn)
        deltaRelR = dot(Rp_inv, dot(Rs, Rn_inv))
        deltaRelT = dot(Rp_inv, Ts-Tp) - dot(deltaRelR, Tn)
        
        # update relRs_ and relTs_ of iFrame
        newRelR = dot(relRs_[iFrame-1,:,:], deltaRelR)
        newRelT = dot(relRs_[iFrame-1,:,:], deltaRelT) + relTs_[iFrame-1,:].reshape(3,1)
        relRs_[iFrame-1,:,:] = newRelR
        relTs_[iFrame-1,:] = newRelT.T
        
        #-------update pose_ of iFrame
        # pose0
        pose0 = poses_[iFrame-1,:]
        R0, T0 = GetRtFromOnePose(pose0)        
        # pose1
        R1 = np.dot(R0, newRelR)
        T1 = np.dot(R0, newRelT) + T0
        # reshape to pose format
        RT1 = np.c_[R1,T1]
        pose1 = RT1.reshape((1,12))
        poses_[iFrame,:] = pose1
        
    
    # verify
    verifyR, verifyT = GetRelRtBetween2Poses(poses_[iFrame1,:], newPose)
    verifyEulers = RotateMat2EulerAngle_XYZ(verifyR)
    verifyNormEulers = LA.norm(verifyEulers)
    verifyNormT = LA.norm(verifyT)
    assert verifyNormEulers < 0.01 and verifyNormT < 0.01
    
    
    return poses_, relRs_, relTs_
    


def FixJumpPoses(poses):
    poses_ = copy.deepcopy(poses)
    relRs, relTs, relEulers, diffNormRelEulers, diffNormRelTs = GetDiffRels(poses_)
    
    EulersThreshold = 2.0  # degree
    TsThreshold = 0.5    
    deJumpedFrames = []
    # searh for the pose jump
    # note that the lenth of poses, relRs and diffNormRelEulers are different
    for iFrame in range(2, poses_.shape[0]-1, 1):
        if diffNormRelEulers[iFrame-2] > EulersThreshold or diffNormRelTs[iFrame-2] > TsThreshold:
            # note: to make the function ForwardUpdatePoses to be general, here we compute the new pose at first
            prevRelR = relRs[iFrame-2,:,:]
            prevRelT = relTs[iFrame-2,:].reshape(3,1)
            # pose0
            pose0 = poses_[iFrame-1]
            R0, T0 = GetRtFromOnePose(pose0)                
            # compute pose1 using the previous relRT
            R = np.dot(R0, prevRelR)
            T = np.dot(R0, prevRelT) + T0            
            RT = np.c_[R,T]
            newPose = RT.reshape((1,12))            
            # fix it
            poses_, relRs, relTs = ForwardUpdatePoses(poses_, iFrame, newPose, relRs, relTs)
            print('fixed jump pose at', iFrame)
            deJumpedFrames.append(iFrame)
            # update diffs
            relRs, relTs, relEulers, diffNormRelEulers, diffNormRelTs = GetDiffRels(poses_)      
    print(deJumpedFrames)
    return poses_  


def LoadAllExtendedKeyPts(strSequence, nFrames):
    AllExtendedKeyPts = []
    for iFrame in range(nFrames):
        AllExtendedKeyPts.append(LoadExtendedKeyPts(strSequence, iFrame))
    return AllExtendedKeyPts



def RefinementCore(poses, strSequence, iFrame0, iFrame1, relRs, relTs, inlierThreshold0):
    poses_ = copy.deepcopy(poses)    
    # to get more key points around the transfered key points
    KeyPts0, PlanarPts0 = LoadExtendedKeyPts(strSequence, iFrame0)
    KeyPts1, PlanarPts1 = LoadExtendedKeyPts(strSequence, iFrame1)
    print('num of extended KeyPts =', KeyPts0.shape[0], ',', KeyPts1.shape[0])    
    print('num of PlanarPts =', PlanarPts0.shape[0], ',', PlanarPts1.shape[0])      
    
    # translate the cur keyPts using the pose from odometry
    pose0 = poses[iFrame0,:]
    pose1 = poses[iFrame1,:]
    oriRelR, oriRelT = GetLidarRelRtBetween2Poses(pose0, pose1, R_Tr, T_Tr, R_Tr_inv, T_Tr_inv)
    KeyPts1_ = np.array(((np.dot(oriRelR, KeyPts1.T) + oriRelT).T), dtype=np.float32)
    
    PlanarPts1_ = copy.deepcopy(PlanarPts1)
    PlanarPts1_[:,0:3] = np.array(((np.dot(oriRelR, PlanarPts1[:,0:3].T) + oriRelT).T), dtype=np.float32)
    
    # re-registration using the extended key points
    R_ICP, T_ICP, isSuccess = ICP_Pt2PtAndPt2Plane(KeyPts0, KeyPts1_, PlanarPts0, PlanarPts1_, maxIterTimes=50, minIterTimes=20-1, 
                                  inlierThreshold0=inlierThreshold0, decay_rate0 = 0.9, 
                                  inlierThreshold1=5.0, decay_rate1 = 0.9,
                                  smallShiftThreshold=0.1, ep=0.001)
#    R_ICP, T_ICP, isSuccess = ICP(KeyPts0, KeyPts1_)
    
    # if the match is failed, then continue
    if isSuccess == False:
        return -1, poses_, relRs, relTs
                    
    relativeR = np.dot(R_ICP, oriRelR)
    relativeT = np.dot(R_ICP, oriRelT) + T_ICP  
    
    # if the pose change is too much, then consider this refine is failed
    oriRelEulers = RotateMat2EulerAngle_XYZ(oriRelR)
    relativeEulers = RotateMat2EulerAngle_XYZ(relativeR)
    diffRelEulers = LA.norm(oriRelEulers-relativeEulers)
    diffRelT = LA.norm(oriRelT-relativeT)
    if diffRelEulers > 10 or diffRelT > 5:
        return 0, poses_, relRs, relTs
        
    # otherwise, refine the pose of iFrame1
    # RT of pose0
    R0, T0 = GetRtFromOnePose(pose0)
    # get RT of pose1
    R_poseDiff = np.dot(R_Tr, np.dot(relativeR, R_Tr_inv))
    T_poseDiff = np.dot(R_Tr, np.dot(relativeR, T_Tr_inv) + relativeT) + T_Tr
    R = np.dot(R0, R_poseDiff)
    T = np.dot(R0, T_poseDiff) + T0
    # format to pose1
    RT = np.c_[R,T]
    pose1 = RT.reshape((12,))    
    
    # backward update and forward update
#    poses_ , relRs, relTs = BackwardUpdatePoses(poses, iFrame0, iFrame1, pose1, relRs, relTs)
#    poses_ , relRs, relTs = ForwardUpdatePoses(poses_, iFrame1, pose1, relRs, relTs)
    poses_ , relRs, relTs = ForwardUpdatePoses(poses, iFrame1, pose1, relRs, relTs)
    
    
    if iShowMatchingResult > 0 and iFrame0 > 0:
        ShowMatchingResult(RawDataDir, iFrame0, iFrame1, KeyPts0, KeyPts1, PlanarPts0[:,0:3], PlanarPts1[:,0:3], 
                           0, 0, 0, oriRelR, oriRelT, R_ICP, T_ICP)

    return 1, poses_, relRs, relTs


# iOption: 0, refine frame by frame; 1, refine only for key frames
def RefineOdometry(strSequence, poses__, Tr, iOption, debugInfo, iStartFrame):
    poses___ = copy.deepcopy(poses__)
    relRs, relTs, relEulers, diffNormRelEulers, diffNormRelTs = GetDiffRels(poses___)
    if iGroundTruth > 0:
        poses = debugInfo[0]    
    
    # get all pairs index
    AllPairIdx = []
    for iFrame in range(poses___.shape[0]-1):
        iFrame0 = iFrame
        iFrame1 = iFrame+1
        pairsData = io.loadmat(str(PairsDir+str(iFrame0).zfill(6)+'-'+str(iFrame1).zfill(6)+'.bin.mat'))
        pairsIdx0 = pairsData['inliersIdx0'].flatten()
        pairsIdx1 = pairsData['inliersIdx1'].flatten()
        AllPairIdx.append(pairsIdx0)
        AllPairIdx.append(pairsIdx1)        
    
    # refine the odometry using ICP and extended keyPts
    # the key frames are got from pairs transfer
    nMinTransferPairs = 1
    nMaxTransferFrames_bkp = 20
    nMaxTransferFrames = nMaxTransferFrames_bkp
    iFrame = iStartFrame
    iEndFrame = poses___.shape[0]-2
#    iEndFrame = 0
    t0 = time()
    failedFrames = []
    neighborIcpFrames = []
    while iFrame < iEndFrame:
        iFrame0 = iFrame
        iFrame1 = iFrame+1
        
        curLongestPair = []
        if iOption == 0:
            curLongestPair.append(iFrame0)
            curLongestPair.append(iFrame1)
        if iOption == 1:
            # initiate current longest pair
            pairsData = io.loadmat(str(PairsDir+str(iFrame0).zfill(6)+'-'+str(iFrame1).zfill(6)+'.bin.mat'))
            curLongestPair.append(int(pairsData['iFrame0']))
            curLongestPair.append(int(pairsData['iFrame1']))
            curLongestPair.append(pairsData['inliersIdx0'].flatten())
            curLongestPair.append(pairsData['inliersIdx1'].flatten())        
            # search for the longest pair
            while curLongestPair[3].shape[0] > nMinTransferPairs:
                iFrame0 = curLongestPair[1]
                iFrame1 = curLongestPair[1] + 1
                if iFrame1 >=  poses___.shape[0]-1:
                    break
                pairsData = io.loadmat(str(PairsDir+str(iFrame0).zfill(6)+'-'+str(iFrame1).zfill(6)+'.bin.mat'))
                Idx0 = pairsData['inliersIdx0'].flatten()
                Idx1 = pairsData['inliersIdx1'].flatten()
                transferedIdxes = GetTransferPairIdx(curLongestPair[3], Idx0)
                
                # if the transfered pairs is not enough, then the transfer is stoped
                # and the curLongestPair stays the same
                if len(transferedIdxes) < nMinTransferPairs or curLongestPair[1] - curLongestPair[0] >= nMaxTransferFrames:
                    break
                # while if it is enough, update the curLongestPair
                transferedIdxes = np.array(transferedIdxes)
                curLongestPair[1] = iFrame1
                curLongestPair[2] = curLongestPair[2][transferedIdxes[:,0]]
                curLongestPair[3] = Idx1[transferedIdxes[:,1]]
    
        
        t1 = time()
        print(strSequence, ':', poses__.shape[0], ':', curLongestPair[0],'-',curLongestPair[1])
        reCode, poses___, relRs, relTs = RefinementCore(poses___, strSequence, curLongestPair[0], curLongestPair[1], relRs, relTs, 1.0)        
        t2 = time()
        print(round(t2-t0, 2), 's:', round(t2-t1, 2), 's')
        
        
        # if the match is failed, then continue
        if reCode == -1:
            if curLongestPair[1] - curLongestPair[0] > 1:
                nMaxTransferFrames = 1  # iFrame stays the same
                continue
            else:
                print(curLongestPair[0],'-',curLongestPair[1], 'refine failed')
                failedFrames.append([curLongestPair[0],curLongestPair[1]])
                nMaxTransferFrames = nMaxTransferFrames_bkp
                iFrame += 1
                continue
           
        if reCode == 0:
            if curLongestPair[1] - curLongestPair[0] > 1:
                nMaxTransferFrames = 1  # iFrame stays the same
                continue
            else:
                print(curLongestPair[0],'-',curLongestPair[1], 'refine failed (unreliable)')
                failedFrames.append([curLongestPair[0],curLongestPair[1]])
                nMaxTransferFrames = nMaxTransferFrames_bkp
                iFrame += 1
                continue

        
        iFrame = curLongestPair[1]
        print('refine success, frame length =', curLongestPair[1] - curLongestPair[0])
        nMaxTransferFrames = nMaxTransferFrames_bkp
        
        
        # compare with the groud-truth if needed
        R_poseDiff_ori, T_poseDiff_ori = GetRelRtBetween2Poses(poses__[curLongestPair[0]], poses__[curLongestPair[1]])
        R_poseDiff, T_poseDiff = GetRelRtBetween2Poses(poses___[curLongestPair[0]], poses___[curLongestPair[1]])
        if iGroundTruth > 0:
            R_GT, T_GT = GetRelRtBetween2Poses(poses[curLongestPair[0]], poses[curLongestPair[1]])
            RRE_ori, RTE_ori = ComputeErrorsofRT(R_GT, T_GT, R_poseDiff_ori, T_poseDiff_ori)
            RRE_after, RTE_after = ComputeErrorsofRT(R_GT, T_GT, R_poseDiff, T_poseDiff)            
            print('ori error:', round(RRE_ori,4), round(RTE_ori,4))
            print('cur error:', round(RRE_after,4), round(RTE_after,4))
            debugInfo.append([curLongestPair[0], curLongestPair[1], np.sum(np.abs(T_GT.reshape(1,3))), RRE_ori, RTE_ori, RRE_after, RTE_after])
        else:            
            R_delta = dot(np.linalg.inv(R_poseDiff_ori), R_poseDiff)
            eulers_delta = RotateMat2EulerAngle_XYZ(R_delta)
            T_delta = T_poseDiff - T_poseDiff_ori
            deltaAngle = np.sum(np.abs(eulers_delta.reshape(1,3)))
            deltaT = np.sum(np.abs(T_delta.reshape(1,3)))   
            print('deltaAngle =', round(deltaAngle,2), ', deltaT =', round(deltaT,2))
            debugInfo.append([curLongestPair[0], curLongestPair[1], deltaAngle, deltaT])
            
        print('\n')
    if iGroundTruth > 0:
        del debugInfo[0]  # remove groudtruth data from the list
    aDebugInfo = np.array(debugInfo, dtype=np.float32)
    
    print(failedFrames)
    return poses___, aDebugInfo


def ComputeErrorsofRT(R_GT, T_GT, R_Estimated, T_Estimated):
    errorR = dot(np.linalg.inv(R_GT), R_Estimated)
    eulers_error = RotateMat2EulerAngle_XYZ(errorR)
    T_error = T_Estimated - T_GT    
    RRE = np.sum(np.abs(eulers_error.reshape(1,3)))
    RTE = np.sum(np.abs(T_error.reshape(1,3)))    
    return RRE, RTE



def CloseLoopPipeline(strSequence, poses, Tr, refinementInfo):    
    KeyFrames = np.array(refinementInfo[:,0], dtype=np.int32)
    nKeyFrames = KeyFrames.shape[0]
    poses_ = copy.deepcopy(poses)
    relRs, relTs, relEulers, diffNormRelEulers, diffNormRelTs = GetDiffRels(poses_)
    
    nMaxFrameLength = 30
    
    # initialize
    t0 = time()
    iKeyFrame = 0
    iKeyFrame0 = iKeyFrame
    iKeyFrame1 = iKeyFrame + 2  
    iFrame0 = KeyFrames[iKeyFrame0]
    iFrame1 = KeyFrames[iKeyFrame1]
    
    reCode = 1
    while iKeyFrame1 < nKeyFrames-5:        
            
        t1 = time()    
        reCode, poses_, relRs, relTs = RefinementCore(poses_, strSequence, iFrame0, iFrame1, relRs, relTs, 0.5)
        t2 = time()
        print(round(t2-t0, 2), 's:', round(t2-t1, 2), 's')
    
        print(strSequence, ':', poses_.shape[0], ':', iFrame0, '-', iFrame1,
              'refine code', reCode, 'frame length =', iFrame1 - iFrame0, '\n')
    
#        if reCode > 0 and (iFrame1 - iFrame0) < nMaxFrameLength:
#            iKeyFrame1 += 1
#            iFrame1 = KeyFrames[iKeyFrame1]
#        else:
#            iKeyFrame0 = iKeyFrame1
#            iKeyFrame1 = iKeyFrame0 + 2
#            iFrame0 = KeyFrames[iKeyFrame0]
#            iFrame1 = KeyFrames[iKeyFrame1]
        
        iKeyFrame0 += 2
        iKeyFrame1 = iKeyFrame0 + 2
        iFrame0 = KeyFrames[iKeyFrame0]
        iFrame1 = KeyFrames[iKeyFrame1]
        
    return poses_, list([])
    
    


iShowResult = 1
iShowMatchingResult = 0

# enables: 0, disable;    1, enable;    2, read from file.
iGroundTruth = 1
iEstimatedOdometry = 2
iDejump = 0
iRefineOdometry = 2
iCloseLoop = 0

ErrorAnalysis = 1
AllErrorAngles = np.zeros((1,1), dtype=np.float32)
AllErrorTs = np.zeros((1,1), dtype=np.float32)

#   0, 2     2, 5     5, 8    8, 11    11, 14    14, 18     18, 20     20, 22
# prepare data path
for iSequence in range(0, 11, 1):
    strSequence = str(iSequence).zfill(2)
    RawDataDir = str(strDataBaseDir + strSequence + '/velodyne/')
    KeyPtsDir = str(strDataBaseDir + strSequence + '/KeyPts/')
    FeaturesDataDir = str(strDataBaseDir + strSequence + '/Features/')
    PairsDir = str(strDataBaseDir + strSequence + '/InliersIdx/')
    RefinementDataDir = str(strDataBaseDir + strSequence + '/RefinenmentData/')
    calibFileFullPath = str(strCalibDataDir + strSequence + '/calib_.txt')
    
    # extract calib data
    calib=np.loadtxt(calibFileFullPath)
    Tr=np.array(calib[4,:].reshape(3,4),dtype=np.float32)
    R_Tr=Tr[:,0:3]
    R_Tr_inv=np.linalg.inv(R_Tr)
    T_Tr=Tr[:,3].reshape(3,1)
    T_Tr_inv = -np.dot(R_Tr_inv, T_Tr)
    
    
    # 0. load ground truth poes and estimated poses
    if iGroundTruth > 0:
        poses = np.loadtxt(strGroundTruthPosesDir+strSequence+'.txt')
    if iEstimatedOdometry > 0:
        poses_ = np.loadtxt(strEstimatedPosesDir+strSequence+'.txt')
#        poses_ = np.loadtxt(strEstimatedPosesDir+'1_2-0_'+strSequence+'.txt')
    
    
    # 1. remove jump poses
    if iDejump == 1:
        poses__ = FixJumpPoses(poses_)
        np.savetxt(str(strDejumpyedPosesDir+strSequence+'.txt'), poses__)
    elif iDejump == 2:
        poses__ = np.loadtxt(strDejumpyedPosesDir+strSequence+'.txt')
    
    
    # 2. refine odometry using extended keyPts and ICP
    if iRefineOdometry == 1:
        debugInfo = []        
        if iGroundTruth > 0:
            debugInfo.append(poses)
        poses___, aDebugInfo = RefineOdometry(strSequence, poses__, Tr, iOption=1, debugInfo=debugInfo, iStartFrame = 0)
        np.savetxt((strRefinedPosesDir+strSequence+'.txt'), poses___)
    elif iRefineOdometry == 2:
        poses___ = np.loadtxt(strRefinedPosesDir+strSequence+'.txt')
        RefinementDataDir = str(strDataBaseDir + strSequence + '/RefinenmentData/')
        mat = io.loadmat(RefinementDataDir+'DebugInfo.mat')
        aDebugInfo = mat['aDebugInfo']
    
    
    # 3. close loop
    if iCloseLoop == 1:
        poses____, AllCloseParis = CloseLoopPipeline(strSequence, poses___, Tr, aDebugInfo)
        np.savetxt((strClosedPosesDir+strSequence+'.txt'), poses____)            
    elif iCloseLoop == 2:
        poses____ = np.loadtxt(strClosedPosesDir+strSequence+'.txt')
    
    
    # extract trajectories
    if iGroundTruth > 0:
        trajectory = poses[:,[3,7,11]]
    if iEstimatedOdometry > 0:
        trajectory_ = poses_[:,[3,7,11]]
    if iDejump > 0:
        trajectory__ = poses__[:,[3,7,11]]
    if iRefineOdometry > 0:
        trajectory___ = poses___[:,[3,7,11]]
    if iCloseLoop > 0:
        trajectory____ = poses____[:,[3,7,11]]
    
    if iShowResult > 0 and iCloseLoop == 1 and len(AllCloseParis) > 0:
        fig = mlab.figure(bgcolor=(1, 1, 1), size=(1640, 1300))
        for iPairGroup in range(2):
            pairGroup = np.array(AllCloseParis[iPairGroup], dtype=np.int32)
            if len(pairGroup) <= 0:
                continue
            pairs0 = trajectory___[pairGroup[:,0],:]
            pairs1 = trajectory___[pairGroup[:,1],:]
            pairs1_ = trajectory____[pairGroup[:,1],:]
            ShowTrajactory(fig, poses___, 0.0)
            ShowTrajactory(fig, poses____, 1.0)
            if iPairGroup == 0:
                mlab.quiver3d(pairs1[:,0], pairs1[:,1], pairs1[:,2], \
                                     pairs1_[:,0]-pairs1[:,0], pairs1_[:,1]-pairs1[:,1], pairs1_[:,2]-pairs1[:,2], \
                                     figure=fig, line_width=5, scale_factor=1)
                mlab.quiver3d(pairs1[:,0], pairs1[:,1], pairs1[:,2], \
                                 pairs0[:,0]-pairs1[:,0], pairs0[:,1]-pairs1[:,1], pairs0[:,2]-pairs1[:,2], \
                                 figure=fig, line_width=5, scale_factor=1)
            else:
                mlab.quiver3d(pairs1[:,0], pairs1[:,1], pairs1[:,2], \
                                 pairs0[:,0]-pairs1[:,0], pairs0[:,1]-pairs1[:,1], pairs0[:,2]-pairs1[:,2], \
                                 figure=fig, line_width=0.5, scale_factor=1)
            mlab.title('Close Loop')
            mlab.view(270, 90, 1500, [0,0,0])
            mlab.axes(x_axis_visibility = True)
#        mlab.show()
    
        
    # ------------------- visual comparision
    if iShowResult > 0:
#        fig = mlab.figure(bgcolor=(1, 1, 1), size=(1640, 1300))
        fig = mlab.figure(bgcolor=(0, 0, 0), size=(1640, 1300))
        
        # show trajactory
        if iEstimatedOdometry > 0:
            ShowTrajactory(fig, poses_, 0.0, 0)
        if iDejump > 0:
            ShowTrajactory(fig, poses__, 0.2, 0)
        if iRefineOdometry > 0:
            ShowTrajactory(fig, poses___, 0.5, 0.5)
        if iCloseLoop > 0:
            ShowTrajactory(fig, poses____, 0.9, 0.5)
        if iGroundTruth > 0:
            ShowTrajactory(fig, poses, 1.0, 1.0)
        
        
        # show the different
        if iGroundTruth > 0:
            if iCloseLoop > 0:
                CompareTrajactory(fig, trajectory, trajectory____, 0, 0)
            elif iRefineOdometry > 0:
                CompareTrajactory(fig, trajectory, trajectory___, 0, 0)
            elif iDejump > 0:
                CompareTrajactory(fig, trajectory, trajectory__, 0, 0)
        
        mlab.title('Trajectories')
        mlab.view(270, 90, 1500, [0,0,0])
#        mlab.view(270, 90, 120, [375,0,43])
        mlab.show()

    
    if iGroundTruth > 0 and ErrorAnalysis > 0:
        # the common error analysis
        if iCloseLoop > 0:
            GroundTruthRels, EstimatedRels, errorRelEulers, errorRelTs = GetErrorRTs(poses, poses____, Tr, isPlot=0)
        elif iRefineOdometry > 0:
            GroundTruthRels, EstimatedRels, errorRelEulers, errorRelTs = GetErrorRTs(poses, poses___, Tr, isPlot=1)
        elif iDejump > 0:
            GroundTruthRels, EstimatedRels, errorRelEulers, errorRelTs = GetErrorRTs(poses, poses__, Tr, isPlot=0)
        elif iEstimatedOdometry > 0:
            GroundTruthRels, EstimatedRels, errorRelEulers, errorRelTs = GetErrorRTs(poses, poses_, Tr, isPlot=0)
            
        AllErrorAngles = np.r_[AllErrorAngles, np.sum(np.abs(errorRelEulers), axis=1).reshape(errorRelEulers.shape[0],1)]
        AllErrorTs = np.r_[AllErrorTs, LA.norm(errorRelTs, axis=1).reshape(errorRelTs.shape[0],1)]
        print('\n')
        
    # the error analysis for the refinement
    if iRefineOdometry > 0:
        io.savemat(RefinementDataDir+'DebugInfo.mat', {'aDebugInfo':aDebugInfo})


AllErrorAngles = np.delete(AllErrorAngles, 0, axis=0)
AllErrorTs = np.delete(AllErrorTs, 0, axis=0)

print(np.mean(AllErrorAngles))
print(np.mean(AllErrorTs))









