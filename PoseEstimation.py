#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 12:37:16 2019

@author: rain
"""

import os
import numpy as np
from numpy import dot
from numpy import linalg as LA
from scipy import io
from multiprocessing import Pool
from threading import Thread
from multiprocessing import Process, Manager, Value
from time import time, sleep
import math

from Voxel import *
from Match import *
from Transformations import *



def GeneratorThreadProc(iThread, iKeyPtSource, DataDir, iFrame, listPreProcData, flags4MultiProc):
    fileFullPath = str(DataDir+str(iFrame).zfill(6)+'.bin')
    KeyPts, AllVoxels0, AllVoxels1, AllVoxels2 = LoadVoxelModelAndKeyPts(fileFullPath)
    
    if iKeyPtSource != 0:
        strCurrentSequence = DataDir.split('/')[-3]
        if iKeyPtSource == 1:
            pathKeyPts1 = os.path.join(dirKeyPts3DFeatNet, strCurrentSequence, str(iFrame).zfill(6) + '.bin')
            KeyPts = np.fromfile(pathKeyPts1, dtype=np.float32, count=-1).reshape([-1, 3+FEATURE_DIMENSION_1])
            KeyPts = KeyPts[:,0:3]
        elif iKeyPtSource == 2:
            pathKeyPts2 = os.path.join(strUsipKeyPtsDir, strCurrentSequence, str(iFrame).zfill(6) + '.bin')
            KeyPts = np.fromfile(pathKeyPts2, dtype=np.float32, count=-1).reshape([-1, 3])
            KeyPts = np.dot(R90, KeyPts.T).T
            
    KeyPts, PatchesList = GetPatchesList(KeyPts, AllVoxels0, AllVoxels1, AllVoxels2)
    
    listPreProcData[iThread].append(KeyPts)
    listPreProcData[iThread].append(PatchesList)
    flags4MultiProc[iThread] = 1
    

def KeyPtsDataGenerator(isLoadFeaturesFromFile, iKeyPtSource, DataDir, listKeyPtsData, nPreparedFrames):  
    # ----------------- if loading features from file    
    if isLoadFeaturesFromFile == True:
        baseDir=os.path.dirname(os.path.dirname(DataDir))
        featuresBaseDir = os.path.join(baseDir,'Features')  
        fileList = os.listdir(RawDataDir)
        nFrames = len(fileList)
        for iFrame in range(nFrames):
            featuresFileFullPath = str(featuresBaseDir+'/'+str(iFrame).zfill(6)+'.bin.mat')
            mat = io.loadmat(featuresFileFullPath)
            KeyPts = mat['KeyPts']
            Features = mat['Features']
            Weights = mat['Weights']
            listKeyPtsData[iFrame].append(KeyPts)
            listKeyPtsData[iFrame].append(Features)
            listKeyPtsData[iFrame].append(Weights)
            nPreparedFrames[0] += 1
            print('nPreparedFrames =', nPreparedFrames[0])
        return listKeyPtsData


    # ----------------- if not loading features data from file
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    import keras
    from keras.models import Model, load_model
    PatchEncoder = load_model('EncoderModel4VoxelPatch.h5')
    
    fileList = os.listdir(RawDataDir)
    nFrames = len(fileList)
    
    # multi threads for extracting data from files in parallel
    nThreads = 3
    manager = Manager()
    while nPreparedFrames[0] + nThreads <= nFrames:
        listPreProcData = manager.list([])
        flags4MultiProc = manager.list([])
        
        # initialize manager's list
        for iThread in range(nThreads):
            listPreProcData.append(manager.list([]))
            flags4MultiProc.append(0)
        
        # let the processes work
        for iThread in range(nThreads):
            iFrame = nPreparedFrames[0] + iThread
            t = Process(target=GeneratorThreadProc, args=(iThread, iKeyPtSource, DataDir, iFrame, listPreProcData, flags4MultiProc))
            t.start()
#        GeneratorThreadProc(0, iKeyPtSource, DataDir, 0, listPreProcData, flags4MultiProc)
            
        # wait for the processes
        while sum(flags4MultiProc) < nThreads:
            sleep(0.01)
        
        # get feature map and keyPts in this fuction's process
        # and fill them into the global list
        for iThread in range(nThreads):
            KeyPts, PatchesList = listPreProcData[iThread]
            Features = GetFeaturesFromPatches(PatchEncoder, PatchesList)
            Weights = np.ones((KeyPts.shape[0],1),dtype=np.float32)
            iFrame = nPreparedFrames[0] + iThread
            listKeyPtsData[iFrame].append(KeyPts)
            listKeyPtsData[iFrame].append(Features)
            listKeyPtsData[iFrame].append(Weights)
        
        
        nPreparedFrames[0] += nThreads
        print('\nnPreparedFrames =', nPreparedFrames[0])
        
        for iThread in range(nThreads):
            del listPreProcData[nThreads-iThread-1][:]
        del flags4MultiProc[:]
        del listPreProcData, flags4MultiProc
        
        
    # if have tails
    while nPreparedFrames[0] < nFrames:
        iFrame = nPreparedFrames[0]
        fileFullPath = str(DataDir+str(iFrame).zfill(6)+'.bin')
        KeyPts, AllVoxels0, AllVoxels1, AllVoxels2 = LoadVoxelModelAndKeyPts(fileFullPath)
            
        if iKeyPtSource != 0:
            strCurrentSequence = DataDir.split('/')[-3]
            if iKeyPtSource == 1:
                pathKeyPts1 = os.path.join(dirKeyPts3DFeatNet, strCurrentSequence, str(iFrame).zfill(6) + '.bin')
                KeyPts = np.fromfile(pathKeyPts1, dtype=np.float32, count=-1).reshape([-1, 3+FEATURE_DIMENSION_1])
                KeyPts = KeyPts[:,0:3]
            elif iKeyPtSource == 2:
                pathKeyPts2 = os.path.join(strUsipKeyPtsDir, strCurrentSequence, str(iFrame).zfill(6) + '.bin')
                KeyPts = np.fromfile(pathKeyPts2, dtype=np.float32, count=-1).reshape([-1, 3])
                KeyPts = np.dot(R90, KeyPts.T).T
                
        KeyPts, PatchesList = GetPatchesList(KeyPts, AllVoxels0, AllVoxels1, AllVoxels2)
        
        Features = GetFeaturesFromPatches(PatchEncoder, PatchesList)
        Weights = np.ones((KeyPts.shape[0],1),dtype=np.float32)
        listKeyPtsData[iFrame].append(KeyPts)
        listKeyPtsData[iFrame].append(Features)
        listKeyPtsData[iFrame].append(Weights)
        nPreparedFrames[0] += 1    
        
    print('\nnPreparedFrames =', nPreparedFrames[0])
    return listKeyPtsData
   

def GetRelativePoseBetween2Frames(iFrame0, iFrame1):
    # get keyPtsData0 from the data in iFrame0
    KeyPts0, Features0, Weights0 = listKeyPtsData[iFrame0]
    print('nKeyPts0 =', KeyPts0.shape[0])    
    
    # get keyPtsData1
    KeyPts1, Features1, Weights1 = listKeyPtsData[iFrame1]
    print('nKeyPts1 =', KeyPts1.shape[0])
    
    # solve the delta pose
    relativeR, relativeT, isSuccess, inliersIdx0, inliersIdx1, residualThreshold = SolveRelativePose(KeyPts0, Features0, Weights0, KeyPts1, Features1, Weights1)
    relativeT = relativeT.reshape(3,1)
    
    inliersData.append([iFrame0, iFrame1, inliersIdx0, inliersIdx1])
    
    nInliers = inliersIdx0.shape[0]
    return relativeR, relativeT, isSuccess, nInliers, residualThreshold
        
    
    

if __name__ == "__main__":
    isLoadFeaturesFromFile = False
    #isLoadFeaturesFromFile = True
    
    dirKeyPts3DFeatNet = '/media/rain/Win10_F/KITTI_odometry/output_3DFeatNet/Descriptors/'
    FEATURE_DIMENSION_1 = 32
    R90 = EulerAngle2RotateMat(-math.pi/2,0,-math.pi/2,'xyz')
    
    #nHours = 11
    #for iHour in range(nHours):
    #    print('iHour =', iHour)
    #    sleep(1*60*60)  # hour*min*sec
    
    
    # 0, Ours; 1, 3DFeatNet; 2, USIP
    #for iKeyPtSource in [1, 2]:
    for iKeyPtSource in [0]:
        
        #listSequence = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17 ,18, 19, 20, 21]
        listSequence = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    #    listSequence = [11, 12, 13, 14, 15, 16, 17 ,18, 19, 20, 21]
        # listSequence = [5]
        for iSequence in listSequence:
            # prepare data path
            strSequence=str(iSequence).zfill(2)
            RawDataDir = str(strDataBaseDir + strSequence + '/velodyne/')
            calibFileFullPath = str(strCalibDataDir + strSequence + '/calib_.txt')
            
            # extract calib data
            calib=np.loadtxt(calibFileFullPath)
            Tr=np.array(calib[4,:].reshape(3,4),dtype=np.float32)
            R_Tr=Tr[:,0:3]
            R_Tr_inv=np.linalg.inv(R_Tr)
            T_Tr=Tr[:,3].reshape(3,1)
            T_Tr_inv = -np.dot(R_Tr_inv, T_Tr)
            
            # nFrames
            fileList = os.listdir(RawDataDir)
            nFrames=  len(fileList)
            
            # start to prepare data
            manager = Manager()  
            listKeyPtsData = manager.list([])
            nPreparedFrames = manager.list([])
            nPreparedFrames.append([])
            nPreparedFrames[0] = 0
            for i in range(nFrames):
                listKeyPtsData.append(manager.list([]))        
            t = Process(target=KeyPtsDataGenerator, args=(isLoadFeaturesFromFile, iKeyPtSource, RawDataDir, listKeyPtsData, nPreparedFrames))
            t.start()
    #        KeyPtsDataGenerator(isLoadFeaturesFromFile, iKeyPtSource, RawDataDir, listKeyPtsData, nPreparedFrames)
                
            # waiting for data
            while nPreparedFrames[0] < 1:
                print('nPreparedFrames =', nPreparedFrames[0])
                sleep(1)        
                
            # initialize poses
            poses=[]
            pose0=np.array([1,0,0,0,  0,1,0,0,  0,0,1,0],dtype=np.float32).reshape(12,1)
            poses.append(pose0)
            R0, T0 = GetRtFromOnePose(pose0)
            inliersData = []
            
            t0 = time()
            iFrame0 = 0
            iFrame1 = 1
        #    nFrames = 1000
            for iFrame0 in range(nFrames-1):
                iFrame1 = iFrame0 + 1
                while nPreparedFrames[0]-1 < iFrame1:
                    sleep(0.5)  # no need to wait too long
                    
                # get relative pose between iFrame0 and iFrame1
                print('\n')
                print(strSequence+':'+str(str(nFrames)+':'+str(iFrame0)+'-'+str(iFrame1)))
                t1 = time()
                relativeR, relativeT, isSuccess, nInliers, residualThreshold = GetRelativePoseBetween2Frames(iFrame0, iFrame1)
                
                # pose0
                pose0 = poses[iFrame0]
                R0, T0 = GetRtFromOnePose(pose0)
                R0_inv = np.linalg.inv(R0)            
                
                # get pose1
                R_poseDiff = np.dot(R_Tr, np.dot(relativeR, R_Tr_inv))
                T_poseDiff = np.dot(R_Tr, np.dot(relativeR, T_Tr_inv) + relativeT) + T_Tr
                R = np.dot(R0, R_poseDiff)
                T = np.dot(R0, T_poseDiff) + T0        
                
                # reshape to pose format
                RT = np.c_[R,T]
                pose1 = RT.reshape((12,1))
                poses.append(pose1)
                
                t4 = time()
                print(round(t4-t0, 2), 's:', round(t4-t1, 2), 's')
                
                
            poses=np.array(poses,dtype=np.float32)
            poses=poses.reshape(poses.shape[0],12)    
            
            
            # save the poses data
            np.savetxt(str(strEstimatedPosesDir+strSequence+'.txt'),poses)
            
            # save the features and match pairs data
            if isLoadFeaturesFromFile == False:
                if iKeyPtSource == 0:
                    FeatruesDataDir = str(strDataBaseDir+strSequence+'/Features/')
                elif iKeyPtSource == 1:
                    FeatruesDataDir = str(strDataBaseDir+strSequence+'/Features-3DFeatNet/')
                elif iKeyPtSource == 2:
                    FeatruesDataDir = str(strDataBaseDir+strSequence+'/Features-USIP/')        
                
                isFolder = os.path.exists(FeatruesDataDir)
                if not isFolder:
                    os.makedirs(FeatruesDataDir)
                for iFrame in range(nFrames):
                    KeyPts, Features, Weights = listKeyPtsData[iFrame]
                    fileFullPath = str(FeatruesDataDir+str(iFrame).zfill(6)+'.bin.mat')
                    io.savemat(fileFullPath, {'KeyPts':KeyPts, 'Features':Features, 'Weights':Weights})
            
            InliersDataDir = str(strDataBaseDir+strSequence+'/InliersIdx/')
            isFolder = os.path.exists(InliersDataDir)
            if not isFolder:
                os.makedirs(InliersDataDir)
            if iKeyPtSource == 0:
                for iFrame in range(len(inliersData)):
                    iFrame0 = inliersData[iFrame][0]
                    iFrame1 = inliersData[iFrame][1]
                    inliersIdx0 = inliersData[iFrame][2]        
                    inliersIdx1 = inliersData[iFrame][3]
                    fileFullPath = str(InliersDataDir+str(iFrame0).zfill(6)+'-'+str(iFrame1).zfill(6)+'.bin.mat')
                    io.savemat(fileFullPath, {'iFrame0':iFrame0, 'iFrame1':iFrame1, 
                                              'inliersIdx0':inliersIdx0, 'inliersIdx1':inliersIdx1})
            
            del listKeyPtsData 
            












