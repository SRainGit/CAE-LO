# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 15:20:19 2019

@author: yinde
"""

import os
import numpy as np
from scipy import io
from sklearn.neighbors import NearestNeighbors
import math
 
from Transformations import *
from Dirs import *


def GetAllKeyPts(strSequence, iFrameStep, iDataSource):
    KeyPtsDir = str(strDataBaseDir + strSequence + '/KeyPts/')
    calibFileFullPath = str(strCalibDataDir + strSequence + '/calib_.txt')
    poses = np.loadtxt(strGroundTruthPosesDir + strSequence + '.txt')
    
    if iDataSource == 2:
        R90 = EulerAngle2RotateMat(-math.pi/2,0,-math.pi/2,'xyz')
    
    # extract calib data
    calib=np.loadtxt(calibFileFullPath)
    Tr=np.array(calib[4,:].reshape(3,4),dtype=np.float32)
    R_Tr=Tr[:,0:3]
    R_Tr_inv=np.linalg.inv(R_Tr)
    T_Tr=Tr[:,3].reshape(3,1)
    T_Tr_inv = -np.dot(R_Tr_inv, T_Tr)
    
    # nFrames
    fileList = os.listdir(KeyPtsDir)
    nFrames=  len(fileList)
    
    KeyPtsList = []
    for iFrame in range(0,nFrames,iFrameStep):
        pose = np.array([poses[iFrame,[0,1,2,3]],poses[iFrame,[4,5,6,7]],poses[iFrame,[8,9,10,11]]],dtype=np.float32)
        
        # get keypoints from file
        if iDataSource == 0:
            keyPtsData = io.loadmat(str(KeyPtsDir+str(iFrame).zfill(6)+'.bin.mat'))
            KeyPts = keyPtsData['KeyPts']
        elif iDataSource == 1:
#            fileName = str(str3DFeatNetDir + 'Descriptors\\' + strSequence + '\\' + str(iFrame).zfill(6)+'.bin')
            fileName = str(str3DFeatNetDir + 'Descriptors/' + strSequence + '/' + str(iFrame).zfill(6)+'.bin')
            keyPtsData = np.fromfile(fileName, dtype=np.float32, count=-1).reshape([-1,35])
            KeyPts = keyPtsData[:,0:3]
        elif iDataSource == 2:
            fileName = str(strUsipKeyPtsDir + strSequence + '/' + str(iFrame).zfill(6)+'.bin') 
            keyPtsData = np.fromfile(fileName, dtype=np.float32, count=-1).reshape([-1,3])
            KeyPts = keyPtsData
            
         # transfer keypoints into the world frame
        if iDataSource == 0 or iDataSource == 1:
            KeyPts_ = TranslatePtsIntoWorldFrame(pose, Tr, KeyPts)
        elif iDataSource == 2:
            KeyPts = np.dot(R90, KeyPts.T).T
            KeyPts_ = TranslatePtsIntoWorldFrame(pose, Tr, KeyPts)
            
        KeyPtsList.append(KeyPts_)

    return KeyPtsList


def GetPairDistances(KeyPtsList):
    nFrames = len(KeyPtsList)
    distances = np.array([0],dtype=np.float32).reshape(1,1)
    
    for iFrame0 in range(nFrames-1):
        iFrame1 = iFrame0 + 1
        keyPts0 = KeyPtsList[iFrame0]
        keyPts1 = KeyPtsList[iFrame1]
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(keyPts0)
        dists, indices = nbrs.kneighbors(keyPts1)
        distances = np.r_[distances,dists]

    distances = np.delete(distances, 0, axis=0)
    return distances

def ComputeDispersionOfKeypoints(KeyPtsList):
    nFrames = len(KeyPtsList)
    distances = np.array([0],dtype=np.float32).reshape(1,1)
    
    for iFrame0 in range(nFrames):
        keyPts0 = KeyPtsList[iFrame0]
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(keyPts0)
        dists, indices = nbrs.kneighbors(keyPts0)
        distances = np.r_[distances,dists]

    distances = np.delete(distances, 0, axis=0)  # notice the situation of distance=0
    return distances
    
    
    


iGroundTruth = 0

mode = 0
if mode == 0:
    FileNameTitle = 'AccuracyOfKeyPts_'
elif mode == 1:
    FileNameTitle = 'InnerAccuracyOfKeyPts_'


#iFrameSteps = [1, 2, 5, 10]
iFrameSteps = [1]
Discretizations = [0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4]

listSequence = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#listSequence = [0]


for iFrameStep in iFrameSteps:
    for iDataSource in range(2,3,1):  #0, our data; 1, 3DFeatNet; 2, USIP
    #    allDistances = np.array([0],dtype=np.float32).reshape(1,1)
        for iSequence in listSequence:
            strSequence = str(iSequence).zfill(2)
            print(iDataSource, strSequence)
            KeyPtsList = GetAllKeyPts(strSequence, iFrameStep, iDataSource)
            if mode == 0:
                dists = GetPairDistances(KeyPtsList)
            elif mode == 1:
                dists = ComputeDispersionOfKeypoints(KeyPtsList)
            
            allDistances = dists        
                
            counts = []
            preCnt = 0
            for iDis in range(len(Discretizations)):
                distances_ = allDistances/Discretizations[iDis]
                cnt = np.sum(distances_ < 1)
                counts.append(cnt - preCnt)
                preCnt = cnt
                
            cnt = np.sum(distances_ >= 1)
            counts.append(cnt)
                        
            io.savemat(str(strBaseDir + FileNameTitle + str(iFrameStep) + '_' + str(iDataSource) + '_' + strSequence + '.mat'), {'counts':counts})















