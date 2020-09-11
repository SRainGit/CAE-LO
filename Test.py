#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 09:52:04 2019

@author: rain
"""

import os
import numpy as np
import copy
from time import time
from sklearn.neighbors import NearestNeighbors

from Dirs import *
from Voxel import *


def GetFileList(file_dir):
    fileList=[]
    if not os.path.exists(file_dir):
        print('Wrong path!')
        return        
    # get file list
    for root, dirs, files in os.walk(file_dir, topdown=False):
        for name in files:
            fileFullPath=os.path.join(root, name)
            fileList.append(fileFullPath)            
    return fileList



def GetPatches(strSequence, Frames):
    oriDataBaseDir = '/media/rain/Win10_F/KITTI_odometry/velodyne/sequences/'
    rawDataFolderName = 'velodyne'
    oriRawDataDir = os.path.join(oriDataBaseDir, strSequence, rawDataFolderName)
    nFiles = Frames.shape[0]   
    
    patches = []
    for iData in range(nFiles):
        fileName = str(Frames[iData]).zfill(6) + '.bin'
        oriRawFileFullPath = os.path.join(oriRawDataDir, fileName)
        PC = np.fromfile(oriRawFileFullPath, dtype=np.float32, count=-1).reshape([-1,4])[:,0:3]
        
        RandIdxes=np.random.random((800,1))
        RandIdxes=RandIdxes*(PC.shape[0]-1)+1
        RandIdxes=np.array(RandIdxes,dtype=int)
        
        pts = PC[RandIdxes[:,0],0:3]        
        
        t0 = time()
        nbrs = NearestNeighbors(n_neighbors=512,radius=0.2, algorithm='auto').fit(PC)
        distances, indices = nbrs.kneighbors(pts,return_distance=True) 
        
        neighbors = PC[indices,:]
        t1 = time()
        print(round(t1-t0, 2))
            
    return patches


maxVolume = 0
finalResult = []
for iSequence in range(1, 2, 1):
    strSequence=str(iSequence).zfill(2)
    dirSequence=str(strDataBaseDir + strSequence)
    rawDataList = GetFileList(dirSequence)
    rawDataList = [oneFile for oneFile in rawDataList if oneFile[(len(oneFile)-3):len(oneFile)]=='bin']
    nFiles = len(rawDataList)
    AllFrames = np.arange(0, nFiles, 1, dtype=np.int32)
    
    patches = GetPatches(strSequence, AllFrames)
    
        





















