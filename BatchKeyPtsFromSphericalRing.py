#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 12:28:54 2019

@author: rain
"""


import os
from scipy import io
import numpy as np
import mayavi.mlab as mlab
import matplotlib.pyplot as plt
from multiprocessing import Process, Manager
from time import time, sleep
import copy
import mayavi

from Voxel import *
from SphericalRing import *


def GetFileList(file_dir):
    '''
    Args:
        file_dir: file directory
    Returns:
        lsit of files
    '''    
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



def GetKeyPts(Image):
    verticalThreshold = 0.2
    
    KeyPts = []
    RespondImage = np.zeros((Image.shape[0],Image.shape[1]), dtype=np.float32)
    for iPixelX, iPixelY in AllPixelIndexList_WithoutWindowEdge:
        # skip the non-empty pixels
        if sum(Image[iPixelX, iPixelY, :]) == 0:
            continue
        oneWindow = Image[iPixelX-WindowRadius:iPixelX+WindowRadius+1, iPixelY-WindowRadius:iPixelY+WindowRadius+1, :]
        if np.max(oneWindow[:,:,2]) - np.min(oneWindow[:,:,2]) < verticalThreshold:
            continue
        
        KeyPts.append([Image[iPixelX, iPixelY, 0], Image[iPixelX, iPixelY, 1], Image[iPixelX, iPixelY, 2]])
        RespondImage[iPixelX, iPixelY] = 1
        
    KeyPts = np.array(KeyPts, dtype=np.float32)
    
    return KeyPts, RespondImage
    


def KeyPtsExtactionPipline(fileName):
    mat = io.loadmat(fileName)
    avlBlocksList = mat['avlBlocksList']
    cntVoxelsLength = mat['cntVoxelsLength'].flatten()
    AllVoxels = mat['AllVoxels']
    AllVoxels1 = mat['AllVoxels1']
    AllVoxels2 = mat['AllVoxels2']
    
    Blocks, VoxelModel1, VoxelModel2 = RebuildVoxelModelWithVoxelList(avlBlocksList, cntVoxelsLength, AllVoxels, AllVoxels1, AllVoxels2)
    avlBlocksList_ = GetInterestBlocks(Blocks, avlBlocksList)
    Blocks__, avlBlocksList__ = GetInterestVoxels(Blocks, avlBlocksList_)
    KeyVoxelsList = GetKeyVoxels(Blocks__, avlBlocksList__)
    
    return KeyVoxelsList


def BatchExtraction(fileList, iThread, flags4MultiProc):
    TargetFolderName='KeyVoxels'
    
    for iFile in range(len(fileList)):
        print(str(len(fileList))+':'+str(iFile))
        rawFileFullPath=fileList[iFile]
        
        # process
        KeyVoxelsList = KeyPtsExtactionPipline(rawFileFullPath)        
        
        # process on path
        baseDir=os.path.dirname(os.path.dirname(rawFileFullPath))
        targetDir=os.path.join(baseDir,TargetFolderName)        
        isFolder = os.path.exists(targetDir)
        if not isFolder:
            os.makedirs(targetDir)
        
        # fulfill name and write
        fileName=os.path.basename(rawFileFullPath)
        targetFullPath=os.path.join(targetDir,fileName)
        io.savemat(targetFullPath, {'KeyVoxelsList':KeyVoxelsList})
        print(targetFullPath)
        
        
    flags4MultiProc[iThread] = 1


def OnPorcess():
    manager = Manager()
    
    for iSequence in range(0,2,1):
        strSequence=str(iSequence).zfill(2)
        dirSequence=str('/media/rain/Win10_F/KITTI_odometry/velodyne/sequences/'+strSequence)
        fileList=GetFileList(dirSequence)
        voxelFileList = [oneFile for oneFile in fileList if oneFile.split("/")[-2]=='VoxelModel']
                
        flags4MultiProc = manager.list([])
        nThreads = 9
        fileLists=[]
        for iList in range(nThreads):
            fileLists.append(voxelFileList[iList:len(voxelFileList):nThreads])
            flags4MultiProc.append(0)
            
#        BatchVoxelization(rawDataLists[0], 0, flags4MultiProc)
        for iThread in range(nThreads):
            t = Process(target=BatchExtraction, args=(fileLists[iThread], iThread, flags4MultiProc))
            t.start()

        while sum(flags4MultiProc)<nThreads:
            sleep(0.5)    

        del flags4MultiProc


OnPorcess()




baseDir = '/media/rain/Win10_F/KITTI_odometry/velodyne/sequences/01/'
iFrame = 498
RawFilePath = baseDir + 'velodyne/' + str(iFrame).zfill(6) + '.bin'
SphericalRingPath = baseDir + 'SphericalRing/' + str(iFrame).zfill(6) + '.bin.mat'

PC = np.fromfile(RawFilePath, dtype=np.float32, count=-1).reshape([-1,4])
mat = io.loadmat(SphericalRingPath)
SphericalRingImage = mat['SphericalRing']
t0 = time()

# rebuild the voxel models
KeyPts, RespondImage = GetKeyPts(SphericalRingImage)
print('cntkeyPts =', KeyPts.shape[0])
t1 = time()
print(round(t1-t0, 2), 's, GetKeyPixels')


#t5 = time()
#print(round(t5-t4, 2), 's')


color0 = np.ones((PC.shape[0],1), dtype=np.float32)*0.0
color3 = np.ones((KeyPts.shape[0],1), dtype=np.float32)*0.9

#fig = mlab.figure(bgcolor=(0, 0, 0), size=(1640, 500))
#mlab.imshow(SphericalRingImage)    
#mlab.view(270, 0, 1200, [0,0,0])

fig = mlab.figure(bgcolor=(0, 0, 0), size=(1640, 500))
mlab.imshow(RespondImage)    
mlab.view(270, 0, 1200, [0,0,0])


fig = mayavi.mlab.figure(bgcolor=(0, 0, 0), size=(1640, 1500))
node_PC = mayavi.mlab.points3d(PC[:,0], PC[:,1], PC[:,2],
                     mode="point",  figure=fig)
node_PC.mlab_source.dataset.point_data.scalars = color0


node_KeyPC=mayavi.mlab.points3d(KeyPts[:,0], KeyPts[:,1], KeyPts[:,2],
                     scale_factor=0.15, figure=fig)
node_KeyPC.glyph.scale_mode = 'scale_by_vector'
node_KeyPC.mlab_source.dataset.point_data.scalars = color3


mayavi.mlab.show()


















