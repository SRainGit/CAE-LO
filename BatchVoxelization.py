#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 11:06:49 2019

@author: rain
"""
# https://blog.csdn.net/weixin_39999955/article/details/83819196


import os
from multiprocessing import Process, Manager, freeze_support
from scipy import io
import numpy as np
import mayavi.mlab
from Voxel import VoxelSize, VisibleLength, VisibleWidth, VisibleHeight
from Voxel import Voxelization, VoxelModel2PC_3Scales#, VoxelModel2ColofulPC
from Voxel import PatchSize, PatchRadius
from numpy import linalg as LA
from time import time, sleep

from Dirs import *


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
    

def BatchVoxelization(fileList, iThread, flags4MultiProc):
    TargetFolderName='VoxelModel'
    
    for iFile in range(len(fileList)):
        print(str(len(fileList))+':'+str(iFile))
        rawFileFullPath=fileList[iFile]
        PC = np.fromfile(rawFileFullPath, dtype=np.float32, count=-1).reshape([-1,4])
        
        # Voxelization
        Blocks, VoxelModel1, VoxelModel2, avlBlocksList, cntVoxelsLength, AllVoxels, AllVoxels0, AllVoxels1, AllVoxels2 = Voxelization(PC)
    
        baseDir=os.path.dirname(os.path.dirname(rawFileFullPath))
        targetDir=os.path.join(baseDir,TargetFolderName)        
        isFolder = os.path.exists(targetDir)
        if not isFolder:
            os.makedirs(targetDir)
                    
        fileName=os.path.basename(rawFileFullPath)+'.mat'
        targetFullPath=os.path.join(targetDir,fileName)
        io.savemat(targetFullPath, {'avlBlocksList':avlBlocksList, 'cntVoxelsLength':cntVoxelsLength, 'AllVoxels':AllVoxels, 
                                    'AllVoxels0':AllVoxels0, 'AllVoxels1':AllVoxels1, 'AllVoxels2':AllVoxels2})
        print(targetFullPath)
        
    flags4MultiProc[iThread] = 1
        


def Porcess():
    manager = Manager()
    
    for iSequence in range(8,22,1):
        strSequence=str(iSequence).zfill(2)
        dirSequence=str(strDataBaseDir+strSequence)
        rawDataList=GetFileList(dirSequence)
        rawDataList = [oneFile for oneFile in rawDataList if oneFile[(len(oneFile)-3):len(oneFile)]=='bin']
                
        flags4MultiProc = manager.list([])
        nThreads = 11
        rawDataLists=[]
        for iList in range(nThreads):
            rawDataLists.append(rawDataList[iList:len(rawDataList):nThreads])
            flags4MultiProc.append(0)
            
#        BatchVoxelization(rawDataLists[0], 0, flags4MultiProc)
        for iThread in range(nThreads):
            t = Process(target=BatchVoxelization, args=(rawDataLists[iThread], iThread, flags4MultiProc))
            t.start()
            
        while sum(flags4MultiProc)<nThreads:
            sleep(0.5)    

        del flags4MultiProc



if __name__ == "__main__":
    freeze_support()
    Porcess()
    
    
    
    #----------------(for test at first) Visualization of Voxelmodel---------------------------------------------
    PC = np.fromfile(strDataBaseDir + '00/velodyne/000000.bin', dtype=np.float32, count=-1).reshape([-1,4])
    t0=time()
    Blocks, VoxelModel1, VoxelModel2, avlBlocksList, cntVoxelsLength, AllVoxels, AllVoxels0, AllVoxels1, AllVoxels2 = Voxelization(PC)
    
    t1=time()
    VoxelPC, PC1, PC2 = VoxelModel2PC_3Scales(avlBlocksList, cntVoxelsLength, AllVoxels, AllVoxels1, AllVoxels2)
    t2=time()
    
    print('nVoxels =', VoxelPC.shape[0])
    print(round(t1-t0, 2), 's')
    print(round(t2-t1, 2), 's')
    
    
    SingleColor0=np.ones((PC.shape[0],1), dtype=np.float32)*1.0
    SingleColor1=np.ones((VoxelPC.shape[0],1), dtype=np.float32)*0.6
    color1 = np.ones((PC1.shape[0],1), dtype=np.float32)*0.3
    color2 = np.ones((PC2.shape[0],1), dtype=np.float32)*0.0
    
    fig = mayavi.mlab.figure(bgcolor=(0, 0, 0), size=(1640, 1500))
    
    node_PC = mayavi.mlab.points3d(PC[:,0], PC[:,1], PC[:,2],
                         mode="point",figure=fig)
    node_PC.mlab_source.dataset.point_data.scalars = SingleColor0
    
    node_VoxelPC=mayavi.mlab.points3d(VoxelPC[:,0], VoxelPC[:,1], VoxelPC[:,2],
                         mode="point",figure=fig)
    node_VoxelPC.glyph.scale_mode = 'scale_by_vector'
    node_VoxelPC.mlab_source.dataset.point_data.scalars = SingleColor1
    
    node_PC1=mayavi.mlab.points3d(PC1[:,0], PC1[:,1], PC1[:,2],
                         scale_factor=0.02, figure=fig)
    node_PC1.glyph.scale_mode = 'scale_by_vector'
    node_PC1.mlab_source.dataset.point_data.scalars = color1
    
    node_PC2=mayavi.mlab.points3d(PC2[:,0], PC2[:,1], PC2[:,2],
                         scale_factor=0.1, figure=fig)
    node_PC2.glyph.scale_mode = 'scale_by_vector'
    node_PC2.mlab_source.dataset.point_data.scalars = color2
    
    mayavi.mlab.show()
    

 

















