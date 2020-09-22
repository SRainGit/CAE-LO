#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 11:06:49 2019

@author: rain
"""


import os
from multiprocessing import Process, Manager, freeze_support
from scipy import io
import numpy as np
import mayavi.mlab as mlab
import matplotlib.pyplot as plt
from numpy import linalg as LA
from time import time, sleep
from PIL import Image

from Dirs import *
from SphericalRing import *
from Match import *
from Transformations import *


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


def BatchProjection(strSequence, Frames, iProc,  iThread, flags4MultiProc):
    rawDataFolderName = 'velodyne'
    TargetFolderName = 'SphericalRing'
    rawDataDir = os.path.join(strDataBaseDir, strSequence, rawDataFolderName)
    rawDataList = GetFileList(strDataBaseDir+strSequence)
    rawDataList = [oneFile for oneFile in rawDataList if oneFile[(len(oneFile)-3):len(oneFile)]=='bin']
    nFramesInSequence = len(rawDataList)
    
    nFiles = Frames.shape[0]
    
    for iData in range(nFiles ):
        fileName = str(Frames[iData]).zfill(6) + '.bin'
        rawFileFullPath = os.path.join(rawDataDir, fileName)
        PC = np.fromfile(rawFileFullPath, dtype=np.float32, count=-1).reshape([-1,4])
        
        # process
        SphericalRing, GridCounter = ProjectPC2SphericalRing(PC)
    
        baseDir=os.path.dirname(os.path.dirname(rawFileFullPath))
        targetDir=os.path.join(baseDir,TargetFolderName)        
        isFolder = os.path.exists(targetDir)
        if not isFolder:
            os.makedirs(targetDir)
                    
        fileName=os.path.basename(rawFileFullPath)+'.mat'
        targetFullPath=os.path.join(targetDir,fileName)
        io.savemat(targetFullPath, {'SphericalRing':SphericalRing, 'GridCounter':GridCounter})
        print(strSequence, ':', nFramesInSequence, ':', Frames[iData], '\n', targetFullPath) 
        
    flags4MultiProc[iThread] = 1
    

def BatchCorrectPC(strSequence, Frames, iProc,  iThread, flags4MultiProc):
    oriDataBaseDir = '/media/rain/Win10_H/KITTI_odometry/velodyne/sequences/'
    curDataBaseDir = '/media/rain/Win10_F/KITTI_odometry/velodyne/sequences/'
    rawDataFolderName = 'velodyne'
    oriRawDataDir = os.path.join(oriDataBaseDir, strSequence, rawDataFolderName)
    curRawDataDir = os.path.join(curDataBaseDir, strSequence, rawDataFolderName)    
    nFiles = Frames.shape[0]    
    for iData in range(nFiles ):
        fileName = str(Frames[iData]).zfill(6) + '.bin'
        oriRawFileFullPath = os.path.join(oriRawDataDir, fileName)
        curRawFileFullPath = os.path.join(curRawDataDir, fileName)
        PC = np.fromfile(oriRawFileFullPath, dtype=np.float32, count=-1).reshape([-1,4])        
        PC_ = CorrectPC(PC[:,0:3])
        PC_ = np.c_[PC_, PC[:,3]]
        PC_ = PC_.flatten()        
        with open(curRawFileFullPath, 'wb') as f:
            f.write(PC_)        
        print(strSequence, ':',  iProc, ':', iThread, ':', nFiles, ':', iData, ':', '\n', curRawFileFullPath)        
    flags4MultiProc[iThread] = 1
    
    
def GetAllRespondImgs(strSequence, subFrames, RespondLayer):
    DataFolderName = 'SphericalRing'
    SphericalRingDir = os.path.join(strDataBaseDir, strSequence, DataFolderName)
    nFrames = subFrames.shape[0]
    
    # get all spherical rings and grid counters
    SphericalRings = np.zeros((nFrames, nLines, ImgW-CropWidth_SphericalRing, len(Channels4AE)), dtype=np.float32)
    GridCounters = np.zeros((nFrames, ImgH, ImgW), dtype=np.int8)    
    for iData in range(nFrames):
        fileName = str(subFrames[iData]).zfill(6) + '.bin.mat'
        fileFullPath = os.path.join(SphericalRingDir, fileName)
        mat = io.loadmat(fileFullPath)
        SphericalRing = mat['SphericalRing']
        GridCounter = mat['GridCounter']        
        SphericalRings[iData,:,:,:] = np.array(SphericalRing[0:nLines, 0:ImgW-CropWidth_SphericalRing, Channels4AE], dtype=np.float32)
        GridCounters[iData,:,:] = GridCounter
    print('finished loading data, predicting...')
    
    # predict
    RespondImgs = np.array(RespondLayer.predict(SphericalRings), dtype=np.float32)
    print('finished predicting, extracting keyPts...')
    
    RespondsList = []
    RespondsList.append(RespondImgs)
    
    return SphericalRings, GridCounters, RespondsList


def BatchGetKeyPts(strSequence, FrameList, SphericalRings, GridCounters, RespondImgs, iProc, iThread, flags4MultiProc):
    KeyPtFolderName = 'KeyPts'
    FeaturesFolderName = 'Features'
    rawDataList = GetFileList(os.path.join(strDataBaseDir, strSequence))
    rawDataList = [oneFile for oneFile in rawDataList if oneFile[(len(oneFile)-3):len(oneFile)]=='bin']
    nFramesInSequence = len(rawDataList)
    
    nDatas = len(FrameList)
    
    for iData in range(nDatas):
        iFrame = FrameList[iData]
        
        SphericalRing = SphericalRings[iData,:,:,:]
        GridCounter = GridCounters[iData,:,:]
        RespondImg = RespondImgs[iData,:,:,:]
        
        # extract key points
        KeyPts, KeyPixels, PlanarPts = GetKeyPtsByAE(SphericalRing, GridCounter, RespondImg)        
#        # extract features of the key points
#        Features = GetFeaturesFromSphericalRing(RespondImg, KeyPixels)
#        Weights = np.ones((KeyPts.shape[0],1),dtype=np.float32)
        
        # extend key points for the final pose refine
        ExtendedKeyPts = ExtendKeyPtsInShpericalRing(SphericalRing, GridCounter, KeyPixels)
        
        
        # save key points
        keyPtDir = os.path.join(strDataBaseDir, strSequence, KeyPtFolderName)      
        isFolder = os.path.exists(keyPtDir)
        if not isFolder:
            os.makedirs(keyPtDir)                    
        fileName = str(iFrame).zfill(6)+'.bin.mat'        
        ketPtFullPath = os.path.join(keyPtDir, fileName)
        io.savemat(ketPtFullPath, {'KeyPts':KeyPts, 'ExtendedKeyPts':ExtendedKeyPts, 'PlanarPts':PlanarPts})
        print(strSequence, ':', nFramesInSequence, ':', iFrame, '\n', ketPtFullPath)
        
#        # save features
#        featuresDir = os.path.join(strDataBaseDir, strSequence, FeaturesFolderName)      
#        isFolder = os.path.exists(featuresDir)
#        if not isFolder:
#            os.makedirs(featuresDir)                    
#        fileName = str(iFrame).zfill(6)+'.bin.mat'        
#        featuresFullPath = os.path.join(featuresDir, fileName)
#        io.savemat(featuresFullPath, {'KeyPts':KeyPts, 'Features':Features, 'Weights':Weights})
#        print(strSequence, ':', nFramesInSequence, ':', iFrame, '\n', featuresFullPath)
        
    flags4MultiProc[iThread] = 1


def BatchPorcess(iOption):
    manager = Manager()
    if iOption == 2:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"    
        import tensorflow as tf
        from keras.backend.tensorflow_backend import set_session
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.4
        set_session(tf.Session(config=config))
        import keras
        from keras.models import load_model
        RespondLayer = load_model('SphericalRingPCRespondLayer.h5')
    
    for iSequence in range(9, 11, 1):
        strSequence=str(iSequence).zfill(2)
        dirSequence=str(os.path.join(strDataBaseDir, strSequence))
        rawDataList = GetFileList(dirSequence)
        rawDataList = [oneFile for oneFile in rawDataList if oneFile[(len(oneFile)-3):len(oneFile)]=='bin']
        nFiles = len(rawDataList)
        
        AllFrames = np.arange(0, nFiles, 1, dtype=np.int32)
        
        # divide the whole frames into several parts
        # because going to put a whole part data into memory for opt2
        nProcs = 1
        if iOption == 2:
            nMaxFilesPerProc = 600
            nProcs = int(nFiles/nMaxFilesPerProc) + 1
        nFilesPerProc = int(nFiles/nProcs)
        for iProc in range(nProcs):
            iStartProcFrame = iProc*nFilesPerProc
            iEndProcFrame = (iProc+1)*nFilesPerProc
            if iProc+1 == nProcs:
                iEndProcFrame = nFiles
            subFrames = AllFrames[slice(iStartProcFrame, iEndProcFrame, 1)]
            nSubFrames = subFrames.shape[0]
            
            flags4MultiProc = manager.list([])
            nThreads = 8
            FrameLists = []
            for iList in range(nThreads):
                slices = slice(iList,nSubFrames,nThreads)
                FrameLists.append(subFrames[slices])
                flags4MultiProc.append(0)
                
            if iOption == 2:
                SphericalRings, GridCounters, RespondsList = GetAllRespondImgs(strSequence, subFrames, RespondLayer)
                RespondImgs = RespondsList[0]
                
                SphericalRingLists = []
                GridCounterLists = []
                RespondImgLists = []
                for iList in range(nThreads):
                    slices = slice(iList,nSubFrames,nThreads)
                    SphericalRingLists.append(SphericalRings[slices,:,:,:])
                    GridCounterLists.append(GridCounters[slices,:,:])
                    RespondImgLists.append(RespondImgs[slices,:,:])
            
    #        BatchProjection(rawDataLists[0], 0, flags4MultiProc)
            for iThread in range(nThreads):
                if iOption == 1:
                    t = Process(target = BatchProjection, args=(strSequence, FrameLists[iThread], iProc,  iThread, flags4MultiProc))
                if iOption == 2:
                    t = Process(target = BatchGetKeyPts, args=(strSequence, FrameLists[iThread], SphericalRingLists[iThread], GridCounterLists[iThread], 
                                                              RespondImgLists[iThread], iProc, iThread, flags4MultiProc))
                    # BatchGetKeyPts(strSequence, FrameLists[iThread], SphericalRingLists[iThread], GridCounterLists[iThread], 
                    #                                             RespondImgLists[iThread], iProc, iThread, flags4MultiProc)
                if iOption == 3:
                    t = Process(target = BatchCorrectPC, args=(strSequence, FrameLists[iThread], iProc,  iThread, flags4MultiProc))
                t.start()
    
            while sum(flags4MultiProc)<nThreads:
                sleep(1)
    
            del flags4MultiProc


if __name__ == "__main__":
    freeze_support()
    
    # for cupy multi-thread processing
    import multiprocessing as mp
    mp.set_start_method('spawn')
    
    # BatchPorcess(2)
       
    
    #----------------(for test at first) Visualization of Voxelmodel---------------------------------------------
        
    PC = np.fromfile(strDataBaseDir + '/00/velodyne/000500.bin', dtype=np.float32, count=-1)
    PC = PC.reshape([-1,4])
    t0=time()
    
    SphericalRing, GridCounter = ProjectPC2SphericalRing(PC)
    RangeImage = ProjectPC2RangeImage(PC[:,0:3])
    t1=time()
    print(round(t1-t0, 2), 's')
    
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    import keras
    from keras.models import load_model
    RespondLayer = load_model('SphericalRingPCRespondLayer.h5')
    
    
    # prediction
    SphericalRing_ = SphericalRing[0:nLines, 0:ImgW-CropWidth_SphericalRing, Channels4AE]
    SphericalRing_ = SphericalRing_.reshape(1, SphericalRing_.shape[0], SphericalRing_.shape[1], SphericalRing_.shape[2])
    RespondImg = RespondLayer.predict(SphericalRing_)
    RespondImg = RespondImg.reshape(RespondImg.shape[1],RespondImg.shape[2],RespondImg.shape[3])
    t2=time()
    print(round(t2-t1, 2), 's')
    
        
    KeyPts, KeyPixels, PlanarPts = GetKeyPtsByAE(SphericalRing, GridCounter, RespondImg)
    RangeImage[KeyPixels[:,0],KeyPixels[:,1]] =  np.max(RangeImage)
    
    ExtendedKeyPts = ExtendKeyPtsInShpericalRing(SphericalRing, GridCounter, KeyPixels)
    t3=time()
    print(round(t3-t2, 2), 's')
    
    
    print('cntKeyPts =', KeyPts.shape[0])
    print('cntExtendedKeyPts =', ExtendedKeyPts.shape[0])
    print('cntPlanarPts =', PlanarPts.shape[0])
    
    
    
    SingleColor0=np.ones((PC.shape[0],1), dtype=np.float32)*0.0
    #SingleColor1=np.ones((PC_.shape[0],1), dtype=np.float32)*0.0
    KeyColors=np.ones((KeyPts.shape[0],1), dtype=np.float32)*1.0
    ExtendedKeyColors=np.ones((ExtendedKeyPts.shape[0],1), dtype=np.float32)*0.8
    PlanarColors=np.ones((PlanarPts.shape[0],1), dtype=np.float32)*0.3
    
    
    
    fig = mlab.figure(bgcolor=(0, 0, 0), size=(1640, 1500))
    node_VoxelPC=mlab.points3d(PC[:,0], PC[:,1], PC[:,2],
                         mode="point", figure=fig)
    node_VoxelPC.glyph.scale_mode = 'scale_by_vector'
    node_VoxelPC.mlab_source.dataset.point_data.scalars = SingleColor0
    
    
    #node_VoxelPC=mlab.points3d(PC_[:,0], PC_[:,1], PC_[:,2],
    #                     mode="point", figure=fig)
    #node_VoxelPC.glyph.scale_mode = 'scale_by_vector'
    #node_VoxelPC.mlab_source.dataset.point_data.scalars = colors_
    
    #mlab.title('ScoreMap')
    mlab.axes(xlabel='X', ylabel='Y', zlabel='Z')
    
    
    PtSize = 0.2
    node = mlab.points3d(KeyPts[:,0], KeyPts[:,1], KeyPts[:,2], scale_factor=PtSize, figure=fig)
    node.glyph.scale_mode = 'scale_by_vector'
    node.mlab_source.dataset.point_data.scalars = KeyColors
    
    # PtSize = 0.1
    # node = mlab.points3d(ExtendedKeyPts[:,0], ExtendedKeyPts[:,1], ExtendedKeyPts[:,2], scale_factor=PtSize, figure=fig)
    # node.glyph.scale_mode = 'scale_by_vector'
    # node.mlab_source.dataset.point_data.scalars = ExtendedKeyColors
    
    #PtSize = 0.1
    #node = mlab.points3d(PlanarPts[:,0], PlanarPts[:,1], PlanarPts[:,2], scale_factor=PtSize, figure=fig)
    #node.glyph.scale_mode = 'scale_by_vector'
    #node.mlab_source.dataset.point_data.scalars = PlanarColors
    #
    #mlab.quiver3d(PlanarPts[:,0], PlanarPts[:,1], PlanarPts[:,2], \
    #                     PlanarPts[:,3], PlanarPts[:,4], PlanarPts[:,5], \
    #                     figure=fig, line_width=0.5, scale_factor=1)
    
    fig = mlab.figure(bgcolor=(0, 0, 0), size=(1640, 500))
    mlab.imshow(RangeImage)    
    mlab.view(270, 0, 1800, [0,0,0])
    
    
    
    mlab.show()
    
    
     

















