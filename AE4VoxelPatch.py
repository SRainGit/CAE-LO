#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 09:37:34 2019

@author: rain
"""

import os
import math
import numpy as np
import mayavi
import mayavi.mlab
from scipy import io
import random
from multiprocessing import Pool
from threading import Thread
from multiprocessing import Process
from time import time


os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
import keras
from keras.models import Model, load_model
from keras.utils import multi_gpu_model
from keras.layers import Input, add
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape
from keras import regularizers
from keras.regularizers import l2
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.convolutional import Conv3D, MaxPooling3D, UpSampling3D, ZeroPadding3D, Conv3DTranspose
from keras.utils import np_utils

from Dirs import *
from Voxel import *


def GetFileList(file_dir):
    '''
    Args:
        file_dir: file directory
    Returns:
        nClasses, list of images and labels
    '''    
    fileList=[]
    if not os.path.exists(file_dir):
        print('Wrong path!')
        return
        
    # get file list
    for root, dirs, files in os.walk(file_dir, topdown=False):
        for name in files:
            fileList.append(os.path.join(root, name))   
    assert len(fileList) > 0
    return fileList


def BatchInputData(fileList, nRandPatchesPerFile):
    assert Scales == 3  # this code only supports 3 scales currently
    assert nRandPatchesPerFile % Scales == 0  # one sameple is in 3 scales for 3 patches
    nPatchGroups = int(nRandPatchesPerFile/Scales)
    
    ReserveRatio = 10
    
    RandDataSource = 0  # 0, from AllVoxels0; 1, from KeyPts
    RandDataSource = 1
    
    # ------------------------------ big loop for extract patches from the input files
    VoxelEdge0X_ = CropBlocks*BlockSize
    VoxelEdge0X = (nBlocksL-CropBlocks)*BlockSize
    VoxelEdge0Y_ = CropBlocks*BlockSize
    VoxelEdge0Y = (nBlocksW-CropBlocks)*BlockSize
    VoxelEdge0Z_ = CropBlocks*BlockSize
    VoxelEdge0Z = (nBlocksH-CropBlocks)*BlockSize
    aVisibleRange = np.array([VisibleLength, VisibleWidth, VisibleHeight], dtype=np.float32).reshape(1,3)
    
    voxelSize0 = VoxelSizes[0]
    AllPatchesList = []
    for file in fileList:
        # load voxel data
        mat = io.loadmat(file)
        AllVoxels0 = mat['AllVoxels0']
        AllVoxels1 = mat['AllVoxels1']
        AllVoxels2 = mat['AllVoxels2']
        # load keyPts data
        if RandDataSource == 1:
            baseDir = os.path.dirname(os.path.dirname(file))
            KeyPtsFile = os.path.join(baseDir,'KeyPts',file.split("/")[-1])    
            mat = io.loadmat(KeyPtsFile)
            KeyPts = mat['KeyPts'] 
         
        # generate random indexes
        RandIdxes=np.random.random((nPatchGroups*ReserveRatio,))
        if RandDataSource == 0:
            RandIdxes=RandIdxes*(len(AllVoxels0)-1)+1
        elif RandDataSource == 1:
            RandIdxes=RandIdxes*(KeyPts.shape[0]-1)+1
        RandIdxes=np.array(RandIdxes,dtype=int)
        
        # extract patches using the random indexes
        cntValidPt = 0
        validPts = np.zeros((nPatchGroups,3), dtype=np.float32)
        for iSample in range(RandIdxes.shape[0]):
            # get corresponding random data
            ## 0, using rand voxels in AllVoxels0
            if RandDataSource == 0:
                voxel0 = AllVoxels0[RandIdxes[iSample],:]
            elif RandDataSource == 1:
                ## 1, using rand pts in KeyPts
                pts = KeyPts[RandIdxes[iSample],:]
                voxel0 = ((pts+aVisibleRange)/voxelSize0).reshape(3,)
            
            # not using the boundary blocks for simplify
            if voxel0[0] < VoxelEdge0X_ or voxel0[0] >= VoxelEdge0X or\
               voxel0[1] < VoxelEdge0Y_ or voxel0[1] >= VoxelEdge0Y or\
                voxel0[2] < VoxelEdge0Z_ or voxel0[2] >= VoxelEdge0Z:
                    continue            
            validPts[cntValidPt,:] = (voxel0*voxelSize0).reshape(1,3)- aVisibleRange
            
            
            
            cntValidPt += 1
            if cntValidPt == nPatchGroups:
                 break                
        
        assert cntValidPt == nPatchGroups
        
        validPts, PatchesList = GetPatchesList(validPts, AllVoxels0, AllVoxels1, AllVoxels2)
        AllPatchesList += PatchesList
            
    Patches = np.array(AllPatchesList, dtype=np.float64)
    Patches = Patches.reshape(len(fileList)*nRandPatchesPerFile, PatchSize, PatchSize, PatchSize, 1)
    return Patches
    

#------------- load data by model.fit_generator -----------------------------------
def YieldBatchData(ModelFileList, nBatchFiles, nRandPatchesPerFile):
    iFile=0
    while True:
        if iFile+nBatchFiles>=len(ModelFileList):
            iFile=0
            continue
        Patches=BatchInputData(ModelFileList[iFile:(iFile+nBatchFiles)], nRandPatchesPerFile)
        iFile=iFile+nBatchFiles
        yield(Patches,Patches)
            
        
#----------make file list----------------------------------------------------------------
trainRatio = 0.9
fileList = GetFileList(strDataBaseDir)
matFileList = [oneFile for oneFile in fileList if oneFile.split("/")[-2]=='VoxelModel']

# get training list and testing list
random.shuffle(matFileList)
nTrain = int(trainRatio*len(matFileList))
trainingFileList = matFileList[0:nTrain]
validationFileList = matFileList[nTrain:len(matFileList)]




#-------------Autoendocer---------------------------------------------------------------------
bTrain = 1
KS0 = 3
KS1 = 3
KS2 = 2
epochs = 10
nGPUs = 2
nBatchFiles = nGPUs*1
nRandPatchesPerFile = Scales*256

t0 = time()
Patches = BatchInputData(trainingFileList[0:8], nRandPatchesPerFile)
t1 = time()
print(round(t1-t0, 2))

ACTIVATION = 'linear'
ACTIVATION1 = 'sigmoid'
ACTIVATION2 = 'tanh'
ACTIVATION3 = 'relu'


# feed
if bTrain == 1:
    # Convolutional autoencoder
    x = Input(shape=(PatchSize, PatchSize, PatchSize, 1))
    
    # Encoder
    conv1_1 = Conv3D(filters=8, kernel_size=(KS0, KS0, KS0), strides=1, activation=ACTIVATION3, padding='same')(x)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2), strides=2, padding='same')(conv1_1)
    conv1_2 = Conv3D(filters=16, kernel_size=(KS1, KS1, KS1), strides=1, activation=ACTIVATION3, padding='same')(pool1)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2), strides=2, padding='same')(conv1_2)
    conv1_3 = Conv3D(filters=32, kernel_size=(KS1, KS1, KS1), strides=1, activation=ACTIVATION3, padding='same')(pool2)
    flatten = Flatten()(conv1_3)
    
    fn1 = Dense(200, activation=ACTIVATION3, use_bias=True)(flatten)
    fn2 = Dense(20, activation=ACTIVATION, use_bias=True)(fn1)
    fn3 = Dense(200, activation=ACTIVATION3, use_bias=True)(fn2)
    
#    # Decoder
    fn4 = Dense(2048, activation=ACTIVATION3, use_bias=True)(fn3)
    reshape = Reshape((4,4,4,32))(fn4)
    conv2_1 = Conv3D(filters=16, kernel_size=(KS1, KS1, KS1), strides=1, activation=ACTIVATION3, padding='same')(reshape)
    up2 = UpSampling3D(size=(2, 2, 2))(conv2_1)
    conv2_2 = Conv3D(filters=8, kernel_size=(KS1, KS1, KS1), strides=1, activation=ACTIVATION3, padding='same')(up2)
    up3 = UpSampling3D(size=(2, 2, 2))(conv2_2)
    r = Conv3D(filters=1, kernel_size=(KS0, KS0, KS0), strides=1, activation=ACTIVATION1, padding='same')(up3)
    
    autoencoder = Model(inputs=x, outputs=r)
    encoder = Model(x, fn2)
    
    parallel_model = multi_gpu_model(autoencoder, gpus=nGPUs)
    parallel_model.compile(optimizer='adadelta', loss='binary_crossentropy')
    
    autoencoder.summary()
    from keras.utils import plot_model
    plot_model(autoencoder, show_shapes=1, to_file='Autoencoder4VoxelPatch.png')
    
    # SVG(model_to_dot(autoencoder).create(prog='dot', format='svg'))
    
    weights_r = autoencoder.layers[4].get_weights()

    history = parallel_model.fit_generator(YieldBatchData(trainingFileList, nBatchFiles, nRandPatchesPerFile), 
#    history = autoencoder.fit_generator(YieldBatchData(trainingFileList, nBatchFiles, nRandPatchesPerFile), 
                                           steps_per_epoch = len(trainingFileList)/nBatchFiles,
                                           epochs = epochs, 
                                           max_queue_size = 50,
                                           validation_data = YieldBatchData(validationFileList, nBatchFiles, nRandPatchesPerFile),
                                           validation_steps = len(validationFileList)/nBatchFiles,
                                           workers = 4, 
                                           use_multiprocessing = True,
                                           shuffle = True)
    
    # save model
    autoencoder.save('./TrainedModels/AutoencoderModel4VoxelPatch.h5')
    encoder.save(strVoxelPatchEncoderPath)
else:
    autoencoder = load_model('./TrainedModels/AutoencoderModel4VoxelPatch.h5')
    encoder = load_model(strVoxelPatchEncoderPath)


iTestModel=0
VoxelPC=[]
testingModels = BatchInputData(validationFileList[0:3], 33)
while iTestModel <= 10:
    iTestModel = iTestModel+1
    VoxelModel = testingModels[iTestModel,:,:,:,:]
    VoxelModel = VoxelModel.reshape(PatchSize, PatchSize, PatchSize)
    VoxelPC = VoxelModel2PC(VoxelModel)
    
    encodeModels = encoder.predict(testingModels)
    codes = encodeModels[iTestModel,:]
#    encodedPC,colorsOfEncodedPC = VoxelModel2ColofulPC(encodedModel)
#    print(encodedModel.flatten())
    
    decodedModels = autoencoder.predict(testingModels)
    decodedModels = decodedModels.reshape(decodedModels.shape[0],decodedModels.shape[1], decodedModels.shape[2],decodedModels.shape[3])
    decodedModel = decodedModels[iTestModel,:,:,:]
    decodedVoxelPC = VoxelModel2PC(decodedModel)    

    offset2Show = 2*PatchSize*VoxelSize
    # show VoxelPC
    fig_testModel = mayavi.mlab.figure(bgcolor=(0, 0, 0), size=(1240, 800))
    mayavi.mlab.points3d(VoxelPC[:,0], VoxelPC[:,1], VoxelPC[:,2],
                         VoxelPC[:,2],          # Values used for Color
                         mode="point",
                         colormap='spectral', # 'bone', 'copper', 'gnuplot'
                         figure=fig_testModel,
                         )
    
    #fig_encodedMode = mayavi.mlab.figure(bgcolor=(0, 0, 0), size=(640, 500))
    #nodeEncondedPC=mayavi.mlab.points3d(encodedPC[:,0], encodedPC[:,1], encodedPC[:,2],
    #                     mode="point",
    #                     figure=fig_encodedMode,
    #                     )
    #nodeEncondedPC.mlab_source.dataset.point_data.scalars = colorsOfEncodedPC
    
    #fig_decodedModel = mayavi.mlab.figure(bgcolor=(0, 0, 0), size=(640, 500))
    mayavi.mlab.points3d(decodedVoxelPC[:,0], decodedVoxelPC[:,1], decodedVoxelPC[:,2]+offset2Show,
                         decodedVoxelPC[:,2],          # Values used for Color
                         mode="point",
                         figure=fig_testModel,
                         )
    mayavi.mlab.show()















