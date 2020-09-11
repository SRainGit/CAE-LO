#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 15:35:48 2019

@author: rain
"""


import os
import math
import numpy as np
from numpy import linalg as LA
import mayavi
import mayavi.mlab as mlab
from scipy import io
import random
from multiprocessing import Pool
from threading import Thread
from multiprocessing import Process
from time import time
import copy


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
from Voxel import VoxelSize,VisibleLength,VisibleWidth,VisibleHeight,BlockRealSize,BlockSize,nBlocksL,nBlocksW,nBlocksH
from Voxel import PatchSize,PatchRadius,Scales,ScaleRatios,CropBlocks

from SphericalRing import *



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


def BatchInputData(fileList, nRandDataPerFile):
    SphericalRings = np.zeros((len(fileList), nLines, ImgW-CropWidth_SphericalRing, len(Channels4AE)), dtype=np.float32)
    
    cntFile = 0
    for file in fileList:
        # load data
        mat = io.loadmat(file)
        SphericalRing = mat['SphericalRing']
        
        SphericalRings[cntFile,:,:,:] = SphericalRing[0:nLines, 0:ImgW-CropWidth_SphericalRing, Channels4AE]
        cntFile += 1
    
    return SphericalRings
    

#------------- load data by model.fit_generator -----------------------------------
def YieldBatchData(ModelFileList, nBatchFiles, nRandDataPerFile):
    iFile=0
    while True:
        if iFile+nBatchFiles>=len(ModelFileList):
            iFile=0
            continue
        SphericalRings=BatchInputData(ModelFileList[iFile:(iFile+nBatchFiles)], nRandDataPerFile)
        iFile=iFile+nBatchFiles
        yield(SphericalRings, SphericalRings)
            
        
#----------make file list----------------------------------------------------------------
trainRatio = 0.9
fileList = GetFileList(strDataBaseDir)
matFileList = [oneFile for oneFile in fileList if oneFile.split("/")[-2]=='SphericalRing']

# get training list and testing list
random.shuffle(matFileList)
#matFileList = matFileList[0:4000]
nTrain = int(trainRatio*len(matFileList))
trainingFileList = matFileList[0:nTrain]
validationFileList = matFileList[nTrain:len(matFileList)]


#TrainingBlocks = BatchInputData(trainingFileList[0:3], 192)


#-------------Autoendocer---------------------------------------------------------------------
bTrain = 0
KS0 = 1
KS1 = 3
KS2 = 5
epochs = 10
nGPUs = 2
nBatchFiles = nGPUs*16
nRandDataPerFile = 64

ACTIVATION = 'linear'
ACTIVATION1 = 'linear'
ACTIVATION2 = 'relu'


# feed
if bTrain == 1:
    # Convolutional autoencoder
    x = Input(shape=(nLines, ImgW-CropWidth_SphericalRing, len(Channels4AE)))
    
    # Encoder
    conv1_1 = Conv2D(filters=32, kernel_size=(KS1, KS1), strides=1, activation=ACTIVATION2, use_bias=True, padding='same')(x)
    conv1_1_2 = Conv2D(filters=8, kernel_size=(KS0, KS0), strides=1, activation=ACTIVATION2, use_bias=True, padding='same')(conv1_1)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1_1_2)
    conv1_2 = Conv2D(filters=16, kernel_size=(KS1, KS1), strides=1, activation=ACTIVATION2, use_bias=True, padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1_2)
    
    conv2_2 = Conv2D(filters=16, kernel_size=(KS1, KS1), strides=1, activation=ACTIVATION2, use_bias=True, padding='same')(pool2)
    up2 = UpSampling2D(size=(2, 2))(conv2_2)
    conv2_3 = Conv2D(filters=8, kernel_size=(KS1, KS1), strides=1, activation=ACTIVATION2, use_bias=True, padding='same')(up2)
    up3 = UpSampling2D(size=(2, 2))(conv2_3)
    r = Conv2D(filters=len(Channels4AE), kernel_size=(KS0, KS0), strides=1, activation=ACTIVATION1, use_bias=True, padding='same')(up3)
    
    autoencoder = Model(inputs=x, outputs=r)
    RespondLayer = Model(x, conv1_1_2)
    RespondLayer2 = Model(x, conv1_2)
#    RespondLayer3 = Model(x, conv1_3)
#    RespondLayer4 = Model(x, conv1_4)
    
    
#    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    parallel_model = multi_gpu_model(autoencoder, gpus=nGPUs)
    parallel_model.compile(optimizer='Adam', loss='mean_squared_error')
    
    autoencoder.summary()
    from keras.utils import plot_model
    plot_model(autoencoder, show_shapes=1, to_file='AE4SpericalRingPC.png')
    #
    #SVG(model_to_dot(autoencoder).create(prog='dot', format='svg'))
    
    weights_r = autoencoder.layers[4].get_weights()

    history = parallel_model.fit_generator(YieldBatchData(trainingFileList, nBatchFiles, nRandDataPerFile), 
#    history = autoencoder.fit_generator(YieldBatchData(trainingFileList, nBatchFiles, nRandDataPerFile), 
                                           steps_per_epoch = len(trainingFileList)/nBatchFiles,
                                           epochs = epochs, 
                                           max_queue_size = 10,
                                           validation_data = YieldBatchData(validationFileList, nBatchFiles, nRandDataPerFile),
                                           validation_steps = len(validationFileList)/nBatchFiles,
                                           workers = 2, 
                                           use_multiprocessing = True,
                                           shuffle = True)
    
    # save model
    autoencoder.save('AE4SphericalRingPC.h5')
    RespondLayer.save('SphericalRingPCRespondLayer.h5')
    RespondLayer2.save('SphericalRingPCRespondLayer2.h5')
#    RespondLayer3.save('SphericalRingPCRespondLayer3.h5')
#    RespondLayer4.save('SphericalRingPCRespondLayer4.h5')
else:
    autoencoder = load_model('AE4SphericalRingPC.h5')
    RespondLayer = load_model('SphericalRingPCRespondLayer.h5')
    # RespondLayer2 = load_model('SphericalRingPCRespondLayer2.h5')
#    RespondLayer3 = load_model('SphericalRingPCRespondLayer3.h5')
#    RespondLayer4 = load_model('SphericalRingPCRespondLayer4.h5')


def GetKeyPixelsAndKeyPts(EncodedModel, SphericalImage):
    WindowSize = 5
    WindowRadius = int(WindowSize/2)
    
    NormDiffThreshold = 10.0
    
    
    keyPts = []
    mask = np.ones((WindowSize,WindowSize), dtype=np.int32)
    mask[WindowRadius,WindowRadius] = 0
    for iPixel in range(AllPixelIndexes_WithoutWindowEdge.shape[0]):
        iX = AllPixelIndexes_WithoutWindowEdge[iPixel, 0]
        iY = AllPixelIndexes_WithoutWindowEdge[iPixel, 1]
        
        # extract all the windows
        oneWindow = EncodedModel[iX-WindowRadius:iX+WindowRadius+1, iY-WindowRadius:iY+WindowRadius+1, :]
        norms = oneWindow[mask>0, :]

        norm = EncodedModel[iX, iY, :]
        diffsNorm = norms - norm
        diffs = LA.norm(diffsNorm, axis=1)
        minDiff = min(diffs)
        if minDiff > NormDiffThreshold:
            pt = SphericalImage[iX, iY, :]
            if LA.norm(pt) > VisibleBottom:
                keyPts.append(pt)
        
    print('cntKeyPixels =', len(keyPts))
    keyPts = np.array(keyPts, dtype=np.float32)
    return 1, keyPts



iTestModel=0
VoxelPC=[]
testingModels = BatchInputData(validationFileList[0:10], 33)
while iTestModel <= 0:
    iTestModel = iTestModel+1
    testModel = testingModels[iTestModel,:,:,:]
    
    encodedModels = RespondLayer.predict(testingModels)
    encodedModel = encodedModels[iTestModel,:,:,:]
    RespondImage = LA.norm(encodedModel, axis=2)
    KeyPixelsImage, keyPts = GetKeyPixelsAndKeyPts(encodedModel, testModel)
    
    decodedModels = autoencoder.predict(testingModels)
    decodedModel = decodedModels[iTestModel,:,:,:]


    testModel_ = np.zeros((ImgH, ImgW, len(Channels4AE)), dtype=np.float32)
    testModel_[0:nLines, 0:ImgW-CropWidth_SphericalRing,:] = testModel
    testPC = ProjectImage2PC(testModel_)
    decodedModel_ = np.zeros((ImgH, ImgW, len(Channels4AE)), dtype=np.float32)
    decodedModel_[0:nLines, 0:ImgW-CropWidth_SphericalRing,:] = decodedModel
    decodedPC = ProjectImage2PC(decodedModel_)


    PtSize = 0.2
    Colors0=np.ones((testPC.shape[0],1), dtype=np.float32)*0.0
    keyColors0=np.ones((keyPts.shape[0],1), dtype=np.float32)*1.0

    fig = mlab.figure(bgcolor=(0, 0, 0), size=(1640, 500))
    node = mlab.points3d(testPC[:,0], testPC[:,1], testPC[:,2], mode="point", figure=fig)
    node.glyph.scale_mode = 'scale_by_vector'
    node.mlab_source.dataset.point_data.scalars = Colors0
    node = mlab.points3d(keyPts[:,0], keyPts[:,1], keyPts[:,2], scale_factor=PtSize, figure=fig)
    node.glyph.scale_mode = 'scale_by_vector'
    node.mlab_source.dataset.point_data.scalars = keyColors0
    
    fig = mlab.figure(bgcolor=(0, 0, 0), size=(1640, 500))
    mlab.imshow(RespondImage)    
    mlab.view(270, 0, 1800, [0,0,0])
    
    fig = mlab.figure(bgcolor=(0, 0, 0), size=(1640, 500))
    mlab.imshow(KeyPixelsImage)    
    mlab.view(270, 0, 1200, [0,0,0])
    
    fig = mlab.figure(bgcolor=(0, 0, 0), size=(1640, 1500))
    node = mayavi.mlab.points3d(decodedPC[:,0], decodedPC[:,1], decodedPC[:,2], mode="point", figure=fig)
    
    mlab.show()














