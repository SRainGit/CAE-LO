#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 10:09:33 2019

@author: rain
"""
import os
from scipy import io
import numpy as np
import mayavi.mlab

from Voxel import VoxelSize, VisibleLength, VisibleWidth, VisibleHeight, ModelLength, ModelWidth, ModelHeight
from Voxel import VoxelModel2PC, VoxelModel2ColofulPC
from Match import GetFeatureColorPC

from sklearn.cluster import KMeans
#import pickle
from sklearn.externals import joblib


import keras
from keras.models import Model, load_model
PatchEncoder = load_model('EncoderModel4VoxelPatch.h5')

bTrain = 0
#bTrain = 1

FusedPC = np.zeros((3,1),  dtype=np.float32)
Codes = np.zeros((1,20),  dtype=np.float32)
Weights = np.array([0],  dtype=np.float32).reshape(1,1)
    
#listSequence = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
listSequence = [0]
for iSequence in listSequence:
#for iSequence in range(22):
    strSequence=str(iSequence).zfill(2)   
    
    voxelDataDir=str('/media/rain/Win10_F/KITTI_odometry/data_odometry_velodyne/dataset/sequences/'+strSequence+'/VoxelModel/')
    if bTrain == 0:
        poses=np.loadtxt('/media/rain/Win10_F/KITTI_odometry/data_odometry_poses/dataset/poses/'+strSequence+'.txt')
        calib=np.loadtxt('/media/rain/Win10_F/KITTI_odometry/data_odometry_calib/dataset/sequences/'+strSequence+'/calib_.txt')
        Tr=np.array(calib[4,:].reshape(3,4),dtype=np.float32)
    
    
    fileList=os.listdir(voxelDataDir)
    nFrames=len(fileList)
    
    frameStep = 2000
    
    for iFrame in range(0, nFrames, frameStep):
        print(str(strSequence + ':' + str(nFrames) + ':' + str(iFrame)))
        file0=str(voxelDataDir+str(iFrame).zfill(6)+'.bin.mat')
        mat0 = io.loadmat(file0)
        VoxelModel0 = mat0['VoxelModel']
        PC0, Codes0, Weights0 = GetFeatureColorPC(VoxelModel0, PatchEncoder)
        Codes = np.r_[Codes, Codes0]
        Weights = np.r_[Weights, Weights0]
        
        if bTrain == 0:
            pose=np.array([poses[iFrame,[0,1,2,3]],poses[iFrame,[4,5,6,7]],poses[iFrame,[8,9,10,11]]],dtype=np.float32)
            onesVector=np.ones([PC0.shape[0],1],dtype=np.float32)
            PC_=np.c_[PC0[:,[0,1,2]], onesVector]
            PC_=np.dot(pose,np.r_[np.dot(Tr,PC_.T),onesVector.T])
            FusedPC=np.c_[FusedPC,PC_]


Codes = np.delete(Codes, 0, axis=0)
Weights = np.delete(Weights, 0, axis=0)
Weights = Weights.reshape(Weights.shape[0],)


if bTrain == 1:
    FeatureClusterModel = KMeans(n_clusters = 2)
    FeatureClusterModel.fit(X=Codes, sample_weight=Weights)
    
    joblib.dump(FeatureClusterModel, 'FeatureClusterModel.joblib')    
else:
    FeatureClusterModel = joblib.load('FeatureClusterModel.joblib')
    
    FusedPC=FusedPC.T
    print('nPts =', FusedPC.shape[0])
    FusedPC = np.delete(FusedPC, 0, axis=0)


predictedLabel = FeatureClusterModel.predict(Codes)
predictedLabel = predictedLabel / np.max(predictedLabel)

clusterSpace = FeatureClusterModel.transform(Codes)
#center1 = FeatureClusterModel.cluster_centers_[1,:]
#dist2Center = Codes - center1
#scores_ = np.linalg.norm((Codes - center1), axis=1)

scores0 = - clusterSpace[:,0]
scores0 = scores0 - np.min(scores0)
scores0 = scores0 / np.max(scores0)

scores1 = - clusterSpace[:,1]
scores1 = scores1 - np.min(scores1)
scores1 = scores1 / np.max(scores1)

#scores2 = - clusterSpace[:,2]
#scores2 = scores2 - np.min(scores2)
#scores2 = scores2 / np.max(scores2)

scores1_ = clusterSpace[:,1] / clusterSpace[:,0]
scores1_ = scores1_ / np.max(scores1_)

if bTrain == 0:
    Weights = Weights/np.max(Weights)
    fig = mayavi.mlab.figure(bgcolor=(0, 0, 0), size=(1200, 800))
    nodeCodePC = mayavi.mlab.points3d(FusedPC[:,0], FusedPC[:,1], FusedPC[:,2],
                         mode="point", figure=fig)
    nodeCodePC.glyph.scale_mode = 'scale_by_vector'
    #nodeCodePC.mlab_source.dataset.point_data.scalars = Weights
#    nodeCodePC.mlab_source.dataset.point_data.scalars = predictedLabel
    nodeCodePC.mlab_source.dataset.point_data.scalars = scores1
    
    mayavi.mlab.title('Score Norm Map')
    mayavi.mlab.axes(xlabel='x', ylabel='y', zlabel='z')
    #mayavi.mlab.outline(nodeCodePC)
    mayavi.mlab.show()



