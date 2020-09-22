#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 11:02:26 2019

@author: rain
"""

import numpy as np
import mayavi.mlab as mlab
from scipy import io
import math

from Transformations import *
from Dirs import *
 
    

def FusePCsFromFrames(poses, Tr, frames):
    nFrames = frames.shape[0]
    if nFrames > 1:
        nFrameStep = frames[1] - frames[0]
    else:
        nFrameStep = 1
        
    if iDataSource >= 2:
        R90 = EulerAngle2RotateMat(-math.pi/2,0,-math.pi/2,'xyz')
    
    colors = np.linspace(0, 1, num=nFrames, dtype=np.float32)
    PCs = np.array([0,0,0],dtype=np.float32).reshape(1,3)
    AllKeyPts = np.array([0,0,0],dtype=np.float32).reshape(1,3)
    color4PCs = np.array([0],dtype=np.float32).reshape(1,1)
    color4KeyPts = np.array([0],dtype=np.float32).reshape(1,1)
    cntFrame = 0
    for iFrame in frames:
        print(iFrame)
        
        PC = np.fromfile(str(DataDir+str(iFrame).zfill(6)+".bin"), dtype=np.float32, count=-1).reshape([-1,4])
        
        pose = np.array([poses[iFrame,[0,1,2,3]],poses[iFrame,[4,5,6,7]],poses[iFrame,[8,9,10,11]]],dtype=np.float32)
        
        PC_ = TranslatePtsIntoWorldFrame(pose, Tr, PC[:,0:3])
        PCs = np.r_[PCs,PC_]
        
            
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
        elif iDataSource == 3:
            fileName = str(strIssKeyPtsDir + strSequence + '/' + str(iFrame).zfill(6)+'.bin')
            keyPtsData = np.fromfile(fileName, dtype=np.float32, count=-1).reshape([-1,3])
            KeyPts = keyPtsData[:,0:3]
        elif iDataSource == 4:
            fileName = str(strHarrisKeyPtsDir + strSequence + '/' + str(iFrame).zfill(6)+'.bin')
            keyPtsData = np.fromfile(fileName, dtype=np.float32, count=-1).reshape([-1,3])
            KeyPts = keyPtsData[:,0:3]
        elif iDataSource == 5:
            fileName = str(strSiftKeyPtsDir + strSequence + '/' + str(iFrame).zfill(6)+'.bin')
            keyPtsData = np.fromfile(fileName, dtype=np.float32, count=-1).reshape([-1,3])
            KeyPts = keyPtsData[:,0:3]
            
        
        if iDataSource >= 2:
            KeyPts = np.dot(R90, KeyPts.T).T
            
            
        KeyPts_ = TranslatePtsIntoWorldFrame(pose, Tr, KeyPts)
            
        AllKeyPts = np.r_[AllKeyPts, KeyPts_]
        
        color = colors[cntFrame]*np.ones([PC.shape[0],1],dtype=np.float32)
        color4PCs = np.r_[color4PCs,color]
        color = colors[cntFrame]*np.ones([KeyPts.shape[0],1],dtype=np.float32)
        color4KeyPts = np.r_[color4KeyPts,color]
        
        cntFrame += 1

#    PCs = np.delete(PCs,[0],axis=0)
    AllKeyPts = np.delete(AllKeyPts,[0],axis=0)
    color4PCs = np.delete(color4PCs,[0],axis=0)
    color4KeyPts = np.delete(color4KeyPts,[0],axis=0)

    return PCs, AllKeyPts, color4PCs, color4KeyPts



iDataSource = 1  #0, our data; 1, 3DFeatNet; 2, USIP; 3, ISS; 4, Harris; 5; SIFT

iGroundTruth = 0

iFromDebugInfo = 0

iFrameStart = 400
iFrameStop = 550
iFrameStep = 10
 
strSequence = '01'
if iGroundTruth > 0:
    poses = np.loadtxt(strGroundTruthPosesDir+strSequence+'.txt')
#poses_ = np.loadtxt(strEstimatedPosesDir+strSequence+'.txt')
#poses_ = np.loadtxt('/media/rain/Win10_F/KITTI_odometry/poses_/'+strSequence+'.txt')
poses_ = np.loadtxt('/media/rain/Win10_F/KITTI_odometry/poses___/'+strSequence+'.txt')
calib = np.loadtxt(strCalibDataDir+strSequence+'/calib_.txt')
Tr = np.array(calib[4,:].reshape(3,4),dtype=np.float32)

if iGroundTruth > 0:
    trajectory=poses[:,[3,7,11]]

DataDir = strDataBaseDir+strSequence+'/velodyne/'
KeyPtsDir = str(strDataBaseDir+strSequence+'/KeyPts/')
FeaturesDataDir = str(strDataBaseDir+strSequence+'/Features/')

if iFromDebugInfo <= 0:
    frames = np.arange(iFrameStart, iFrameStop, iFrameStep)
else:    
    RefinementDataDir = str(strDataBaseDir + strSequence + '/RefinenmentData/')
    mat = io.loadmat(RefinementDataDir+'DebugInfo.mat')
    aDebugInfo = mat['aDebugInfo']
    frames = np.array(aDebugInfo[:,0], dtype=np.int32)
    frames = frames[30 : 60]
#    frames = frames

if iGroundTruth > 0:
    PCs, AllKeyPts, color4PCs, color4KeyPts = FusePCsFromFrames(poses, Tr, frames)
PCs_, AllKeyPts_, color4PCs_, color4KeyPts_ = FusePCsFromFrames(poses_, Tr, frames)


if iGroundTruth > 0:
    poses__ = poses
else:
    poses__ = poses_
R_Start = np.array([poses__[iFrameStart,[0,1,2]],poses__[iFrameStart,[4,5,6]],poses__[iFrameStart,[8,9,10]]],dtype=np.float32)
T_Start = np.array([poses__[iFrameStart,3],poses__[iFrameStart,7],poses__[iFrameStart,11]],dtype=np.float32)
shiftT = np.dot(R_Start,Tr[:,3].reshape(3,1))+T_Start.reshape(3,1)
print(shiftT)


PtSize = 0.2


if iGroundTruth > 0:
    fig = mlab.figure(bgcolor=(0, 0, 0), size=(1640, 1500))
    nodeFusedPC = mlab.points3d(PCs[:,0], PCs[:,1], PCs[:,2], mode="point",  figure=fig)
    nodeFusedPC.glyph.scale_mode = 'scale_by_vector'
    nodeFusedPC.mlab_source.dataset.point_data.scalars = color4PCs
    mlab.title('Fused PC - Ground Truth')


fig = mlab.figure(bgcolor=(0, 0, 0), size=(1640, 1500))
nodeFusedPC = mlab.points3d(PCs_[:,0], PCs_[:,1], PCs_[:,2], mode="point",  figure=fig)
nodeFusedPC.glyph.scale_mode = 'scale_by_vector'
nodeFusedPC.mlab_source.dataset.point_data.scalars = color4PCs_

node = mlab.points3d(AllKeyPts_[:,0], AllKeyPts_[:,1], AllKeyPts_[:,2], scale_factor=PtSize, figure=fig)
node.glyph.scale_mode = 'scale_by_vector'
node.mlab_source.dataset.point_data.scalars = color4KeyPts_

mlab.title('Fused PC - Estimated')

#mlab.axes(x_axis_visibility = True)

mlab.view(270, 90, 400, [0,0,0])

mlab.show()



















