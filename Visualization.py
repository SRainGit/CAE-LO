#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 10:27:44 2019

@author: rain
"""

import numpy as np
import mayavi.mlab as mlab
import matplotlib.pyplot as plt
from numpy import linalg as LA
from numpy import dot as dot

from Transformations import *


def ShowTrajactory(fig, poses, color, ptSize):
    Rs, Ts = GetRsAndTsFromPoses(poses)
    
    AxisLength = 0.8
    Rs_ = Rs*AxisLength;
    
    colors = np.ones((Ts.shape[0],),dtype=np.float32)*color
    if ptSize == 0:
        node = mlab.points3d(Ts[:,0], Ts[:,1], Ts[:,2], mode="point", figure=fig)
    else:        
        node = mlab.points3d(Ts[:,0], Ts[:,1], Ts[:,2], scale_factor=ptSize, figure=fig)
    node.glyph.scale_mode = 'scale_by_vector'
    node.mlab_source.dataset.point_data.scalars = colors
    
    for iAxis in range(3):
        node = mlab.quiver3d(Ts[:,0], Ts[:,1], Ts[:,2], \
                             Rs_[:,iAxis,0], Rs_[:,iAxis,1], Rs_[:,iAxis,2], \
                             figure=fig, line_width=2.0, scale_factor=1, mode='2ddash')
    
    
def CompareTrajactory(fig, trajectory0, trajectory1, nStartFrame, nEndFrame):
    if nEndFrame <= 0:
        nEndFrame = trajectory0.shape[0]
    node = mlab.quiver3d(trajectory0[nStartFrame:nEndFrame,0], trajectory0[nStartFrame:nEndFrame,1], trajectory0[nStartFrame:nEndFrame,2], \
                         trajectory1[nStartFrame:nEndFrame,0]-trajectory0[nStartFrame:nEndFrame,0], \
                         trajectory1[nStartFrame:nEndFrame,1]-trajectory0[nStartFrame:nEndFrame,1], \
                         trajectory1[nStartFrame:nEndFrame,2]-trajectory0[nStartFrame:nEndFrame,2], \
                         figure=fig, line_width=0.5, scale_factor=1, mode='2ddash')
#    mlab.axes(x_axis_visibility = True)
#    mlab.axes(y_axis_visibility = True)
#    mlab.axes(z_axis_visibility = True)

    
    
def ShowMatchingResult(RawDataDir, iFrame0, iFrame1, KeyPts0, KeyPts1, PlanarPts0, PlanarPts1,
                       bShowPairs, inliersIdx0, inliersIdx1, oriRelR, oriRelT, deltaR, deltaT):
    rawFile0 = RawDataDir + str(iFrame0).zfill(6)+'.bin'
    rawFile1 = RawDataDir + str(iFrame1).zfill(6)+'.bin'
    PC0 = np.fromfile(rawFile0, dtype=np.float32, count=-1).reshape([-1,4])[:,0:3]
    PC1 = np.fromfile(rawFile1, dtype=np.float32, count=-1).reshape([-1,4])[:,0:3]
    pairs0 = KeyPts0[inliersIdx0,:]
    pairs1 = KeyPts1[inliersIdx1,:]
                
    PC1_ = (np.dot(oriRelR, PC1.T) + oriRelT.reshape(3,1)).T
    KeyPts1_ = (np.dot(oriRelR, KeyPts1.T) + oriRelT.reshape(3,1)).T
    pairs1_ = (np.dot(oriRelR, pairs1.T) + oriRelT.reshape(3,1)).T
    PlanarPts1_ = (np.dot(oriRelR, PlanarPts1.T) + oriRelT.reshape(3,1)).T
    
    PC1__ = (np.dot(deltaR, PC1_.T) + deltaT.reshape(3,1)).T
    FusedPC__ = np.r_[PC0, PC1__]
    KeyPts1__ = (np.dot(deltaR, KeyPts1_.T) + deltaT.reshape(3,1)).T
    pairs1__ = (np.dot(deltaR, pairs1_.T) + deltaT.reshape(3,1)).T
    PlanarPts1__ = (np.dot(deltaR, PlanarPts1_.T) + deltaT.reshape(3,1)).T
    
    
    
    Colors0=np.ones((PC0.shape[0],1), dtype=np.float32)*1.0
    KeyColors0=np.ones((KeyPts0.shape[0],1), dtype=np.float32)*0.9
    KeyColors0_=np.ones((KeyPts0.shape[0],1), dtype=np.float32)*0.9
    PlanarColor0=np.ones((PlanarPts0.shape[0],1), dtype=np.float32)*0.7
    
    Colors1=np.ones((PC1.shape[0],1), dtype=np.float32)*0.0
    KeyColors1=np.ones((KeyPts1.shape[0],1), dtype=np.float32)*0.1
    KeyColors1_=np.ones((KeyPts1.shape[0],1), dtype=np.float32)*0.1
    PlanarColor1=np.ones((PlanarPts1.shape[0],1), dtype=np.float32)*0.3
    
    
    Colors4FusedPC=np.r_[Colors0, Colors1]
    Colors4FusedPC=Colors4FusedPC.reshape(Colors4FusedPC.shape[0],)

    shift4Show = 0
    fig = mlab.figure(bgcolor=(0, 0, 0), size=(1500, 900))
    PtSize  = 0.1   
    
    node = mlab.points3d(PC0[:,0], PC0[:,1], PC0[:,2], mode="point", figure=fig)
    node.glyph.scale_mode = 'scale_by_vector'
    node.mlab_source.dataset.point_data.scalars = Colors0
    
    node = mlab.points3d(PC1_[:,0], PC1_[:,1], PC1_[:,2]+shift4Show, mode="point", figure=fig)
    node.glyph.scale_mode = 'scale_by_vector'
    node.mlab_source.dataset.point_data.scalars = Colors1    
    
    node = mlab.points3d(KeyPts0[:,0], KeyPts0[:,1], KeyPts0[:,2], scale_factor=PtSize, figure=fig)
    node.glyph.scale_mode = 'scale_by_vector'
    node.mlab_source.dataset.point_data.scalars = KeyColors0    
    
    node = mlab.points3d(KeyPts1_[:,0], KeyPts1_[:,1], KeyPts1_[:,2]+shift4Show, scale_factor=PtSize, figure=fig)
    node.glyph.scale_mode = 'scale_by_vector'
    node.mlab_source.dataset.point_data.scalars = KeyColors1    
    
    PtSize  = 0.02
    node = mlab.points3d(PlanarPts0[:,0], PlanarPts0[:,1], PlanarPts0[:,2], scale_factor=PtSize, figure=fig)
    node.glyph.scale_mode = 'scale_by_vector'
    node.mlab_source.dataset.point_data.scalars = PlanarColor0    
    
    node = mlab.points3d(PlanarPts1_[:,0], PlanarPts1_[:,1], PlanarPts1_[:,2]+shift4Show, scale_factor=PtSize, figure=fig)
    node.glyph.scale_mode = 'scale_by_vector'
    node.mlab_source.dataset.point_data.scalars = PlanarColor1
    
    if bShowPairs == True:
        mlab.quiver3d(pairs1_[:,0], pairs1_[:,1], pairs1_[:,2]+shift4Show, \
                             pairs0[:,0]-pairs1_[:,0], pairs0[:,1]-pairs1_[:,1], pairs0[:,2]-pairs1_[:,2]-shift4Show, \
                             figure=fig, line_width=0.5, scale_factor=1)
    
    mlab.title('Initial Matching')
    mlab.axes(xlabel='X', ylabel='Y', zlabel='Z')
    
    
    
    PtSize = 0.2
    fig = mlab.figure(bgcolor=(0, 0, 0), size=(1200, 800))
    mlab.points3d(FusedPC__[:,0], FusedPC__[:,1], FusedPC__[:,2],
                         Colors4FusedPC, mode="point", figure=fig)     
    if bShowPairs == True:
        mlab.quiver3d(pairs1__[:,0], pairs1__[:,1], pairs1__[:,2], \
                             pairs0[:,0]-pairs1__[:,0], pairs0[:,1]-pairs1__[:,1], pairs0[:,2]-pairs1__[:,2], \
                             figure=fig, line_width=0.5, scale_factor=1)    
    mlab.title('Fused PC: ' + str(iFrame0) + '-' + str(iFrame1),)
    mlab.axes(xlabel='X', ylabel='Y', zlabel='Z')
    
    
    
#    fig = mlab.figure(bgcolor=(0, 0, 0), size=(1200, 800))
#    mlab.points3d(FusedPC___[:,0], FusedPC___[:,1], FusedPC___[:,2],
#                         Colors4FusedPC, mode="point", figure=fig)    
#    mlab.title('Fused PC - PlanarPts: ' + str(iFrame0) + '-' + str(iFrame1),)
#    mlab.axes(xlabel='X', ylabel='Y', zlabel='Z')
    
    
    
    mlab.show()



def GetErrorEulers(relRs0, relRs1):
    assert relRs0.shape[0] == relRs1.shape[0]
    nDatas = relRs0.shape[0]    
    ErrorEulers = np.zeros((nDatas,3), dtype=np.float32)
    for i in range(nDatas):
        errorR = dot(np.linalg.inv(relRs0[i,:,:]), relRs1[i,:,:])
        eulers = RotateMat2EulerAngle_XYZ(errorR)
        ErrorEulers[i,:] = eulers        
    return ErrorEulers
    
    

def GetErrorRTs(poses, poses_, Tr, isPlot):        
    # ----- then compare with the ground truth
    relRs, relTs, relEulers, diffNormRelEulers, diffNormRelTs = GetLidarDiffRels(poses, Tr)
    relRs_, relTs_, relEulers_, diffNormRelEulers_, diffNormRelTs_ = GetLidarDiffRels(poses_, Tr)
    
    errorRelEulers = GetErrorEulers(relRs, relRs_)
    errorRelTs = relTs_ - relTs
    errorRelEulersNorm = LA.norm(errorRelEulers, axis=1)
    errorRelTsNorm = LA.norm(errorRelTs, axis=1)
    
    if isPlot == True:
        plt.figure()
        plt.subplot(2, 4, 1)
        plt.title('relEulers')
        plt.plot(relEulers,'.')
        plt.subplot(2, 4, 2)
        plt.title('relTs')
        plt.plot(relTs,'.')
        plt.subplot(2, 4, 3)
        plt.title('diffNormRelEulers')
        plt.plot(diffNormRelEulers,'.')
        plt.subplot(2, 4, 4)
        plt.title('diffNormRelTs')
        plt.plot(diffNormRelTs,'.')
        plt.subplot(2, 4, 5)
        plt.title('relEulers_')
        plt.plot(relEulers_,'.')
        plt.subplot(2, 4, 6)
        plt.title('relTs_')
        plt.plot(relTs_,'.')
        plt.subplot(2, 4, 7)
        plt.title('diffNormRelEulers_')
        plt.plot(diffNormRelEulers_,'.')
        plt.subplot(2, 4, 8)
        plt.title('diffNormRelTs_')
        plt.plot(diffNormRelTs_,'.')
        
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.title('errorRelEulersNorm')
        plt.plot(errorRelEulersNorm,'.')
        plt.subplot(1, 2, 2)
        plt.title('errorRelTsNorm')
        plt.plot(errorRelTsNorm,'.')
        
#        plt.figure()
#        plt.subplot(1, 3, 1)
#        plt.title('errorRelEulers-X')
#        plt.plot(errorRelEulers[:,0],'.')
#        plt.subplot(1, 3, 2)
#        plt.title('errorRelEulers-Y')
#        plt.plot(errorRelEulers[:,1],'.')
#        plt.subplot(1, 3, 3)
#        plt.title('errorRelEulers-Z')
#        plt.plot(errorRelEulers[:,2],'.')
#        
#        plt.figure()
#        plt.subplot(1, 3, 1)
#        plt.title('errorRelTs-X')
#        plt.plot(errorRelTs[:,0],'.')
#        plt.subplot(1, 3, 2)
#        plt.title('errorRelTs-Y')
#        plt.plot(errorRelTs[:,1],'.')
#        plt.subplot(1, 3, 3)
#        plt.title('errorRelTs-Z')
#        plt.plot(errorRelTs[:,2],'.')
        
        plt.show
        
    
    GroundTruthRels = []
    GroundTruthRels.append(relRs)
    GroundTruthRels.append(relTs)
    GroundTruthRels.append(relEulers)
    GroundTruthRels.append(diffNormRelEulers)
    GroundTruthRels.append(diffNormRelTs)
    
    EstimatedRels = []
    EstimatedRels.append(relRs_)
    EstimatedRels.append(relTs_)
    EstimatedRels.append(relEulers_)
    EstimatedRels.append(diffNormRelEulers_)
    EstimatedRels.append(diffNormRelTs_)
    
    return GroundTruthRels, EstimatedRels, errorRelEulers, errorRelTs
    

    
    
    
    
    
    
    

