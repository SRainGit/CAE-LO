#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 16:21:56 2019

@author: rain
"""

import numpy as np
from time import time
import mayavi.mlab as mlab
from numpy import linalg as LA
from scipy.spatial.distance import cdist
import copy
from sklearn.neighbors import NearestNeighbors

from Match import *
from Transformations import *


def RandomDownSample4PC(PC, ratio):
    RandIdxes = np.random.random((int(PC.shape[0]*ratio),))*PC.shape[0]
    RandIdxes = np.array(RandIdxes, dtype=np.int32)
    PC_ = PC[RandIdxes,:]
    return PC_    

# refered to https://stackoverflow.com/questions/20120384/iterative-closest-point-icp-implementation-on-python
def ICP(PC0, PC1, maxIterTimes=50, minIterTimes=20-1, inlierThreshold=0.5, smallShiftThreshold=0.05, decay_rate = 0.9, ep=0.001):
    R_star = np.eye(3, dtype=np.float64)
    T_star = np.zeros((3,1), dtype=np.float64)    
    
    for iIter in range(maxIterTimes):
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(PC0)
        distances, indices = nbrs.kneighbors(PC1)
        
        # extract the inlier pairs
        idx1 = (distances < inlierThreshold).flatten()
        fullIdx0 = np.arange(PC0.shape[0])
        idx0 = fullIdx0[indices[idx1]].flatten()
        if idx0.shape[0] < 100:
            print('ICP iters:', iIter+1, ',  inliers:', idx0.shape[0], ',  inlierThreshold:', round(inlierThreshold, 5))
            return  R_star, T_star, False
        
        inliers0 = PC0[idx0,:]
        inliers1 = PC1[idx1,:]
        
        # solve RT
        R, T, isCredible = SolveRT(inliers0, inliers1)
        
        # update pairs1 and RT
        PC1 = (np.dot(R, PC1.T) + T).T 
        R_star = np.dot(R, R_star)
        T_star = np.dot(R, T_star) + T       
        
        # check if need to break
        eulers = RotateMat2EulerAngle_XYZ(R)
        normEulers = LA.norm(eulers)
        normT = LA.norm(T)
        if iIter >= minIterTimes:
            if normEulers < ep and normT < ep:
                break
#        print(normEulers, normT, inlierThreshold)
            
        # update threshold
        if normEulers < smallShiftThreshold and normT < smallShiftThreshold:
            inlierThreshold *= decay_rate
#        inlierThreshold *= decay_rate
        
    isSuccess = True
    print('ICP iters:', iIter+1, ',  inliers:', idx0.shape[0], ',  inlierThreshold:', round(inlierThreshold, 5))
    return  R_star, T_star, isSuccess
            


def GetPtsInliners(PC0, PC1, inlierThreshold):
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(PC0)
    distances, indices = nbrs.kneighbors(PC1)        
    # extract the inlier pairs
    idx1 = (distances < inlierThreshold).flatten()
    fullIdx0 = np.arange(PC0.shape[0])
    idx0 = fullIdx0[indices[idx1]].flatten()        
    inliers0 = PC0[idx0,:]
    inliers1 = PC1[idx1,:]
    return inliers0, inliers1


def GetPlanarPtsInliners(PtsWithNorm0, PtsWithNorm1, inlierThreshold0, inlierThreshold1):
    PC0 = PtsWithNorm0[:,0:3]
    Norms0 = PtsWithNorm0[:,3:6]    
    PC1 = PtsWithNorm1[:,0:3]
    Norms1 = PtsWithNorm1[:,3:6]

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(PC0)
    distances, indices = nbrs.kneighbors(PC1)
    
    # extract the inlier pairs
    idx1 = (distances < inlierThreshold1).flatten()
    fullIdx0 = np.arange(PC0.shape[0])
    idx0 = fullIdx0[indices[idx1]].flatten()
    
    inliers0 = PC0[idx0,:]
    inliers1 = PC1[idx1,:]
    
    norms1 = Norms1[idx1,:]
    vetors = inliers0 - inliers1
    dist2Planes = np.sum(norms1*vetors, axis=1)
    pedals = inliers1 + norms1*np.tile(dist2Planes.reshape(dist2Planes.shape[0],1),[1,3])        
    
    distances = LA.norm((pedals-inliers1), axis=1)
    idx = (distances < inlierThreshold0).flatten()     
    pedals = pedals[idx,:]
    inliers1 = inliers1[idx,:]
    return pedals, inliers1


def ShowPts(fig, pts, ptSize, color):
    colors = color*np.ones((pts.shape[0],1), dtype=np.float32)
    if ptSize <= 0:
        node = mlab.points3d(pts[:,0], pts[:,1], pts[:,2], mode="point",  figure=fig)
    else:
        node = mlab.points3d(pts[:,0], pts[:,1], pts[:,2], scale_factor=ptSize, figure=fig)    
    node.glyph.scale_mode = 'scale_by_vector'
    node.mlab_source.dataset.point_data.scalars = colors
    return 0
        

def ICP_Pt2PtAndPt2Plane(PC0, PC1, PtsWithNorm0, PtsWithNorm1, maxIterTimes=50, minIterTimes=20-1, 
                         inlierThreshold0=0.5, decay_rate0 = 0.9, 
                         inlierThreshold1=2.0, decay_rate1 = 0.5,
                         smallShiftThreshold=0.1, ep=0.01):
    R_star = np.eye(3, dtype=np.float64)
    T_star = np.zeros((3,1), dtype=np.float64)
    
    # sample planarPts
    nMaxPts = 2000
    if PtsWithNorm1.shape[0] > nMaxPts:
        RandIdxes = np.random.random((nMaxPts,))
        RandIdxes = RandIdxes*(PtsWithNorm1.shape[0])
        RandIdxes = np.array(RandIdxes, dtype=np.int32)
        PtsWithNorm1 = PtsWithNorm1[RandIdxes,:]
    
    isSuccess = True
    minNumOfInputPts = 200
    for iIter in range(maxIterTimes):
        inliers0_pts, inliers1_pts = GetPtsInliners(PC0, PC1, inlierThreshold0)
        
        if iIter < 100:
            inliers0_planarPts, inliers1_planarPts = GetPlanarPtsInliners(PtsWithNorm0, PtsWithNorm1, inlierThreshold0, inlierThreshold1)
            inliers0 = np.r_[inliers0_pts, inliers0_planarPts]
            inliers1 = np.r_[inliers1_pts, inliers1_planarPts]
        else:
            inliers0 = inliers0_planarPts
            inliers1 = inliers1_planarPts
        
#        print('inliers0_pts', inliers0_pts.shape[0], 'inlierThreshold0', inlierThreshold0)
#        print('inliers0_planarPts', inliers0_planarPts.shape[0], 'inlierThreshold1', inlierThreshold1)
#        # visualization        
#        fig = mlab.figure(bgcolor=(0, 0, 0), size=(1640, 1500))
#        ShowPts(fig, inliers0_pts, ptSize=0.1, color=1.0)
#        ShowPts(fig, inliers1_pts, ptSize=0.1, color=1.0)
#        ShowPts(fig, inliers0_planarPts, ptSize=0.01, color=0.1)
#        ShowPts(fig, inliers1_planarPts, ptSize=0.01, color=0.1)
#        
#        vectors = inliers0 - inliers1
#        mlab.quiver3d(inliers1[:,0], inliers1[:,1], inliers1[:,2], \
#                             vectors[:,0], vectors[:,1], vectors[:,2], \
#                             figure=fig, line_width=0.5, scale_factor=1)
#        mlab.show()
                
        
        if inliers0.shape[0] < minNumOfInputPts:
            if iIter < 1:
                isSuccess = False
            break
        
        # solve RT
        R, T, isCredible = SolveRT(inliers0, inliers1)
        
        # update pairs1 and RT
        PC1 = (np.dot(R, PC1.T) + T).T 
        PtsWithNorm1[:,0:3] = (np.dot(R, PtsWithNorm1[:,0:3].T) + T).T 
        R_star = np.dot(R, R_star)
        T_star = np.dot(R, T_star) + T       
        
        # check if need to break
        eulers = RotateMat2EulerAngle_XYZ(R)
        normEulers = LA.norm(eulers)
        normT = LA.norm(T)
        if iIter >= minIterTimes:
            if normEulers < ep and normT < ep:
                break
            
        # update threshold
        if normEulers < smallShiftThreshold and normT < smallShiftThreshold:
            inlierThreshold0 *= decay_rate0
            inlierThreshold1 *= decay_rate1
        
    
    print('ICP iters:', iIter+1, ', inliers0:', inliers0_pts.shape[0], ', inliers1:', inliers0_planarPts.shape[0], 
          ', th0:', round(inlierThreshold0, 5), ', th1:', round(inlierThreshold1, 5))
    return  R_star, T_star, isSuccess



    

if __name__ == "__main__":
    strSequence = '01'
    iFrame0 = 0
    iFrameStep = 1
    iFrame1 = iFrame0 + iFrameStep
    DataDir = '/media/rain/Win10_F/KITTI_odometry/velodyne/sequences/'+strSequence+'/velodyne/'
    FileName0 = DataDir + str(iFrame0).zfill(6)+'.bin'
    FileName1 = DataDir + str(iFrame1).zfill(6)+'.bin'
    
    PC0 = np.fromfile(FileName0, dtype=np.float32, count=-1).reshape([-1,4])[:,0:3]
    PC1 = np.fromfile(FileName1, dtype=np.float32, count=-1).reshape([-1,4])[:,0:3]
    
    t0 = time()
    downSampleRatio = 0.5
    PC0_ = RandomDownSample4PC(PC0, downSampleRatio)
    PC1_ = RandomDownSample4PC(PC1, downSampleRatio)
    t1 = time()
    print(round(t1-t0, 2), 's, for RandomDownSample4PC')
    
    
    R_ICP, T_ICP, isSuccess = ICP(PC0_, PC1_)
    t2 = time()
    print(round(t2-t1, 2), 's, for ICP')
    
    PC1_ = (np.dot(R_ICP, PC1.T) + T_ICP.reshape(3,1)).T
    FusedPC = np.r_[PC0, PC1_]
    
    
    Colors0=np.ones((PC0.shape[0],1), dtype=np.float32)*1.0
    
    Colors1=np.ones((PC1.shape[0],1), dtype=np.float32)*0.0
    
    Colors4FusedPC=np.r_[Colors0, Colors1]
    Colors4FusedPC=Colors4FusedPC.flatten()
    
    
    shift4Show = 0
    fig = mlab.figure(bgcolor=(0, 0, 0), size=(1500, 900))
    
    node = mlab.points3d(PC0[:,0], PC0[:,1], PC0[:,2], mode="point", figure=fig)
    node.glyph.scale_mode = 'scale_by_vector'
    node.mlab_source.dataset.point_data.scalars = Colors0
    
    node = mlab.points3d(PC1[:,0], PC1[:,1], PC1[:,2]+shift4Show, mode="point", figure=fig)
    node.glyph.scale_mode = 'scale_by_vector'
    node.mlab_source.dataset.point_data.scalars = Colors1    
    
    mlab.title('Feature Matching')
    mlab.axes(xlabel='X', ylabel='Y', zlabel='Z')
    
    
    
    fig = mlab.figure(bgcolor=(0, 0, 0), size=(1200, 800))
    mlab.points3d(FusedPC[:,0], FusedPC[:,1], FusedPC[:,2],
                         Colors4FusedPC, mode="point", figure=fig)       
    mlab.title('Fused PC')
    mlab.axes(xlabel='X', ylabel='Y', zlabel='Z')   
    
    
    
    mlab.show()
    
    
    
    