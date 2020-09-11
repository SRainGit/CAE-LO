#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 11:55:58 2019

@author: rain
"""

import numpy as np
from time import time
import mayavi.mlab as mlab
from numpy import linalg as LA

from sklearn.neighbors import NearestNeighbors


PC = np.fromfile('/media/rain/Win10_F/KITTI_odometry/velodyne/sequences/00/velodyne/000498.bin', dtype=np.float32, count=-1).reshape([-1,4])
PC = PC[:,0:3]

nNeighbors = 64

t0 = time()
nbrs = NearestNeighbors(n_neighbors=nNeighbors, radius=5.0, algorithm='ball_tree').fit(PC)
t1 = time()
print(round(t1-t0, 2), 's')

distances, indices = nbrs.kneighbors(PC)
t2 = time()
print(round(t2-t1, 2), 's')

neighborPC = PC[indices,:]

center = np.mean(neighborPC, axis=1)
center = np.tile(center, (1, nNeighbors)).reshape(center.shape[0], nNeighbors, center.shape[1])
neighborPC_ = neighborPC - center
#PC_ = np.tile(PC, (1, nNeighbors)).reshape(PC.shape[0], nNeighbors, PC.shape[1])
#neighborPC_ = neighborPC - PC_


scores = np.zeros((neighborPC_.shape[0],), dtype=np.float32)
for iNeighbor in range(neighborPC_.shape[0]):    
    covMat = np.cov(neighborPC_[iNeighbor,:,:], rowvar=0)
    eigVals, eigVector = np.linalg.eig(covMat)
    score = LA.norm(eigVals)
    scores[iNeighbor] = score


neighborScores = scores[indices]
maximumIdx = np.argmin(neighborScores, axis=1)

keyPts = PC[maximumIdx==0,:]

#keyPts = []
#
#keyPts = np.array(keyPts, dtype=np.float32)
print('cntkeyPts =', keyPts.shape[0])
t3 = time()
print(round(t3-t2, 2), 's')



shift4Show = 10
fig = mlab.figure(bgcolor=(0, 0, 0), size=(1500, 900))
PtSize  = 0.1

Colors0=np.ones((PC.shape[0],1), dtype=np.float32)*1.0
Colors1=np.ones((keyPts.shape[0],1), dtype=np.float32)*0.0

node = mlab.points3d(PC[:,0], PC[:,1], PC[:,2], mode="point", figure=fig)
node.glyph.scale_mode = 'scale_by_vector'
node.mlab_source.dataset.point_data.scalars = Colors0

node = mlab.points3d(keyPts[:,0], keyPts[:,1], keyPts[:,2], scale_factor=PtSize, figure=fig)
node.glyph.scale_mode = 'scale_by_vector'
node.mlab_source.dataset.point_data.scalars = Colors1
    
mlab.show()





