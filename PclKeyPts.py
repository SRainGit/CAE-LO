#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 10:48:15 2020

@author: rain
"""

import os
import numpy as np
from scipy import io
import mayavi.mlab as mlab

import PCLKeypoint

from Transformations import *
from Dirs import *


def ensure_keypoint_number(frame_keypoint_np, frame_pc_np, keypoint_num):
    if frame_keypoint_np.shape[0] == keypoint_num:
        return frame_keypoint_np
    elif frame_keypoint_np.shape[0] > keypoint_num:
        return frame_keypoint_np[np.random.choice(frame_keypoint_np.shape[0], keypoint_num, replace=False), :]
    else:
        additional_frame_keypoint_np = frame_pc_np[np.random.choice(frame_pc_np.shape[0], keypoint_num-frame_keypoint_np.shape[0], replace=False), :]
        frame_keypoint_np = np.concatenate((frame_keypoint_np, additional_frame_keypoint_np), axis=0)
        return frame_keypoint_np



#------------------------------------------------------------------------------
input_pc_num = 16384
surface_normal_len = 4
node_num = 4

desired_keypoint_num = 1024
is_ensure_keypoint_num = True


# method = 'iss'
iss_salient_radius = 2
iss_non_max_radius = 2
iss_gamma_21 = 0.975
iss_gamma_32 = 0.975
iss_min_neighbors = 5
threads = 0

# method = 'harris'
radius = 1
nms_threshold = 0.001
threads = 10

# method = 'sift'
min_scale = 0.5
n_octaves = 4
n_scales_per_octave = 8
min_contrast = 0.1


#------------------------------------------------------------------------------------
iMethod = 2
methods = ['cae-lo', '3dfeat-net', 'usip', 'iss', 'harris', 'sift']
method = methods[iMethod]

strSequence = '00'
iFrame = 500
DirNumpFiles = '/media/rain/Win10_F/USIP/kitti/data_odometry_velodyne/numpy/'
DataDir = strDataBaseDir+strSequence+'/velodyne/'


# load PC data
PC = np.fromfile(str(DataDir+str(iFrame).zfill(6)+".bin"), dtype=np.float32, count=-1).reshape([-1,4])


# load norm data
curFolder = DirNumpFiles + strSequence + '/np_0.20_20480_r90_sn/';
pc_np_file = os.path.join(curFolder, '%06d.npy' % iFrame)
pc_np = np.load(pc_np_file)  # Nx4, x, y, z, reflectance
# random choice
choice_idx = np.random.choice(pc_np.shape[0], input_pc_num, replace=False)
pc_np = pc_np[choice_idx, :]
pc_np = pc_np[:, 0:3]  # Nx3


# convert to torch tensor
anc_pc = pc_np.transpose().astype(np.float32)  # 3xN



# get pcl keypoints
if method == 'iss':
    anc_keypoints_list = []
    frame_pc_np = anc_pc.T  # Nx3
    frame_keypoint_np = PCLKeypoint.keypointIss(frame_pc_np,
                                                iss_salient_radius,
                                                iss_non_max_radius,
                                                iss_gamma_21,
                                                iss_gamma_32,
                                                iss_min_neighbors,
                                                threads)  # Mx3
    if is_ensure_keypoint_num:
        frame_keypoint_np = ensure_keypoint_number(frame_keypoint_np, frame_pc_np, desired_keypoint_num)
            
elif method == 'harris':
    frame_pc_np = anc_pc.T  # Nx3
    frame_keypoint_np = PCLKeypoint.keypointHarris(frame_pc_np,
                                                   radius,
                                                   nms_threshold,
                                                   threads)  # Mx3
    if is_ensure_keypoint_num:
        frame_keypoint_np = ensure_keypoint_number(frame_keypoint_np, frame_pc_np, desired_keypoint_num)
            
elif method == 'sift':
    frame_pc_np = anc_pc.T  # Nx3
    frame_keypoint_np = PCLKeypoint.keypointSift(frame_pc_np,
                                                 min_scale,
                                                 n_octaves,
                                                 n_scales_per_octave,
                                                 min_contrast)  # Mx3
    if is_ensure_keypoint_num:
        frame_keypoint_np = ensure_keypoint_number(frame_keypoint_np, frame_pc_np, desired_keypoint_num)
            
elif method == 'random':
    frame_pc_np = anc_pc.T  # Nx3
    frame_keypoint_np = frame_pc_np[np.random.choice(frame_pc_np.shape[0], desired_keypoint_num, replace=False), :]

if iMethod > 2:
    KeyPts = frame_keypoint_np


# get keypoints of methods 0,1,2 from file
if iMethod == 0:
    KeyPtsDir = str(strDataBaseDir+strSequence+'/KeyPts/')
    keyPtsData = io.loadmat(str(KeyPtsDir+str(iFrame).zfill(6)+'.bin.mat'))
    KeyPts = keyPtsData['KeyPts']
elif iMethod == 1:
    fileName = str(str3DFeatNetDir + 'Descriptors/' + strSequence + '/' + str(iFrame).zfill(6)+'.bin')
    keyPtsData = np.fromfile(fileName, dtype=np.float32, count=-1).reshape([-1,35])
    KeyPts = keyPtsData[:,0:3]
elif iMethod == 2:
    fileName = str(strUsipKeyPtsDir + strSequence + '/' + str(iFrame).zfill(6)+'.bin') 
    keyPtsData = np.fromfile(fileName, dtype=np.float32, count=-1).reshape([-1,3])
    KeyPts = keyPtsData

# rotate keypoints
if iMethod >= 2:
    R90 = EulerAngle2RotateMat(-math.pi/2,0,-math.pi/2,'xyz')
    KeyPts = np.dot(R90, KeyPts.T).T



PtSize = 0.4
color_PC = 0.0*np.ones([PC.shape[0],1],dtype=np.float32)
color_KeyPts = 1.0*np.ones([KeyPts.shape[0],1],dtype=np.float32)

fig = mlab.figure(bgcolor=(0, 0, 0), size=(1640, 1500))
# fig = mlab.figure(bgcolor=(1, 1, 1), size=(1640, 1500))
nodeFusedPC = mlab.points3d(PC[:,0], PC[:,1], PC[:,2], mode="point",  figure=fig)
nodeFusedPC.glyph.scale_mode = 'scale_by_vector'
nodeFusedPC.mlab_source.dataset.point_data.scalars = color_PC

node = mlab.points3d(KeyPts[:,0], KeyPts[:,1], KeyPts[:,2], scale_factor=PtSize, figure=fig)
node.glyph.scale_mode = 'scale_by_vector'
node.mlab_source.dataset.point_data.scalars = color_KeyPts

#mlab.axes(x_axis_visibility = True)

# mlab.view(270, 70, 100, [0,0,0])  # for 01-495
mlab.view(0, 0, 150, [0,0,0])  # for 00-1


# filename = str(iFrame) + '-' + str(iMethod) + '.tiff'
# mlab.savefig(filename = filename)


mlab.show()












