#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 10:29:20 2019

@author: rain
"""

import os

Disk_F = '/media/rain/Win10_F/'
# Disk_F = 'F:\\'

strBaseDir = os.path.join(Disk_F,'KITTI_odometry')
str3DFeatNetDir = os.path.join(Disk_F,'KITTI_odometry', 'output_3DFeatNet');
strUsipBaseDir =   os.path.join(Disk_F, 'USIP')


strGroundTruthPosesDir = os.path.join(strBaseDir, 'poses')
strEstimatedPosesDir = os.path.join(strBaseDir, 'poses_')
strDejumpyedPosesDir = os.path.join(strBaseDir, 'poses__')
strRefinedPosesDir = os.path.join(strBaseDir, 'poses___')
#strRefinedPosesDir = os.path.join(strBaseDir, 'poses___WithoutBackwardUpdate')
strClosedPosesDir = os.path.join(strBaseDir, 'poses____')

strDataBaseDir = os.path.join(strBaseDir, 'velodyne', 'sequences')
strCalibDataDir = os.path.join(strBaseDir, 'calib')

strRespondNetModelPath = './TrainedModels/SphericalRingPCRespondLayer.h5'
strVoxelPatchEncoderPath = './TrainedModels/EncoderModel4VoxelPatch.h5'




strUsipPcDir = os.path.join(strUsipBaseDir, '/kitti', 'data_odometry_velodyne', 'numpy')
#strUsipKeyPtsDir = strUsipBaseDir + 'KeyPts/kitti/tsf/'
strUsipKeyPtsDir = os.path.join(strUsipBaseDir, 'KeyPts', 'kitti', 'tsf_1024')
strUsipDescDir = os.path.join(strUsipBaseDir, 'Descriptors')
strIssKeyPtsDir = os.path.join(strUsipBaseDir, 'KeyPts', 'kitti', 'iss_1024')
strHarrisKeyPtsDir = os.path.join(strUsipBaseDir, 'KeyPts', 'kitti', 'harris_1024')
strSiftKeyPtsDir = os.path.join(strUsipBaseDir, 'KeyPts', 'kitti', 'sift_1024')