#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 14:51:34 2019

@author: rain
"""
import numpy as np
import math
from numpy import dot as dot
from math import cos as cos
from math import sin as sin
from math import acos as acos
from numpy import linalg as LA
import copy

RADIAN2DEGREE = 180/math.pi


def TranslatePtsIntoWorldFrame(pose, Tr, Pts):
    onesVector = np.ones([Pts.shape[0],1],dtype=np.float32)
    Pts_ = np.c_[Pts, onesVector]
    Pts_ = np.dot(pose, np.r_[np.dot(Tr, Pts_.T),onesVector.T])
    return Pts_.T


            

#--------------for general poses
def GetDiffRels(poses):        
    RelRs, RelTs, RelEulers = GetRelsVars(poses)
    diffRelEulers = LA.norm(np.abs(RelEulers[1:RelEulers.shape[0],:]) - np.abs(RelEulers[0:RelEulers.shape[0]-1,:]), axis=1)
    diffRelTs = LA.norm(np.abs(RelTs[1:RelTs.shape[0],:]) - np.abs(RelTs[0:RelTs.shape[0]-1,:]), axis=1)    
    return RelRs, RelTs, RelEulers, diffRelEulers, diffRelTs

def GetRelsVars(poses):
    RelRs, RelTs = GetRelRsAndTs(poses)
    RelEulers = ConvertRelRsIntoEulers(RelRs)    
    return RelRs, RelTs , RelEulers

def GetRelRsAndTs(poses):
    RelRs = np.zeros((poses.shape[0]-1, 3, 3), dtype=np.float64)
    RelTs = np.zeros((poses.shape[0]-1, 3), dtype=np.float64)
    for iFrame in range(poses.shape[0]-1):
        pose0 = poses[iFrame,:]
        pose1 = poses[iFrame+1,:]
        R, T =  GetRelRtBetween2Poses(pose0, pose1)        
        RelRs[iFrame, :, :] = R
        RelTs[iFrame, :] = T.reshape(1,3)
    return RelRs, RelTs

# from pose1 to pose0
def GetRelRtBetween2Poses(pose0, pose1):
    R0, T0 = GetRtFromOnePose(pose0)
    R0_inv = np.linalg.inv(R0)
    T0_inv = -dot(R0_inv, T0)
    R1, T1 = GetRtFromOnePose(pose1)
    R = dot(R0_inv, R1)
    T = dot(R0_inv, T1) + T0_inv
    return R, T


#-----------------for lidar poses
# from pose1 to pose0
def GetLidarRelRtBetween2Poses(pose0, pose1, R_Tr, T_Tr, R_Tr_inv, T_Tr_inv):
    R0, T0 = GetRtFromOnePose(pose0)
    R0_inv = np.linalg.inv(R0)
    T0_inv = -dot(R0_inv, T0)
    R1, T1 = GetRtFromOnePose(pose1)
    R = dot(R_Tr_inv, dot(R0_inv, dot(R1, R_Tr)))
    T = dot(R_Tr_inv, dot(R0_inv, dot(R1, T_Tr) + T1) + T0_inv) + T_Tr_inv
    return R, T
    
def GetLidarRelRsAndTs(poses, Tr):
    R_Tr, T_Tr = GetRtFromOnePose(Tr)
    R_Tr_inv = np.linalg.inv(R_Tr)
    T_Tr_inv = -dot(R_Tr_inv, T_Tr)
    RelRs = np.zeros((poses.shape[0]-1, 3, 3), dtype=np.float64)
    RelTs = np.zeros((poses.shape[0]-1, 3), dtype=np.float64)
    for iFrame in range(poses.shape[0]-1):
        pose0 = poses[iFrame,:]
        pose1 = poses[iFrame+1,:]
        R, T =  GetLidarRelRtBetween2Poses(pose0, pose1, R_Tr, T_Tr, R_Tr_inv, T_Tr_inv)        
        RelRs[iFrame, :, :] = R
        RelTs[iFrame, :] = T.reshape(1,3)        
    return RelRs, RelTs
    
def GetLidarRelsVars(poses, Tr):
    RelRs, RelTs = GetLidarRelRsAndTs(poses, Tr)
    RelEulers = ConvertRelRsIntoEulers(RelRs)    
    return RelRs, RelTs, RelEulers
    
def GetLidarDiffRels(poses, Tr):        
    RelRs, RelTs, RelEulers = GetLidarRelsVars(poses, Tr)
    diffNormRelEulers = LA.norm(RelEulers[1:RelEulers.shape[0],:]-RelEulers[0:RelEulers.shape[0]-1,:], axis=1)
    diffNormRelTs = LA.norm(RelTs[1:RelTs.shape[0],:]-RelTs[0:RelTs.shape[0]-1,:], axis=1)    
    return RelRs, RelTs, RelEulers, diffNormRelEulers, diffNormRelTs
 

#--------------supporting functions
def GetRsAndTsFromPoses(poses):
    nFrames = poses.shape[0]
    Rs = np.zeros((nFrames,3,3), dtype=np.float32)
    Ts = np.zeros((nFrames,3), dtype=np.float32)
    for iFrame in range(nFrames):
        R, T = GetRtFromOnePose(poses[iFrame,:])
        Rs[iFrame,:,:] = R
        Ts[iFrame,:] = T.reshape(3,)
    return Rs, Ts
    
def GetRtFromOnePose(pose):
    pose = pose.reshape(3,4)
    R = pose[:,0:3]
    T = pose[:,3].reshape(3,1)
    return R, T

def ConvertRelRsIntoEulers(relRs):
    Eulers = np.zeros((relRs.shape[0], 3), dtype=np.float64)
    for i in range(relRs.shape[0]):
        Eulers[i,:] = RotateMat2EulerAngle_XYZ(relRs[i,:,:])         
    return Eulers





#----------------------------Basic options for rotation matrix
def RotateMat2EulerAngle_XYZ(R):
    angles=np.zeros((3,))
    angles[0]=math.atan2(R[2,1],R[2,2])*RADIAN2DEGREE
    angles[1]=math.atan2(-R[2,0],math.sqrt(math.pow(R[2,1],2)+math.pow(R[2,2],2)))*RADIAN2DEGREE
    angles[2]=math.atan2(R[1,0],R[0,0])*RADIAN2DEGREE
    return angles

def EulerAngle2RotateMat(angX,angY,angZ,RotateSequnce):
    R=np.eye(3, dtype=np.float64)
    R_X=np.array([[1,0,0],
                 [0,cos(angX),-sin(angX)],
                 [0,sin(angX),cos(angX)]],
                 dtype=np.float64)
    R_Y=np.array([[cos(angY),0,sin(angY)],
                  [0,1,0],
                  [-sin(angY),0,cos(angY)]],
                 dtype=np.float64)
    R_Z=np.array([[cos(angZ),-sin(angZ),0],
                  [sin(angZ),cos(angZ),0],
                  [0,0,1]],
                 dtype=np.float64)
    for i in range(3):
        if RotateSequnce[i]=='x' or RotateSequnce[i]=='X':
            R=np.dot(R_X,R)
        elif RotateSequnce[i]=='y' or RotateSequnce[i]=='Y':
            R=np.dot(R_Y,R)
        elif RotateSequnce[i]=='z' or RotateSequnce[i]=='Z':
            R=np.dot(R_Z,R)
        else:
            print('Error, please check.')        
    return R    

def RotMat2Quatern(R): 
    # convert a rotate matrix into the corresponding quatern
    # wiki URL: https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#cite_note-5
    # paper URL: http://arc.aiaa.org/doi/pdf/10.2514/2.4654
    K = np.zeros((4,4), dtype = np.float64)    
    t = 1/3    
    K[0,0] = t * (R[0,0] - R[1,1] - R[2,2])
    K[0,1] = t * (R[1,0] + R[0,1])
    K[0,2] = t * (R[2,0] + R[0,2])
    K[0,3] = t * (R[1,2] - R[2,1])  
    K[1,0] = t * (R[1,0] + R[0,1])
    K[1,1] = t * (R[1,1] - R[0,0] - R[2,2])
    K[1,2] = t * (R[2,1] + R[1,2])
    K[1,3] = t * (R[2,0] - R[0,2])   
    K[2,0] = t * (R[2,0] + R[0,2])
    K[2,1] = t * (R[2,1] + R[1,2])
    K[2,2] = t * (R[2,2] - R[0,0] - R[1,1])
    K[2,3] = t * (R[0,1] - R[1,0])    
    K[3,0] = t * (R[1,2] - R[2,1])
    K[3,1] = t * (R[2,0] - R[0,2])
    K[3,2] = t * (R[0,1] - R[1,0])
    K[3,3] = t * (R[0,0] + R[1,1] + R[2,2])     
    eigVals, eigVector = np.linalg.eig(K)
    sortIdx = np.argsort(eigVals)
    q = eigVector[:,sortIdx[3]].T
    q = np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)    
    return q    

def Quatern2RotMat(q):
   R = np.zeros((3,3), dtype=np.float32)
   R[0,0] = 1 - 2*q[2]*q[2] - 2*q[3]*q[3]
   R[0,1] = 2*q[1]*q[2] - 2*q[3]*q[0]
   R[0,2] = 2*q[2]*q[0] + 2*q[3]*q[1]
   R[1,0] = 2*q[1]*q[2] + 2*q[3]*q[0]
   R[1,1] = 1 - 2*q[1]*q[1] - 2*q[3]*q[3]
   R[1,2] = 2*q[2]*q[3] - 2*q[1]*q[0]
   R[2,0] = 2*q[1]*q[3] - 2*q[2]*q[0]
   R[2,1] = 2*q[2]*q[3] + 2*q[1]*q[0]
   R[2,2] = 1 - 2*q[1]*q[1] - 2*q[2]*q[2]
   return R

def Quatern2AngleAndAxis(q):
    if (LA.norm(q)-1) > 0.0001:
        print('Error of q.')        
    halfTheta=acos(q[0]);
    w1=q[1]/sin(halfTheta);
    w2=q[2]/sin(halfTheta);
    w3=q[3]/sin(halfTheta);    
    AngleAndAxis=np.array([2*halfTheta,w1,w2,w3], dtype=np.float64)
    return AngleAndAxis

def AngleAxis2Quatern(angle, axisVector):
    q = np.zeros((4,), dtype=np.float32)    
    halfAngle = angle/2
    sinHalfAngle = sin(halfAngle)
    q[0] = cos(halfAngle)
    q[1] = axisVector[0]*sinHalfAngle
    q[2] = axisVector[1]*sinHalfAngle
    q[3] = axisVector[2]*sinHalfAngle    
    return q

