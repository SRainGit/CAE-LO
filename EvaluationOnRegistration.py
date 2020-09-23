#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 10:11:15 2019

@author: rain
"""

import os
import numpy as np
from scipy import io
import math
from numpy import linalg as LA
 
from Dirs import *
from Transformations import *
from Visualization import *



NUM_METHODS = 3

t_RRE = 1.0  # degrees
t_RTE = 0.5

AllErrors_Eulers = []
AllErrors_T = []
AllProportions = []
AllTrailCounts = []

iFrameStep = 1
for iKeyPt in range(NUM_METHODS):
    AllErrors_Eulers.append([])
    AllErrors_T.append([])
    AllProportions.append([])
    AllTrailCounts.append([])
    
    for iDesc in range(NUM_METHODS):
        AllErrors_Eulers[iKeyPt].append([])
        AllErrors_T[iKeyPt].append([])
        AllProportions[iKeyPt].append([])
        AllTrailCounts[iKeyPt].append([])
        
for iKeyPt in range(NUM_METHODS):
    for iDesc in range(NUM_METHODS):
#for iKeyPt in [0,1]:
#    for iDesc in [0,1]:
        CurAllErrorAngles = np.zeros((1,3), dtype=np.float32)
        CurAllErrorTs = np.zeros((1,3), dtype=np.float32)
        CurAllProportions = np.zeros((1,1), dtype=np.float32)
        CurAllTrailCounts = np.zeros((1,1), dtype=np.float32)

        for iSequence in range(0, 11, 1):
            print(iKeyPt, iDesc, iSequence)
            strSequence = str(iSequence).zfill(2)            
            
            # calib data
            calibFileFullPath = str(strCalibDataDir + strSequence + '/calib_.txt')    
            calib=np.loadtxt(calibFileFullPath)
            Tr=np.array(calib[4,:].reshape(3,4),dtype=np.float32)
            
            # load poses
            poses = np.loadtxt(strGroundTruthPosesDir+strSequence+'.txt')  # ground truth
            strFileName = str(iFrameStep) + '_' + str(iKeyPt) + '-' + str(iDesc) + '_' + strSequence + '.txt'
#            strFileName = strSequence + '.txt'
            posePath = os.path.join(strEstimatedPosesDir, strFileName)
            poses_ = np.loadtxt(posePath)
            
            # get errors
            GroundTruthRels, EstimatedRels, errorRelEulers, errorRelTs = GetErrorRTs(poses, poses_, Tr, isPlot=0)            
            
            ## inlier rage and avarange iterations
            strFileName = 'Matchablity_' + str(iFrameStep) + '_' + str(iKeyPt) + '-' + str(iDesc) + '_' + strSequence + '.mat'
            matchablityPath = os.path.join(strBaseDir, strFileName)
            mat = io.loadmat(matchablityPath)
            proportions = mat['AllProportions'].T
            trailCounts = mat['AllTrialCounts'].T
            
            # append to current arrays
            CurAllErrorAngles = np.r_[CurAllErrorAngles, errorRelEulers]
            CurAllErrorTs = np.r_[CurAllErrorTs, errorRelTs]
            CurAllProportions = np.r_[CurAllProportions, proportions]
            CurAllTrailCounts = np.r_[CurAllTrailCounts, trailCounts]
            
        CurAllErrorAngles = np.delete(CurAllErrorAngles, 0, axis=0)
        CurAllErrorTs = np.delete(CurAllErrorTs, 0, axis=0)
        CurAllProportions = np.delete(CurAllProportions, 0, axis=0)
        CurAllTrailCounts = np.delete(CurAllTrailCounts, 0, axis=0)
        
        AllErrors_Eulers[iKeyPt][iDesc].append(CurAllErrorAngles)
        AllErrors_T[iKeyPt][iDesc].append(CurAllErrorTs)
        AllProportions[iKeyPt][iDesc].append(CurAllProportions)
        AllTrailCounts[iKeyPt][iDesc].append(CurAllTrailCounts)


EvaluationResults = np.zeros((NUM_METHODS*NUM_METHODS, 7), dtype=np.float32)
successIdx = np.zeros((NUM_METHODS*NUM_METHODS, AllErrors_Eulers[0][0][0].shape[0]), dtype=np.bool)

for iKeyPt in range(NUM_METHODS):
    for iDesc in range(NUM_METHODS):
        
        Errors_Eulers = np.squeeze(np.array(AllErrors_Eulers[iKeyPt][iDesc], dtype=np.float32))
        Errors_T = np.squeeze(np.array(AllErrors_T[iKeyPt][iDesc], dtype=np.float32))
        Proportions = np.squeeze(np.array(AllProportions[iKeyPt][iDesc], dtype=np.float32))
        TrailCounts = np.squeeze(np.array(AllTrailCounts[iKeyPt][iDesc], dtype=np.float32))
        
                    
        # RRE and RTE (relative rotation error and relative translation error)
        RREs = np.sum(np.abs(Errors_Eulers), axis=1)
        RTEs = LA.norm(Errors_T, axis=1)
        RRE = np.mean(RREs)
        RTE = np.mean(RTEs)
        stdRRE = np.std(RREs)
        stdRTE = np.std(RTEs)
        
        # registration success rate
        idx_RRE = (RREs < t_RRE)
        idx_RTE = (RTEs < t_RTE)
        finalIdx = idx_RRE * idx_RTE
        successIdx[iKeyPt*3+iDesc,:] = finalIdx
        cntSuccesses = np.sum(finalIdx)
        SuccessRate = cntSuccesses/RREs.shape[0]
        
        # inlier ratio
        InlierRatio = np.mean(Proportions)
        
        # average iterations
        AverageIterations = np.mean(TrailCounts)
        
        EvaluationResults[iKeyPt*3+iDesc,:] = [RRE, stdRRE, RTE, stdRTE, SuccessRate, InlierRatio, AverageIterations]
#        print(EvaluationResults[iKeyPt*3+iDesc,:])
        print(RRE, RTE)
        
#        # display results
#        print('RRE =', RRE)
#        print('RTE =', RTE)
#        print('SuccessRate =', SuccessRate)
#        print('InlierRatio =', InlierRatio)
#        print('AverageIterations =', AverageIterations)
        

io.savemat(os.path.join(strBaseDir, 'EvaluationResults.mat'), {'EvaluationResults':EvaluationResults})











