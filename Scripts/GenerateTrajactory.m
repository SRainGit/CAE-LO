% generate odometry by frame matching
clc; clear; close all;

addpath('./External');

BASE_DIR_E = '/media/rain/Win_E/';
BASE_DIR_F = '/media/rain/Win10_F/';
% BASE_DIR_E = 'E:';
% BASE_DIR_F = 'F:'; 

% dirs for normal use
addpath(fullfile(BASE_DIR_E, 'SLAM', 'Codes', 'KITTI', 'BlockBased', 'Scripts'));
KITTI_ROOT_DIR = fullfile(BASE_DIR_F, 'KITTI_odometry');
FEATNET_DIR = fullfile(BASE_DIR_F, 'KITTI_odometry', 'output_3DFeatNet');
USIP_DIR = fullfile(BASE_DIR_F, 'USIP');
USIP_KEYPT_DIR = fullfile(BASE_DIR_F, 'USIP', 'KeyPts', 'kitti', 'tsf_1024');
ISS_KEYPT_DIR = fullfile(BASE_DIR_F, 'USIP', 'KeyPts', 'kitti', 'iss_1024');
HARRIS_KEYPT_DIR = fullfile(BASE_DIR_F, 'USIP', 'KeyPts', 'kitti', 'harris_1024');
SIFT_KEYPT_DIR = fullfile(BASE_DIR_F, 'USIP', 'KeyPts', 'kitti', 'sift_1024');

% % dirs for demos
% KITTI_ROOT_DIR = fullfile(BASE_DIR_E, 'SLAM', 'Codes', 'KITTI', 'BlockBased', 'DemoData', 'KITTI_odometry');
% FEATNET_DIR = fullfile(KITTI_ROOT_DIR, 'output_3DFeatNet');
% USIP_DIR = fullfile(KITTI_ROOT_DIR, 'output_USIP');
% USIP_KEYPT_DIR = fullfile(USIP_DIR, 'KeyPts', 'kitti', 'tsf_1024', '');


m = 4;  % Dimensionality of raw data
DRAW_ALL_PUTATIVE = false;  % If true, will draw all inlier/outlier matches
MAX_MATCHES = 1000; % Maximum number of inlier+outlier matches to draw

iMethod = -1;  %0, CAE-LO; 1, 3DFeatNet; 2, USIP; 3, ISS; 4, Harris; 5; SIFT

FEATURE_DIM_1 = 32;
FEATURE_DIM_2 = 128;

% INLIERTHRESHOLD = 0.4;
INLIERTHRESHOLD = 1.0;


R90 = EulerAngle2RotateMat(-pi/2,0,-pi/2,'xyz');


% for iFrameStep = [1,2,5]
for iFrameStep = 1
for iSequence = 0:10
for iDesc = 2
for iKeyPt = 1:3

    cntMatches = 0;
    AllProportions = [];
    AllTrialCounts = [];        
    strSequence = sprintf('%02d', iSequence);    
    
    RAW_DATA_FOLDER = fullfile(KITTI_ROOT_DIR, 'velodyne', 'sequences', strSequence, 'velodyne');
    
    %% Dirs for features (including keypoints and descriptors)
    if iDesc == 0
        if iKeyPt == 0
            FEATURES_FOLDER = fullfile(KITTI_ROOT_DIR, 'velodyne', 'sequences', strSequence, 'Features');
        elseif iKeyPt == 1
            FEATURES_FOLDER = fullfile(KITTI_ROOT_DIR, 'velodyne', 'sequences', strSequence, 'Features-3DFeatNet');
        elseif iKeyPt == 2
            FEATURES_FOLDER = fullfile(KITTI_ROOT_DIR, 'velodyne', 'sequences', strSequence, 'Features-USIP');
        end
    elseif iDesc == 1
        if iKeyPt == 0
            FEATURES_FOLDER = fullfile(FEATNET_DIR, 'Descriptors_CAELO', strSequence);
        elseif iKeyPt == 1
            FEATURES_FOLDER = fullfile(FEATNET_DIR, 'Descriptors', strSequence);
        elseif iKeyPt == 2
            FEATURES_FOLDER = fullfile(FEATNET_DIR, 'Descriptors_USIP', strSequence);
        end
    elseif iDesc == 2
        if iKeyPt == 0
            FEATURES_FOLDER = fullfile(KITTI_ROOT_DIR, 'velodyne', 'sequences', strSequence, 'Features');
            DESC_FOLDER = fullfile(USIP_DIR, 'Descriptors-CAELO', strSequence);
        elseif iKeyPt == 1
            FEATURES_FOLDER = fullfile(FEATNET_DIR, 'Descriptors', strSequence);
            DESC_FOLDER = fullfile(USIP_DIR, 'Descriptors-3DFeatNet', strSequence);
        elseif iKeyPt == 2
            KEYPT_FOLDER = fullfile(USIP_KEYPT_DIR, strSequence);
            DESC_FOLDER = fullfile(USIP_DIR, 'Descriptors', strSequence);
        elseif iKeyPt == 3
            KEYPT_FOLDER = fullfile(ISS_KEYPT_DIR, strSequence);
            DESC_FOLDER = fullfile(USIP_DIR, 'Descriptors-ISS', strSequence);
        elseif iKeyPt == 4
            KEYPT_FOLDER = fullfile(HARRIS_KEYPT_DIR, strSequence);
            DESC_FOLDER = fullfile(USIP_DIR, 'Descriptors-Harris', strSequence);
        elseif iKeyPt == 5
            KEYPT_FOLDER = fullfile(SIFT_KEYPT_DIR, strSequence);
            DESC_FOLDER = fullfile(USIP_DIR, 'Descriptors-SIFT', strSequence);
        end
    end
    
    
    % path about poses and calib parameters
    strPoseFilesDir = fullfile(KITTI_ROOT_DIR, 'poses');
    strOutputPoseFilesDir = fullfile(KITTI_ROOT_DIR, 'poses_');
    calibFileFullPath = fullfile(KITTI_ROOT_DIR, 'calib',strSequence,'calib_.txt');

    % extract calib data
    calib = load(calibFileFullPath);
    Tr = reshape(calib(5,:),4,3)';
    R_Tr = Tr(:,1:3);
    R_Tr_inv = inv(R_Tr);
    T_Tr = reshape(Tr(:,4),3,1);

    % get file name list
    dirRawData = dir(RAW_DATA_FOLDER);
    keyFrames = {dirRawData(endsWith({dirRawData.name}, '.bin')).name};
    for iSet = 1 : length(keyFrames)
        keyFrames{iSet} = keyFrames{iSet}(1:end-4);
    end
    keyFrames = char(keyFrames);


    %% runs matching with RANSAC
    tic
    poses = zeros(size(keyFrames,1),12);
    poses(1,1) = 1; poses(1,6) = 1; poses(1,11) = 1;
    prevR = eye(3);
    prevT = zeros(3,1);
    for iFrame0 = 1 : iFrameStep : size(keyFrames, 1) - iFrameStep
%     for iFrame0 = 824 : iFrameStep : size(keyFrames, 1) - iFrameStep
        iFrame1 = iFrame0 + iFrameStep;
        tic
        frameName0 = keyFrames(iFrame0,:);
        frameName1 = keyFrames(iFrame1,:);        
        fprintf('Running on keypts%d, descs%d, frames %s: %s - %s\n', iKeyPt, iDesc, strSequence, frameName0, frameName1);
        
        % point cloud data
        cloud_fnames = {[fullfile(RAW_DATA_FOLDER, frameName0), '.bin'], ...
        [fullfile(RAW_DATA_FOLDER, frameName1), '.bin']};
        for i = 1 : 2
            pointcloud{i} = Utils.loadPointCloud(cloud_fnames{i}, m);
        end
        
        % keypoints and descriptors
        if iDesc == 0
            desc_fnames = {[fullfile(FEATURES_FOLDER, frameName0), '.bin.mat'], ...
                [fullfile(FEATURES_FOLDER, frameName1), '.bin.mat']}; 
            for i = 1 : 2
                featuresData = load(desc_fnames{i});
                result{i}.xyz = featuresData.KeyPts;
                result{i}.desc = featuresData.Features;
            end
        elseif iDesc == 1
            desc_fnames = {[fullfile(FEATURES_FOLDER, frameName0), '.bin'], ...
                [fullfile(FEATURES_FOLDER, frameName1), '.bin']};
            for i = 1 : 2
                xyz_features = Utils.load_descriptors(desc_fnames{i}, sum(3+FEATURE_DIM_1));
                result{i}.xyz = xyz_features(:, 1:3);
                result{i}.desc = xyz_features(:, 4:end);
            end
        elseif iDesc == 2
            % keypoints for method0
            if iKeyPt == 0                
                keypts_fnames = {[fullfile(FEATURES_FOLDER, frameName0), '.bin.mat'], ...
                    [fullfile(FEATURES_FOLDER, frameName1), '.bin.mat']}; 
                for i = 1 : 2
                    featuresData = load(keypts_fnames{i});
                    result{i}.xyz = featuresData.KeyPts;   
                end
                
            % keypoints for method1
            elseif iKeyPt == 1
                feature_fnames = {[fullfile(FEATURES_FOLDER, frameName0), '.bin'], ...
                    [fullfile(FEATURES_FOLDER, frameName1), '.bin']};
                for i = 1 : 2
                    xyz_features = Utils.load_descriptors(feature_fnames{i}, sum(3+FEATURE_DIM_1));
                    result{i}.xyz = xyz_features(:, 1:3);
                end
                
            % keypoints based on the code from method2
            elseif iKeyPt >= 2                
                keypt_fnames = {[fullfile(KEYPT_FOLDER, frameName0), '.bin'], ...
                    [fullfile(KEYPT_FOLDER, frameName1), '.bin']}; 
                for i = 1 : 2
                     Keypts = Utils.load_descriptors(keypt_fnames{i}, sum(3));
                     result{i}.xyz = (R90*Keypts')';
                end                
            end
            
            % correct pts by the 0.22 degree of intrinsic parameter
            if iKeyPt > 0
                for i = 1 : 2
                    result{i}.xyz = CorrectPts(result{i}.xyz, 0.22);
                end
            end
            
            % descriptors for all the methods
            desc_fnames = {[fullfile(DESC_FOLDER, frameName0), '.bin'], ...
                [fullfile(DESC_FOLDER, frameName1), '.bin']}; 
            for i = 1 : 2
                result{i}.desc = Utils.load_descriptors(desc_fnames{i}, sum(FEATURE_DIM_2));
            end
            
        end
        

        %% Match
        [~, matches12] = pdist2(result{2}.desc, result{1}.desc, 'euclidean', 'smallest', 1);
        matches12 = [1:length(matches12); matches12]';  


        cloud1_pts = result{1}.xyz(matches12(:,1), :);
        cloud2_pts = result{2}.xyz(matches12(:,2), :);% 

        cloud2_pts = (prevR * cloud2_pts' + prevT)';

        % RANSAC
        [estimateRt, inlierIdx, trialCount] = ransacfitRt([cloud1_pts'; cloud2_pts'], INLIERTHRESHOLD, false);
        Proportion = length(inlierIdx)/size(matches12, 1);        
        fprintf('Number of inliers: %i / %i (Proportion: %.3f. #RANSAC trials: %i)\n', ...
            length(inlierIdx), size(matches12, 1), ...
            Proportion, trialCount);
        
        cntMatches = cntMatches + 1;
        AllProportions(cntMatches) = Proportion;
        AllTrialCounts(cntMatches) = trialCount;


        % extract RT
        if size(estimateRt,1) >= 3
            updateR = estimateRt(:,1:3);
            updateT = estimateRt(:,4);
        else
            updateR = eye(3);
            updateT = zeros(3,1);
        end
        relativeR = updateR * prevR;
        relativeT = updateR * prevT + updateT;

        % update previous R and previous T
        prevR = relativeR;
        prevT = relativeT;

        % pose0
        pose0 = poses(iFrame0,:);
        R0 = [pose0(1:3); pose0(5:7); pose0(9:11)];
        R0_inv = inv(R0);
        T0 = reshape([pose0(4),pose0(8),pose0(12)],3,1);

        % get pose1
        R_poseDiff = R_Tr * (relativeR * R_Tr_inv);
        T_poseDiff = R_Tr * (-(relativeR * (R_Tr_inv * T_Tr)) + relativeT) + T_Tr;
        R = (R0 * R_poseDiff);
        T = (R0 * T_poseDiff) + T0;

        % add pose1 to poses
        RT = [R,T];
        pose1 = reshape(RT',1,12);
        poses(iFrame1,:) = pose1;


%         % Shows result
%         if size(inlierIdx,1) > 0
%             figure(iFrame0 * 2 - 1); clf
%             if DRAW_ALL_PUTATIVE
%                 Utils.pcshow_matches(pointcloud{1}, pointcloud{2}, ...
%                     result{1}.xyz, result{2}.xyz, ...
%                     matches12, 'inlierIdx', inlierIdx, 'k', MAX_MATCHES);
%             else
%                 Utils.pcshow_matches(pointcloud{1}, pointcloud{2}, ...
%                     result{1}.xyz, result{2}.xyz, ...
%                     matches12(inlierIdx, :), 'k', MAX_MATCHES);
%             end
%             title('Matches')
%             grid off, axis off;
%             view(-40,30);
%         end
%         
%         % Show alignment
%         figure(iFrame0 * 2); clf
%         Utils.pcshow_multiple(pointcloud, {eye(4), estimateRt});
%         title('Alignment')
%         grid off, axis off;
%         view(-40,30);
        
        toc
    end

    strPoseFullPath = fullfile(strOutputPoseFilesDir, [num2str(iFrameStep), '_', ...
        num2str(iKeyPt), '-', num2str(iDesc), '_', strSequence, '.txt']);
    save(strPoseFullPath, '-ascii', 'poses');

    strPropertionsFullPath = fullfile(KITTI_ROOT_DIR,['Matchablity_', num2str(iFrameStep), '_', ...
        num2str(iKeyPt), '-', num2str(iDesc), '_', strSequence, '.mat']);
    save(strPropertionsFullPath, 'AllProportions', 'AllTrialCounts');
end
end


end
end













