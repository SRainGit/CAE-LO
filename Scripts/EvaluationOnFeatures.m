
% Evauate the accuracy of the matching algorithm

clc; clear; close all;

% strBaseDir = '/media/rain/Win10_F/KITTI_odometry/';
strBaseDir = 'F:\KITTI_odometry\';

nDiscretizations = 8;
AllProportions = [];
figure,
h = [50 100 1900 300];
set(gcf,'Position',h)

% frameSteps = [1, 5, 10];
frameSteps = 1;
for iStep = 1:length(frameSteps)
    iFrameStep = frameSteps(iStep);
    for iSequence = 0:10
        for iDataSource = 0:2
            fileName = [strBaseDir, 'Matchablity_', num2str(iFrameStep), '_', num2str(iDataSource), '-', num2str(iDataSource), '_',  num2str(iSequence,'%02d'), '.mat'];
            data = load(fileName);
            AllProportions{iSequence+1,iDataSource+1} = data.AllProportions;
        end
        
        y = squeeze(AllProportions(iSequence+1,1,:));
        axis = subplot(length(frameSteps), 11, (iStep-1)*11+iSequence+1); hold on;
        
        title(['KITTI-',num2str(iSequence,'%02d')])
        
%         boxplot([squeeze(AllProportions{iSequence+1,1}); squeeze(AllProportions{iSequence+1,2})])
        boxplot([squeeze(AllProportions{iSequence+1,1})',...
            squeeze(AllProportions{iSequence+1,2})',squeeze(AllProportions{iSequence+1,3})'])
    end
end

