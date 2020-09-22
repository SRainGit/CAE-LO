
% Comparison on the trajectories

clc; clear; close all;

strBaseDir = '/media/rain/Win10_F/KITTI_odometry/';
% strBaseDir = 'F:\KITTI_odometry\';

fig=figure;
h = [50 100 1600 900];
set(gcf,'Position',h)
rotate3d on

colors = ['r','g','b','c'];

% frameSteps = [1, 5, 10];
frameSteps = 1;
for iStep = 1:length(frameSteps)
    iFrameStep = frameSteps(iStep);
    for iSequence = 0:10
        strSequence = num2str(iSequence,'%02d');
        
%         subplot(3,4,iSequence+1);
%         cla(fig)
        hold on, axis equal;
                
        fileName_GT = [strBaseDir, 'poses/', strSequence, '.txt'];
        data_GT = load(fileName_GT);
        trajectory_GT = [data_GT(:,4),data_GT(:,8),data_GT(:,12)];
        plot3(trajectory_GT(:,1),trajectory_GT(:,2),trajectory_GT(:,3), 'k');
        
        for iDesc = 2
            for iKeyPt = 3:5
                fileName = [strBaseDir, 'poses_/', num2str(iFrameStep), '_', ...
                    num2str(iKeyPt), '-', num2str(iDesc), '_', strSequence, '.txt'];
                data = load(fileName);
                trajectory = [data(:,4),data(:,8),data(:,12)];
                if (iDesc + iKeyPt) == 0
                    plot3(trajectory(:,1),trajectory(:,2),trajectory(:,3),'r');
                else
                    plot3(trajectory(:,1),trajectory(:,2),trajectory(:,3));
                end
            end
        end
        
        title(['KITTI-',strSequence])
        xlabel('X [m]')
        zlabel('Y [m]')
        view(0,0)    
    end
end

