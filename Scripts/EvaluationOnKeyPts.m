
% Evauate the accuracy of the detection of interest point

clc; clear; close all;

% strBaseDir = '/media/rain/Win10_F/KITTI_odometry/';
strBaseDir = 'F:\KITTI_odometry\';

fileNameTitile = 'AccuracyOfKeyPts_';
% fileNameTitile = 'InnerAccuracyOfKeyPts_';

nDiscretizations = 8;
counts = zeros(11,2,nDiscretizations);
KeyPtsRatios = zeros(11,2,nDiscretizations);
figure,
h = [50 100 1900 200];
set(gcf,'Position',h)

% frameSteps = [1, 2, 5, 10];
% frameSteps = [1, 5, 10];
frameSteps = 1;
for iStep = 1:length(frameSteps)
    iFrameStep = frameSteps(iStep);
    for iSequence = 0:10
        for iDataSource = 0:2  % 0, ours; 1, 3DFeatNet; 2, USIP
            fileName = [strBaseDir, fileNameTitile, num2str(iFrameStep), '_', num2str(iDataSource), '_', num2str(iSequence,'%02d'), '.mat'];
            data = load(fileName);

            counts(iSequence+1,iDataSource+1,:) = data.counts;
            for iX = 1:nDiscretizations
                KeyPtsRatios(iSequence+1,iDataSource+1,iX) = ...
                    sum(counts(iSequence+1,iDataSource+1,iX))/sum(counts(iSequence+1,iDataSource+1,:));
            end
        end
        
        y = squeeze(counts(iSequence+1,1,:));
        axis = subplot(length(frameSteps), 11, (iStep-1)*11+iSequence+1); hold on;
        title(['KITTI-',num2str(iSequence,'%02d')])
        
        x_=0:9;
        y_=[0; y; 0];
        h=fill(x_,y_,'r');
        set(h,'edgealpha',0,'facealpha',0.5)
        plot(y(1:3),'r-o','linewidth',2);
        
        y = squeeze(counts(iSequence+1,2,:));
        x_=0:9;
        y_=[0; y; 0];
        h=fill(x_,y_,'g');    
        set(h,'edgealpha',1,'facealpha',0.5)
        plot(y(1:3),'g-o','linewidth',2);

        
        y = squeeze(counts(iSequence+1,3,:));
        x_=0:9;
        y_=[0; y; 0];
        h=fill(x_,y_,'b');    
        set(h,'edgealpha',1,'facealpha',0.5)
        plot(y(1:3),'b-o','linewidth',2);

    %     axis = subplot(1, 11, 0+iSequence+1); hold on;
    %     
    %     x1=0:9;
    %     y1=[0;squeeze(KeyPtsRatios(iSequence+1,1,:));0];
    %     h=fill(x1,y1,'r');
    %     set(h,'edgealpha',0,'facealpha',0.5)
    %     title(num2str(iSequence,'%02d'))
    %     xticks('manual')
    % 
    %     x2=0:9;
    %     y2=[0;squeeze(KeyPtsRatios(iSequence+1,2,:));0];
    %     h=fill(x2,y2,'b');    
    %     set(h,'edgealpha',1,'facealpha',0.5)
    %     title(num2str(iSequence,'%02d'))
    end
end

counts0 = squeeze(counts(:,1,:));
counts1 = squeeze(counts(:,2,:));
counts2 = squeeze(counts(:,3,:));
