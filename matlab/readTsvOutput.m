function [ flowX, flowY ] = readTsvOutput( )
%readTsvOutput Summary of this function goes here
%   Detailed explanation goes here
path = '../data/event_data/eventsmedium/'
nrSlices = 294;


% events_raw = dlmread(file,'\t');
flowX=zeros(128,128,nrSlices);
flowY=zeros(128,128,nrSlices);

for i=1:nrSlices
    if (i<10)
            filepathX= strcat(path,'0000000',num2str(i),'_xv.tsv');
                        filepathY= strcat(path,'0000000',num2str(i),'_yv.tsv');

    elseif(i<100)
                    filepathX= strcat(path,'000000',num2str(i),'_xv.tsv');
                                filepathY= strcat(path,'000000',num2str(i),'_yv.tsv');


    elseif(i<1000)
                    filepathX= strcat(path,'00000',num2str(i),'_xv.tsv');
                                filepathY= strcat(path,'00000',num2str(i),'_yv.tsv');


    end
    flowX(:,:,i)=dlmread(filepathX,'\t');
        flowY(:,:,i)=dlmread(filepathY,'\t');

end
firstSlice = flowX(:,:,1);

%compare to matlab results
eventsfile_tsv_name = '../data//event_data/events_medium_reformat.tsv';
retinaSize = [128 128];
time_start = 0;
time_end = 0.7;
time_resolution = 0.01;
angles = [0 45 90 135];
% optical_flow(eventsfile_tsv_name, retinaSize, [time_start time_end], time_resolution, angles);
load('flow.mat');
opticalFlowX;
opticalFlowY;

firstSliceMatlab=opticalFlowX(:,:,1);

% figure('units','normalized','outerposition',[0 0 1 1]);

% [x,y] = meshgrid(1:1:size(opticalFlowX,1),1:1:size(opticalFlowX,2));
% figure(1)
% quiver(x,y,flowX(:,:,50)',flowY(:,:,50)');
%     xlabel('x');
%     ylabel('y');
%     xlim([-5 size(opticalFlowX,1)+5]);
%     ylim([-5 size(opticalFlowY,2)+5]);
%     %         view([0 90]);
%     grid on
%     drawnow
% figure(2)
% quiver(x,y,opticalFlowX(:,:,50)',opticalFlowY(:,:,50)');
% 
%     xlabel('x');
%     ylabel('y');
%     xlim([-5 size(opticalFlowX,1)+5]);
%     ylim([-5 size(opticalFlowY,2)+5]);
%     %         view([0 90]);
%     grid on
%     drawnow
    
    diffX = opticalFlowX-flowX;
    diffY = opticalFlowY-flowY;
    meanX=mean(diffX(:));
    meanY=mean(diffY(:));
    meanFlowX=mean(opticalFlowX(:));
    meanFlowY=mean(opticalFlowY(:));
    meanFlowXcpp=mean(flowX(:));
    meanFlowYcpp=mean(flowY(:));

    a=5;

end

