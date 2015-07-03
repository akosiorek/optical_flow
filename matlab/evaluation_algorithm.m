function [results_table] = evaluation_algorithm(path_to_data, aedat_file, timecode_file, disregarded_timesteps, data_Descriptor, filter_times, angles, do_plot, negate_GT_Y, withFloReference)
%EVALUATION_ALGORITHM Summary of this function goes here
%   Detailed explanation goes here
if nargin < 1
    path_to_data = '../data/quadrat/';
end
if nargin < 2
    aedat_file='quadrat_DVS_Only.aedat';
end
if nargin < 3
    timecode_file='quadrat_Frames_Timecode.txt';
end
if nargin < 4
    disregarded_timesteps =1; %=0;
    
end
if nargin < 5
    data_Descriptor = 'quadrat' ;
end
if nargin < 6
    time_start = 0;
    time_end = 0.7;
    time_resolution = 0.01;
else
    time_start = filter_times(1);
    time_end = filter_times(2);
    time_resolution = filter_times(3);
end
if nargin < 7
    angles = [0 45 90 135 180 225 270 315];
end
if nargin < 8
    do_plot = 0;
end

if nargin < 9
    negate_GT_Y = 0; %flag to negate the values of the y-flow from the .flo files (e.g. for pushbot)
end
if nargin < 10
    withFloReference = 1;
end
aedat_file = strcat(path_to_data,aedat_file);
timecode_file=strcat(path_to_data,timecode_file);
    strTimeInfo=strcat('_time',num2str(time_start),'-',...
        num2str(time_end),'_res',num2str(time_resolution));


% %Create logfile (Moved logfile handling to eval_setups.m)
% t = [datetime('now')];
% DateString = datestr(t);
% DateString=strrep(DateString,' ','_');
% logfile_name=strcat(path_to_data,data_Descriptor,'_logfile_', DateString, '.txt');
% diary(logfile_name);

% create alternative logfile in table form


%% Set retina size for the camera
retinaSize=[240 180];



%% read out events from .aedat file and save them to a .tsv file
eventsfile_tsv_name= strcat(path_to_data,data_Descriptor,'_events.tsv');

convert_aedat_to_tsv(aedat_file,eventsfile_tsv_name);


%% Read events from the saved .tsv file and calculate optical flow
% time_end=timecode(2,2);
% time_start=timecode(2,2)-0.0001*1e6;
% total_time=time_end-time_start;
% angles = [0 30 60 90 120 150 180 210 240 270 300 330]


%quantize events according to timecode file
% events = read_events(eventsfile_tsv_name, 1);
% quantized = timecode_quantizer(events, timecode, retinaSize);

%for logfile output
eventsfile_tsv_name
data_Descriptor
retinaSize
time_start
time_end
time_resolution
angles
disregarded_timesteps
negate_GT_Y
withFloReference
% optical_flow(eventsfile_tsv_name, retinaSize, [time_start time_end], time_resolution, angles);
load('flow.mat');
opticalFlowX_aedat = opticalFlowX;
opticalFlowY_aedat = opticalFlowY;
% quantized = allEvents %if "AllEvents" has been used
quantized_timestamps=timestamps;




%% Read the ground truth flow from the .flo files
if (withFloReference)
    timecode= parse_timecode(timecode_file, disregarded_timesteps);
    nr_flo_files = size(timecode,1)-1;
    start_scene_offset = timecode(1,1);
    opticalFlowX_flo=zeros(retinaSize(1),retinaSize(2), nr_flo_files);
    opticalFlowY_flo=zeros(retinaSize(1),retinaSize(2), nr_flo_files);
    
    for i=1:nr_flo_files
        scene_index=i+start_scene_offset;
        if (strcmp(data_Descriptor, 'baelle'))
            %different string names for 'baelle':
            if (scene_index<10)
                flo_name=strcat('gtFlo000',num2str(scene_index),'.flo');
            else
                flo_name=strcat('gtFlo00',num2str(scene_index),'.flo');
            end
            
        else
            if (scene_index<10)
                flo_name=strcat('scene0000',num2str(scene_index),'_mdpof.flo');
            else
                flo_name=strcat('scene000',num2str(scene_index),'_mdpof.flo');
            end
            
        end
        
        
        full_path=strcat(path_to_data,flo_name);
        flo_matrix = readFlowFile(full_path);
        opticalFlowX_flo(:,:,i)=flipud(flo_matrix(:,:,1))'; %transpose might be wrong
        %fix wrong sign for flowY values
        if (negate_GT_Y)
            opticalFlowY_flo(:,:,i)=flipud(-1.*flo_matrix(:,:,2))';
        else
            opticalFlowY_flo(:,:,i)=flipud(flo_matrix(:,:,2))';
        end
    end
    
end



%% Obtain mask containing the event locations

%The opticalFlow output only saves slices for which the full amount of
%previous slices were convoluted by the tempral filter. We also remove
%these previous slices from the quantized array
quantizedOffset=size(quantized,1)-size(opticalFlowX_aedat,3);
mask_eventLocations = createMaskFromQuantized(quantized, quantizedOffset);
quantized_timestamps=quantized_timestamps(quantizedOffset+1:end);

% % check for mask correctness
% mask_eventLocations= permute(mask_eventLocations,[2 1 3]);
% visualize_matrix3d(mask_eventLocations,0.5);



%%  Map each quantized timeslice to a frame in the ground truth and calculate angular error
if (withFloReference)
    indexQuantizedToGT = zeros(size(quantized_timestamps,1),2);
    
    for i=1:size(indexQuantizedToGT)
        compareIndex=1;
        diff = 1e6;
        for j=1:size(timecode)-1
            newdiff =(abs(quantized_timestamps(i)-timecode(j,2)));
            if (newdiff<diff)
                compareIndex =j;
                diff=newdiff;
            end
        end
        indexQuantizedToGT(i,1)=compareIndex;
        indexQuantizedToGT(i,2)=diff;
    end
    
    % Calculate the angular error for all '1'-events saved in quantized
    angularErrors= calc_angular_errors(opticalFlowX_aedat, opticalFlowY_aedat ...
        , opticalFlowX_flo, opticalFlowY_flo, indexQuantizedToGT);
    disp(['Angular Errors: ']);
    [RMSE1, AbsMean1, Median1, avgRMSE1, avgAbsMean1, avgMedian1] = eval_angular_errors(angularErrors, mask_eventLocations);
    
end





%% Interpolate the Ground Truth to get exact temporal matches for comparison
if (withFloReference)
    [intp_flowX_GT, intp_flowY_GT] = interpolate_ground_truth(opticalFlowX_aedat, ...
        opticalFlowY_aedat, opticalFlowX_flo, opticalFlowY_flo, quantized_timestamps, ...
        timecode);
    
    
    % Calculate angular errors
    angularErrors2 = calc_angular_errors(opticalFlowX_aedat, opticalFlowY_aedat, ...
        intp_flowX_GT, intp_flowY_GT);
    disp(['Angular Errors for interpolated ground truth slices: ']);
    [RMSE2, AbsMean2, Median2, avgRMSE2, avgAbsMean2, avgMedian2] = eval_angular_errors(angularErrors2, mask_eventLocations);
    
end


%% create table with all relevant results
nrOfEntries = size(opticalFlowX_aedat,3);
%duplicate scalar results to have the same column number as result vectors
if (~withFloReference)
    avgRMSE1=NaN;
    avgRMSE2=NaN;
    avgAbsMean1=NaN;
    avgAbsMean2=NaN;
    avgMedian1=NaN;
    avgMedian2=NaN;
    RMSE1=NaN;
    AbsMean1=NaN;
    Median1=NaN;
    RMSE2=NaN;
    AbsMean2=NaN;
    Median2=NaN;
    indexQuantizedToGT=[NaN NaN];
    
    RMSE1=repmat(RMSE1, nrOfEntries,1);
    RMSE2=repmat(RMSE2, nrOfEntries,1);
    AbsMean1=repmat(AbsMean1, nrOfEntries,1);
    AbsMean2=repmat(AbsMean2, nrOfEntries,1);
    Median1=repmat(Median1, nrOfEntries,1);
    Median2=repmat(Median2, nrOfEntries,1);
    indexQuantizedToGT=repmat(indexQuantizedToGT, nrOfEntries,1);
    
    
end
avgRMSE1=repmat(avgRMSE1, nrOfEntries,1);
avgRMSE2=repmat(avgRMSE2, nrOfEntries,1);
avgAbsMean1=repmat(avgAbsMean1, nrOfEntries,1);
avgAbsMean2=repmat(avgAbsMean2, nrOfEntries,1);
avgMedian1=repmat(avgMedian1, nrOfEntries,1);
avgMedian2=repmat(avgMedian2, nrOfEntries,1);
t_data_Descriptor = {data_Descriptor};
t_data_Descriptor = repmat(t_data_Descriptor, nrOfEntries,1);
retinaSizeX = repmat(retinaSize(:,1), nrOfEntries,1);
retinaSizeY = repmat(retinaSize(:,1), nrOfEntries,1);
time_start = repmat(time_start, nrOfEntries,1);
time_end = repmat(time_end, nrOfEntries,1);
time_resolution = repmat(time_resolution, nrOfEntries,1);
angles = {mat2str(angles)};
angles = repmat(angles, nrOfEntries,1);
sliceNr=(1:nrOfEntries)';

[avgAngle, avgSpeed] = calc_avgAngle_avgSpeed(opticalFlowX_aedat, opticalFlowY_aedat, mask_eventLocations);
% meanAngle=repmat(meanAngle, nrOfEntries,1);
% meanSpeed=repmat(meanSpeed, nrOfEntries,1);


results_table = table(t_data_Descriptor,retinaSizeX, retinaSizeY,time_start,time_end,time_resolution,angles,sliceNr,...
    quantized_timestamps, avgAngle, avgSpeed, RMSE1, AbsMean1, Median1, indexQuantizedToGT(:,2), RMSE2, AbsMean2, Median2, ...
    avgRMSE1, avgAbsMean1, avgMedian1, avgRMSE2, avgAbsMean2, avgMedian2, ...
    'VariableNames',{'Descriptor' 'retinaSizeX' 'retinaSizeY' 'time_start' 'time_end' 'time_resolution' 'angles' ...
    'Slice_Nr' 'Quantized_time' 'avgAngle' 'avgSpeed' 'RMSE_direct' 'AbsMean_direct' 'Median_direct' 'timeDiff_direct' ...
    'RMSE_intp' 'AbsMean_intp' 'Median_intp' ...
    'avgRMSE_direct' 'avgAbsMean_direct' 'avgMedian_direct' ...
    'avgRMSE_intp' 'avgAbsMean_intp' 'avgMedian_intp'});

%% Visualize results in quiver plot and save logfile



if (do_plot)
    maskedFlowx=apply_mask(opticalFlowX_aedat,mask_eventLocations);
    maskedFlowy=apply_mask(opticalFlowY_aedat,mask_eventLocations);
    

    make_quiver_movie(strcat(data_Descriptor,strTimeInfo),opticalFlowX_aedat, opticalFlowY_aedat, quantized_timestamps);
    make_quiver_movie(strcat(data_Descriptor,strTimeInfo,'_masked'),maskedFlowx, maskedFlowy, quantized_timestamps);
    
    if (withFloReference)
        % Create a masked version of the ground truth flow frames for qualitative
        % comparison
        maskedFlowX_GT=zeros(size(opticalFlowX_aedat));
        maskedFlowY_GT=zeros(size(opticalFlowY_aedat));
        for i=1:size(opticalFlowX_aedat,3)
            maskedFlowX_GT(:,:,i)=opticalFlowX_flo(:,:,indexQuantizedToGT(i,1));
            maskedFlowY_GT(:,:,i)=opticalFlowY_flo(:,:,indexQuantizedToGT(i,1));
        end
        maskedFlowX_GT=apply_mask(maskedFlowX_GT,mask_eventLocations);
        maskedFlowY_GT=apply_mask(maskedFlowY_GT,mask_eventLocations);
        
        % make_quiver_movie(strcat(data_Descriptor,'_GT'),opticalFlowX_flo,opticalFlowY_flo);
        % make_quiver_movie(strcat(data_Descriptor,'_GT_interp'),intp_flowX_GT,intp_flowY_GT);
%         make_quiver_movie(strcat(data_Descriptor,'_GT_masked'),maskedFlowX_GT,maskedFlowY_GT, quantized_timestamps);
    end
    
    
    
    
end



% diary off;


end







function [interp_flowX_GT, interp_flowY_GT]= interpolate_ground_truth(flowX, flowY, flowX_GT, flowY_GT, ...
    quantized_timestamps, timecode)
interp_flowX_GT=zeros(size(flowX));
interp_flowY_GT=zeros(size(flowX));

interpolationPoints = zeros(size(flowX,3),5);
%entries: (indexBefore, indexAfter, timeBefore, timeAfter, timeQuantized)
for i=1:size(interp_flowX_GT,3)
    indexBefore=1;
    indexAfter=0;
    diffBefore=1e7;
    diffAfter=1e7;
    for j=1:size(timecode)-1
        current_diffBefore = (quantized_timestamps(i)-timecode(j,2));
        if(current_diffBefore > 0 && current_diffBefore < diffBefore)
            indexBefore=j;
            diffBefore=current_diffBefore;
        end
        current_diffAfter = (timecode(j,2)-quantized_timestamps(i));
        if(current_diffAfter > 0 && current_diffAfter < diffAfter)
            indexAfter=j;
            diffAfter=current_diffAfter;
        end
        
    end
    %Hanlde situation where quantized slices exceed the ground truth time
    if (indexAfter==0)
        indexBefore=size(timecode,1)-2;
        indexAfter=size(timecode,1)-1;
    end
    
    interpolationPoints(i,:)=[indexBefore, indexAfter, timecode(indexBefore,2), ...
        timecode(indexAfter,2), quantized_timestamps(i)];
end

for i=1:size(interp_flowX_GT,3)
    %interpolate between two closest ground truth slices
    index1=interpolationPoints(i,1);
    index2=interpolationPoints(i,2);
    x= cat(3,flowX_GT(:,:,index1),flowX_GT(:,:,index2));
    x=permute(x, [3 1 2]);
    t0 = [interpolationPoints(i,3), interpolationPoints(i,4)];
    x_interp=interp1(t0,x,interpolationPoints(i,5));
    
    y= cat(3,flowY_GT(:,:,index1),flowY_GT(:,:,index2));
    y=permute(y, [3 1 2]);
    t0 = [interpolationPoints(i,3), interpolationPoints(i,4)];
    y_interp=interp1(t0,y,interpolationPoints(i,5));
    
    interp_flowX_GT(:,:,i)=x_interp;
    interp_flowY_GT(:,:,i)=y_interp;
    
end

end








% function [matrix] = delete_defect_pixels(matrix, xCoords, yCoords)
% % matrix(xCoords,yCoords,:)=0;
% % indeces = [xCoords; yCoords];
% for i=1:size(matrix,3)
%     slice = matrix(:,:,i);
%     slice(sub2ind(size(slice),xCoords,yCoords))=0;
%     matrix(:,:,i)=slice;
% end
% end





function [RMSE, AbsMean, Median, avgRSME, avgAbsMean, avgMedian] = eval_angular_errors(angularErrors, set_to_nan_mask)
if nargin >1
    angularErrors(set_to_nan_mask==0)=NaN;
end
avg_sqrtMeanAngularError=0;
c1=0;
avg_meanAngularErrorAbs=0;
c2=0;
avg_medianAngularError=0;
c3=0;
RMSE = [];
AbsMean = [];
Median = [];
for i=1:size(angularErrors,3)
    anglesSlice=angularErrors(:,:,i);
    %     meanAngularError=nanmean(anglesSlice(:));
    %     medianAngularError=nanmedian(anglesSlice(:));
    %     disp(['Mean angular error in slice nr ', num2str(i), ': ', num2str(meanAngularError)])
    %     disp(['Median angular error in slice nr ', num2str(i), ': ', num2str(medianAngularError)]);
    
    tmp = anglesSlice.^2;
    sqrtMeanAngularError=sqrt(nanmean(tmp(:)));
    meanAngularErrorAbs=nanmean(abs(anglesSlice(:)));
    medianAngularError=nanmedian(anglesSlice(:));
    disp(['Slice nr ', num2str(i), ' - RMSE angular error: ',char(9), num2str(sqrtMeanAngularError)])
    disp(['Slice nr ', num2str(i), ' - Abs-Mean angular error: ',char(9), num2str(meanAngularErrorAbs)])
    disp(['Slice nr ', num2str(i), ' - Median angular error : ',char(9), num2str(medianAngularError)]);
    RMSE = [RMSE; sqrtMeanAngularError];
    AbsMean = [AbsMean; sqrtMeanAngularError];
    Median = [Median; sqrtMeanAngularError];
    
    
    
    if (~isnan(sqrtMeanAngularError))
        avg_sqrtMeanAngularError=avg_sqrtMeanAngularError+sqrtMeanAngularError;
        c1=c1+1;
    end
    if (~isnan(meanAngularErrorAbs))
        avg_meanAngularErrorAbs=avg_meanAngularErrorAbs+meanAngularErrorAbs;
        c2=c2+1;
    end
    if (~isnan(medianAngularError))
        avg_medianAngularError=avg_medianAngularError+medianAngularError;
        c3=c3+1;
    end
    
    
    
end
avg_sqrtMeanAngularError=avg_sqrtMeanAngularError/c1;
avg_meanAngularErrorAbs=avg_meanAngularErrorAbs/c2;
avg_medianAngularError=avg_medianAngularError/c3;
disp(['Mean for all sclices - RMSE angular error: ',char(9), num2str(avg_sqrtMeanAngularError)])
disp(['Mean for all slices - Abs-Mean angular error: ',char(9), num2str(avg_meanAngularErrorAbs)])
disp(['Mean for all slices - Median angular error : ',char(9), num2str(avg_medianAngularError)]);

avgRSME=avg_sqrtMeanAngularError;
avgAbsMean=avg_meanAngularErrorAbs;
avgMedian = avg_medianAngularError;

disp(['----']);


end




function [angularErrors] = calc_angular_errors(flowX, flowY, flowX_ref, flowY_ref, index_matcher)
angularErrors = zeros(size(flowX));
vectorLengths = zeros(size(flowX));
vectorLengths_ref = zeros(size(flowX_ref));

for i=1:size(flowX_ref,3)
    vectorLengths_ref(:,:,i) = sqrt( ...
        flowX_ref(:,:,i).*flowX_ref(:,:,i)+ ...
        flowY_ref(:,:,i).*flowY_ref(:,:,i));
end

for i=1:size(flowX,3)
    vectorLengths(:,:,i) = sqrt( ...
        flowX(:,:,i).*flowX(:,:,i)+ ...
        flowY(:,:,i).*flowY(:,:,i));
end

for i=1:size(angularErrors,3)
    if nargin > 4
        %compare with closest available ground truth frame, if an index_matcher
        %variable is given
        compareIndex=index_matcher(i);
    else
        compareIndex=i;
    end
    
    angularErrors(:,:,i) = 180/pi * acos( ...
        (flowX(:,:,i)./vectorLengths(:,:,i)).* ...
        (flowX_ref(:,:,compareIndex)./vectorLengths_ref(:,:,compareIndex)) + ...
        (flowY(:,:,i)./vectorLengths(:,:,i)).* ...
        (flowY_ref(:,:,compareIndex)./vectorLengths_ref(:,:,compareIndex)));
    slice=angularErrors(:,:,i);
end

end


function [avgSpeed, avgAngle] = calc_avgAngle_avgSpeed(flowX, flowY, set_to_nan_mask)
angles = zeros(size(flowX,1),size(flowX,2));
vectorLengths = zeros(size(angles));
avgSpeed=zeros(size(flowX,3),1);
avgAngle=zeros(size(flowX,3),1);

% avgSpeed=nanmean(abs(angles(:)));
% avgAngle=nnanmean(abs(vectorLengths(:)));

for i=1:size(flowX,3)
    
    vectorLengths = sqrt( ...
        flowX(:,:,i).*flowX(:,:,i)+ ...
        flowY(:,:,i).*flowY(:,:,i));
    angles = 180/pi * mod(atan2(flowY(:,:,i),flowX(:,:,i)),2*pi);
    %     angles(angles<0)=angles+360;
    angles(set_to_nan_mask(:,:,i)==0)=NaN;
    vectorLengths(set_to_nan_mask(:,:,i)==0)=NaN;
    avgSpeed(i)=nanmean(abs(vectorLengths(:)));
    avgAngle(i)=nanmean(abs(angles(:)));
    
end

%
%
% for i=1:size(angles,3)
%
% end


end


function [maskedArray] = apply_mask(Array, mask)
maskedArray=Array;
for i=1:size(Array,3)
    maskedArray(mask==0)=0;
end

end

function [mask] = createMaskFromQuantized(quantized, quantizedOffset)
%convert quantized array to full matrix

for j=1:(size(quantized,2))
    quantized_relevant=quantized(quantizedOffset+1:size(quantized,1));
    mask=zeros(size(quantized_relevant{1}));
    for i=1:size(quantized_relevant,1)
        mask(:,:,i)=full(quantized_relevant{i});
        debugSlice=mask(:,:,i);
    end
end

end






function [timecode] = parse_timecode(timecode_file, disregarded_timesteps)
if nargin < 1
    timecode_file='../data/pushbot-timecode.txt';
end

[frame_nr time] = textread(timecode_file, '%f %f', ...
    'headerlines', 3+disregarded_timesteps) ;
% [frame_nr time] = load(timecode_file);
timecode = [frame_nr time];
condition_not_empty=timecode(:,1)==NaN;
timecode(condition_not_empty,:)=[];
% if disregarded_timesteps > 0
%     timecode(1:disregarded_timesteps,:)=[];
% end

end

function [tsv_matrix, retinaSize] = events_from_aedat(aedat_location)
if nargin < 1
    aedat_location='../data/pushbot_DVS.aedat';
end
% [allAddr,allTs]=loadaerdat('../data/pushbot_DVS.aedat');
[x,y,pol,ts]=getDVSeventsDavis(aedat_location);
%write to text file

%comparable .tsv files use indexing 0-127, aedat reader gives 1-128
x=x-1;
y=y-1;
id = (1:size(x))';


tsv_matrix=[ts x y pol id];

end

function save_aedat_events(tsv_matrix,name_of_saved_tsv)
if nargin < 1
    name_of_saved_tsv='aedat_events.tsv';
end
dlmwrite(name_of_saved_tsv,tsv_matrix,'\t');
end



function [ events ] = read_events( file, response )
%READ_EVENTS Reads events from TSV
%   Parses a tsv event file. If an extended (3D) event-stream is used, the
%   additional 3d data is removed. A 3-dim matrix of x,y,t will be
%   returned.

% Get raw events
events_raw = dlmread(file,'\t');

% Reduce if extended file_size
if(size(events_raw,2) == 7)
    if(response)
        events = events_raw(:,4:7);
        response = events(:, end);
        response(response == 0) = -1;
        events(:, end) = response;
    else
        events = events_raw(:,4:6);
    end
else
    events = events_raw;
end

end
