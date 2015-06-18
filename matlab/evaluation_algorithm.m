function evaluation_algorithm(path_to_data, aedat_file, timecode_file, disregarded_timesteps, data_Descriptor)
%EVALUATION_ALGORITHM Summary of this function goes here
%   Detailed explanation goes here
if nargin < 1
   path_to_data = '../data/quadrat/'; 
end
if nargin < 2
    %    aedat_file='../data/pushbot_DVS.aedat';
    aedat_file='quadrat_DVS_Only.aedat';
    
end
if nargin < 3
    %     timecode_file='../data/pushbot-timecode.txt';
    timecode_file='quadrat_Frames_Timecode.txt';
    
end
if nargin < 4
    %     disregarded_timesteps =0
    disregarded_timesteps =1;
    
end
if nargin < 5
   data_Descriptor = 'quadrat' ;
end
aedat_file = strcat(path_to_data,aedat_file);
timecode_file=strcat(path_to_data,timecode_file);

%Create logfile
t = [datetime('now')];
DateString = datestr(t);
DateString=strrep(DateString,' ','_');
logfile_name=strcat(path_to_data,data_Descriptor,'_logfile_', DateString, '.txt');
diary(logfile_name);


%% read out events from .aedat file and save them to a .tsv file
[tsv_matrix, retinaSize] = events_from_aedat(aedat_file);
eventsfile_tsv_name='../data/eval_events.tsv'
save_aedat_events(tsv_matrix,eventsfile_tsv_name)


%% Read events from the saved .tsv file and calculate optical flow
% time_end=timecode(2,2);
% time_start=timecode(2,2)-0.0001*1e6;
time_start=0;
time_end=0.5
time_resolution=0.01;
total_time=time_end-time_start;
% time_resolution=total_time/size(timecode,1);
%match the upscaling of the resolution in quantizer
% time_resolution = time_resolution*1e-6;

%quantize events according to timecode file
events = read_events(eventsfile_tsv_name, 1);
% quantized = timecode_quantizer(events, timecode, retinaSize);

%% run the optical_flow code
% optical_flow(eventsfile_tsv_name, retinaSize, [time_start time_end], time_resolution);
load('flow.mat');
opticalFlowX_aedat = opticalFlowX;
opticalFlowY_aedat = opticalFlowY;
quantized_timestamps=timestamps;
%transpose the slices of both arrays
% opticalFlowX_aedat = permute(opticalFlowX_aedat,[2 1 3]);
% opticalFlowY_aedat = permute(opticalFlowY_aedat,[2 1 3]);
% make_quiver_movie('square_quivers.avi',opticalFlowX_aedat, opticalFlowY_aedat);



%% Read the ground truth flow from the .flo files
timecode= parse_timecode(timecode_file, disregarded_timesteps);
nr_flo_files = size(timecode,1)-1;
start_scene_offset = timecode(1,1);
opticalFlowX_flo=zeros(retinaSize(1),retinaSize(2), nr_flo_files);
opticalFlowY_flo=zeros(retinaSize(1),retinaSize(2), nr_flo_files);

for i=1:nr_flo_files
    scene_index=i+start_scene_offset;
    if (scene_index<10)
        flo_name=strcat('scene0000',num2str(scene_index),'_mdpof.flo');
    else
        flo_name=strcat('scene000',num2str(scene_index),'_mdpof.flo');
    end
    full_path=strcat(path_to_data,flo_name);
    flo_matrix = readFlowFile(full_path);
    opticalFlowX_flo(:,:,i)=flipud(flo_matrix(:,:,1))'; %transpose might be wrong
    opticalFlowY_flo(:,:,i)=flipud(flo_matrix(:,:,2))';
end
% visualize_matrix3d(opticalFlowX_flo,1);
% visualize_matrix3d(opticalFlowY_flo,1);
% make_quiver_movie('ground_truth_quivers.avi',opticalFlowX_flo,opticalFlowY_flo);

%% Obtain mask containing th event locations

%The opticalFlow output only saves slices for which the full amount of
%previous slices were convoluted by the tempral filter. We also remove
%these previous slices from the quantized array
quantizedOffset=size(quantized,1)-size(opticalFlowX_aedat,3);
mask_eventLocations = createMaskFromQuantized(quantized, quantizedOffset);

%Modify mask to remove erroneous pixels from the left side of the image
mask_eventLocations(1:50,:,:)=0;


% mask_eventLocations= permute(mask_eventLocations,[2 1 3]);
% visualize_matrix3d(mask_eventLocations,0.5);



%%  Map each quantized timeslice to a frame in the ground truth and calculate angular error
quantized_timestamps=quantized_timestamps(quantizedOffset+1:end);
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

% masked_opticalFlowX_aedat = apply_mask(opticalFlowX_aedat,mask_eventLocations);
% masked_opticalFlowY_aedat =apply_mask(opticalFlowY_aedat,mask_eventLocations);
% masked_opticalFlowX_flo = apply_mask(opticalFlowY_flo,mask_eventLocations);
% masked_opticalFlowY_flo= apply_mask(opticalFlowY_flo,mask_eventLocations);
%consider converting the masked matrices to sparse matrices

% Calculate the angular error for all '1'-events saved in quantized
angularErrors= calc_angular_errors(opticalFlowX_aedat, opticalFlowY_aedat ...
    , opticalFlowX_flo, opticalFlowY_flo, indexQuantizedToGT);
eval_angular_errors(angularErrors, mask_eventLocations);




%% Interpolate the Ground Truth to get exact temporal matches for comparison

[intp_flowX_GT, intp_flowY_GT] = interpolate_ground_truth(opticalFlowX_aedat, ...
    opticalFlowY_aedat, opticalFlowX_flo, opticalFlowY_flo, quantized_timestamps, ...
    timecode);


% Calculate angular errors
angularErrors2 = calc_angular_errors(opticalFlowX_aedat, opticalFlowY_aedat, ...
    intp_flowX_GT, intp_flowY_GT);
eval_angular_errors(angularErrors2, mask_eventLocations);



%% Visualize results in quiver plot and save logfile
maskedFlowx=apply_mask(opticalFlowX_aedat,mask_eventLocations);
maskedFlowy=apply_mask(opticalFlowY_aedat,mask_eventLocations);

% make_quiver_movie(strcat(data_Descriptor,'.avi'),opticalFlowX_aedat, opticalFlowY_aedat);
% make_quiver_movie(strcat(data_Descriptor,'_masked.avi'),maskedFlowx, maskedFlowy);
% make_quiver_movie(strcat(data_Descriptor,'_GT.avi'),opticalFlowX_flo,opticalFlowY_flo);
% make_quiver_movie(strcat(data_Descriptor,'_GT_interp.avi'),intp_flowX_GT,intp_flowY_GT);

% fprintf(fid,'%d\n',5); %write the value into the file
% fclose(fid);
diary off;


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














function eval_angular_errors(angularErrors, set_to_nan_mask)
if nargin >1
    angularErrors(set_to_nan_mask==0)=NaN;
end
disp(['Angular Errors: ']);

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
    disp(['Slice nr ', num2str(i), ' - RMSE angular error: ', num2str(sqrtMeanAngularError)])
    disp(['Slice nr ', num2str(i), ' - Abs-Mean angular error: ', num2str(meanAngularErrorAbs)])
    disp(['Slice nr ', num2str(i), ' - Median angular error : ', num2str(medianAngularError)]);
    
end
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
end

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
    end
end

end






function [timecode] = parse_timecode(timecode_file, disregarded_timesteps)
if nargin < 1
    timecode_file='../data/pushbot-timecode.txt';
end

[frame_nr time] = textread(timecode_file, '%f %f', ...
    'headerlines', 3+disregarded_timesteps) ;
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
retinaSize=[240 180];

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