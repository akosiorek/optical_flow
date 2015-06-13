function evaluation_algorithm(aedat_file, path_to_flo_files, timecode_file, disregarded_timesteps)
%EVALUATION_ALGORITHM Summary of this function goes here
%   Detailed explanation goes here
if nargin <1
    %    aedat_file='../data/pushbot_DVS.aedat';
    aedat_file='../data/quadrat/quadrat_DVS_Only.aedat';
    
end
if nargin <2
    %     flo_file='../data/scene00002_mdpof.flo';
    path_to_flo_files='../data/quadrat/';
    
end
if nargin < 3
    %     timecode_file='../data/pushbot-timecode.txt';
    timecode_file='../data/quadrat/quadrat_Frames_Timecode.txt';
    
end
if nargin <4
    %     disregarded_timesteps =0
    disregarded_timesteps =1
    
end
%read out events from .aedat file and save them to a .tsv file
[tsv_matrix, retinaSize] = events_from_aedat(aedat_file);
eventsfile_tsv_name='../data/eval_events.tsv'
save_aedat_events(tsv_matrix,eventsfile_tsv_name)




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
%run the optical_flow code
optical_flow(eventsfile_tsv_name, retinaSize, [time_start time_end], time_resolution);

timecode= parse_timecode(timecode_file, disregarded_timesteps);


nr_flo_files = 19;
start_scene_nr = 2;
opticalFlowX=zeros(retinaSize(1),retinaSize(2), nr_flo_files);
opticalFlowY=zeros(retinaSize(1),retinaSize(2), nr_flo_files);
path_to_flo_files='../data/quadrat/';
for i=1:nr_flo_files
    scene_index=i+start_scene_nr;
    if (scene_index<10)
        flo_name=strcat('scene0000',num2str(scene_index),'_mdpof.flo');
    else
        flo_name=strcat('scene000',num2str(scene_index),'_mdpof.flo');
    end
    full_path=strcat(path_to_flo_files,flo_name);
    flo_matrix = readFlowFile(full_path);
    opticalFlowX(:,:,i)=flo_matrix(:,:,1)'; %transpose might be wrong
    opticalFlowY(:,:,i)=flo_matrix(:,:,2)';
end

make_quiver_movie('ground_truth_quivers.avi',opticalFlowX,opticalFlowY);

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
tsv_matrix=[x y ts pol];
retinaSize=[240 180];
%for now, cut down to 128x128
% conditionX=tsvMatrix(:,1)>127;
% conditionY=tsvMatrix(:,2)>127;
% conditionMerged=conditionX | conditionY;
% tsvMatrix(conditionMerged,:)=[];



% clear x y pol ts;
% eventFile='pushbot_DVS.tsv';
%     aedat_events = read_events(tsv_matrix, 1);

% optical_flow(eventFile, retinaSize);

%regarding address format: http://sourceforge.net/p/jaer/discussion/631959/thread/fdbbb82f/
%https://www.ini.uzh.ch/~shih/dbio08/loadaerdat.m
end

function save_aedat_events(tsv_matrix,name_of_saved_tsv)
if nargin < 1
    name_of_saved_tsv='aedat_events.tsv';
end
dlmwrite(name_of_saved_tsv,tsv_matrix,'\t');
end



%
% function [quantized] = timecode_quantizer(events, timecode, retinaSize)
%
%     time_steps = size(timecode,1);
%     quantized = cell(time_steps, 1);
%
%     current_step = 1;
% %     time_start=timecode(current_step,2)
%     if current_step < time_steps
%         time_end = timecode(current_step,2); %time at which reference frame was recorded
% %     else
% %         time_end=events(end,3)
%     end
%
%     quantized{1} = zeros(retinaSize(1), retinaSize(2));
%
%     numEvents = size(events, 1);
%     i = 1;
%     waitbarHandle = waitbar(0, 'Quantizing events. Please wait...');
%     while i <= numEvents
%         quantized{current_step} = sparse(quantized{current_step});
%         if events(i, 3) > time_end
% %             if current_step <time_steps
%                 current_step = current_step +1;
% %             elseif current_step == time_steps
% %                 break
% %             end
%
%             time_end=timecode(current_step,2)
% %             current_step = current_step + 1;
% %             if current_step < time_steps
% %                time_end = timecode(current_step,2);
% %             else
% %                 time_end=events(end,3)
% %             end
%
%
%             quantized{current_step} = zeros(retinaSize(1), retinaSize(2));
%         end
%
%         x = events(i, 1) + 1;
%         y = events(i, 2) + 1;
% %         quantized{current_step}(x, y)
%         response = quantized{current_step}(x, y);
%         response = response + events(i, 4);
%         quantized{current_step}(x, y) = response;
%
%         if mod(i, 100) == 0
%             waitbar(i / numEvents, waitbarHandle);
%         end
%         i = i + 1;
%
%         if current_step+1 ==time_steps
%                break; %no more reference frames in ground truth available
%         end
%     end
%     quantized{current_step} = sparse(quantized{current_step});
%     close(waitbarHandle)
%
% end


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