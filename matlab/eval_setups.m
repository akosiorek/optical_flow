function eval_setups()

%For first three evaluations, adjust the 'cleaner' for defect erroneous
%pixels in convert_aedat_to_tsv.m

%Set date for logfile
t = [datetime('now')];
DateString = datestr(t);
DateString=strrep(DateString,' ','_');

%% Pushbot evaluation settings
% path_to_data='../data/pushbot/';
% aedat_file='pushbot_DVS.aedat';
% timecode_file='pushbot-timecode.txt';
% disregarded_timesteps=0;
% data_Descriptor='pushbot';
% [time_start time_end time_resolution] = get_time_settings(3);
% angles = get_angles(1);
% do_plot = 1;
% negate_GT_Y =1;
% %Create logfile
% logfile_name=strcat(path_to_data,data_Descriptor,'_logfile_', DateString, '.txt');
% diary(logfile_name);
% 
% 
% % pushbot evaluation
% evaluation_algorithm(path_to_data, aedat_file, timecode_file, disregarded_timesteps, ...
%     data_Descriptor, [time_start time_end time_resolution], angles, do_plot, negate_GT_Y)

%% 'Square' evaluation settings
% 
% path_to_data='../data/quadrat/';
% aedat_file='quadrat_DVS_Only.aedat';
% timecode_file='quadrat_Frames_Timecode.txt';
% disregarded_timesteps=1;
% data_Descriptor='quadrat';
% do_plot = 1;
% negate_GT_Y =0;
% %Create logfile
% logfile_name=strcat(path_to_data,data_Descriptor,'_logfile_', DateString, '.txt');
% diary(logfile_name);
% 
% % evaluation for single parameter setting
% [time_start time_end time_resolution] = get_time_settings(3);
% angles = get_angles(1);
% evaluation_algorithm(path_to_data, aedat_file, timecode_file, disregarded_timesteps, ...
%      data_Descriptor, [time_start time_end time_resolution], angles, do_plot, negate_GT_Y)

% evaluation for multiple parameter settings
% for time_settings=1:3
%     for angle_settings=1:2
%         % %square evaluation
%         [time_start, time_end, time_resolution]=get_time_settings(time_settings);
%         angles=get_angles(angle_settings);
%  evaluation_algorithm(path_to_data, aedat_file, timecode_file, disregarded_timesteps, ...
%     data_Descriptor, [time_start time_end time_resolution], angles, do_plot, negate_GT_Y);
%     end
% end


%% skateboard evaluation

path_to_data='../data/skateboard/';
aedat_file='tony_hawk_DVS_Only.aedat';
timecode_file='tony_hawk-timecode.txt';
disregarded_timesteps=1;
data_Descriptor='skateboard';
do_plot = 1;
negate_GT_Y =0;
%Create logfile
logfile_name=strcat(path_to_data,data_Descriptor,'_logfile_', DateString, '.txt');
diary(logfile_name);

% evaluation for single parameter setting
[time_start time_end time_resolution] = get_time_settings(3);
angles = get_angles(1);
evaluation_algorithm(path_to_data, aedat_file, timecode_file, disregarded_timesteps, ...
     data_Descriptor, [time_start time_end time_resolution], angles, do_plot, negate_GT_Y)
 

%% not yet updated
%baelle_sim
%For this, fix the number of headerlines in evaluation_algorithm timecode
%reader


%check out .tsv generated by pinwheel
% evaluation_algorithm('../data/pinwheel/', 'pinwheel_b&f_DVS.aedat', ...
%     'timestamps.txt', 0, 'pinwheel', [time_start time_end time_resolution], ...
%     angles);



    function [time_start, time_end, time_resolution] = get_time_settings(index)
        if (index==1)
                        time_start=0;
            time_end=0.6;
            time_resolution=0.015;

        end
        if (index==2)
            time_start=0;
            time_end=0.6;
            time_resolution=0.05;
        end
        if (index==3)
            time_start=0;
            time_end=0.6;
            time_resolution=0.01;
        end
        
    end

    function [angles] = get_angles(index)
                if (index==1)
                    angles=zeros(8,1);
angles = [0 45 90 135 180 225 270 315];

        end
        if (index==2)
            angles=zeros(12,1);
angles = [0 30 60 90 120 150 180 210 240 270 300 330]
        end
    end

diary off;

end