function eval_setups()

%For first three evaluations, adjust the 'cleaner' for defect erroneous
%pixels in convert_aedat_to_tsv.m

%Set date for logfile
t = [datetime('now')];
DateString = datestr(t);
DateString=strrep(DateString,' ','_');


do_plot=0;
% evalQuadrat(do_plot);
% evalPushbot(do_plot);
% evalSkateboard(do_plot);
% evalBaelle(do_plot);
% evalSquare12(do_plot);
% evalSquare2(do_plot);

evalCompareCxx(0);






%% not yet updated

%check out .tsv generated by pinwheel
% evaluation_algorithm('../data/pinwheel/', 'pinwheel_b&f_DVS.aedat', ...
%     'timestamps.txt', 0, 'pinwheel', [time_start time_end time_resolution], ...
%     angles);



%% evaluation functions


%% 'Square' evaluation settings
    function evalQuadrat(do_plot)
        
        path_to_data='../data/quadrat/';
        aedat_file='quadrat_DVS_Only.aedat';
        timecode_file='quadrat_Frames_Timecode.txt';
        disregarded_timesteps=1;
        data_Descriptor='quadrat';
        negate_GT_Y =0;
        %Create logfile
        logfile_name=strcat(path_to_data,data_Descriptor,'_logfile_', DateString, '.txt');
        resultsTable_name=strcat(path_to_data,data_Descriptor,'_evalResults_', DateString, '.csv');
        
        diary(logfile_name);
        
        
        %Dummy table for concatenation
        results_table = table({''}, 0, 0, 0, 0, 0, {''}, ...
            0, 0, 0, 0, 0, 0, ...
            0, 0, 0, 0, ...
            0, 0, 0, ...
            0, 0, 0, ...
            0, 0, 0, ...
            0, 0, 'VariableNames',{'Descriptor' 'retinaSizeX' 'retinaSizeY' 'time_start' 'time_end' 'time_resolution' 'angles' ...
            'Slice_Nr' 'Quantized_time' 'avgSpeed' 'avgSpeedAll' 'meanAngle' 'medianAngle' 'RMSE_directErr' 'AbsMean_directErr' 'Median_directErr' 'timeDiff_directErr' ...
            'RMSE_intpErr' 'AbsMean_intpErr' 'Median_intpErr' ...
            'avgRMSE_directErr' 'avgAbsMean_directErr' 'avgMedian_directErr' ...
            'avgRMSE_intpErr' 'avgAbsMean_intpErr' 'avgMedian_intpErr' ...
            'avgMeanAngle' 'avgMedianAngle'});
        %
        %
        % % evaluation for single parameter setting
        % [time_start time_end time_resolution] = get_time_settings(3);
        % angles = get_angles(1);
        % results_table = vertcat(results_table, evaluation_algorithm(path_to_data, aedat_file, timecode_file, disregarded_timesteps, ...
        %      data_Descriptor, [time_start time_end time_resolution], angles, do_plot, negate_GT_Y));
        %
        % evaluation for multiple parameter settings
        for time_settings=3%1:6
            for angle_settings=1%:2
                % %square evaluation
                [time_start, time_end, time_resolution]=get_time_settings(time_settings);
                angles=get_angles(angle_settings);
                results_table = vertcat(results_table, evaluation_algorithm(path_to_data, aedat_file, timecode_file, disregarded_timesteps, ...
                    data_Descriptor, [time_start time_end time_resolution], angles, do_plot, negate_GT_Y));
            end
        end
        
        writetable(results_table,resultsTable_name,'Delimiter','\t','QuoteStrings',false)
        
        
    end


%% Pushbot evaluation
    function evalPushbot(do_plot)
        path_to_data='../data/pushbot/';
        aedat_file='pushbot_DVS.aedat';
        timecode_file='pushbot-timecode.txt';
        disregarded_timesteps=0;
        data_Descriptor='pushbot';
        negate_GT_Y =1;
        % %Create logfile
        % logfile_name=strcat(path_to_data,data_Descriptor,'_logfile_', DateString, '.txt');
        resultsTable_name=strcat(path_to_data,data_Descriptor,'_evalResults_', DateString, '.csv');
        %
        % diary(logfile_name);
        %
        %Dummy table for concatenation
        results_table = table({''}, 0, 0, 0, 0, 0, {''}, ...
            0, 0, 0, 0, 0, 0, ...
            0, 0, 0, 0, ...
            0, 0, 0, ...
            0, 0, 0, ...
            0, 0, 0, ...
            0, 0, 'VariableNames',{'Descriptor' 'retinaSizeX' 'retinaSizeY' 'time_start' 'time_end' 'time_resolution' 'angles' ...
            'Slice_Nr' 'Quantized_time' 'avgSpeed' 'avgSpeedAll' 'meanAngle' 'medianAngle' 'RMSE_directErr' 'AbsMean_directErr' 'Median_directErr' 'timeDiff_directErr' ...
            'RMSE_intpErr' 'AbsMean_intpErr' 'Median_intpErr' ...
            'avgRMSE_directErr' 'avgAbsMean_directErr' 'avgMedian_directErr' ...
            'avgRMSE_intpErr' 'avgAbsMean_intpErr' 'avgMedian_intpErr' ...
            'avgMeanAngle' 'avgMedianAngle'});
        % % % pushbot evaluation
        %  [time_start time_end time_resolution] = get_time_settings(3);
        % angles = get_angles(1);
        % results_table = vertcat(results_table,evaluation_algorithm(path_to_data, aedat_file, timecode_file, disregarded_timesteps, ...
        %     data_Descriptor, [time_start time_end time_resolution], angles, do_plot, negate_GT_Y));
        %
        %
        % multiple parameters
        for time_settings=3%1:6
            for angle_settings=1:2
                [time_start, time_end, time_resolution]=get_time_settings(time_settings);
                angles=get_angles(angle_settings);
                results_table = vertcat(results_table, evaluation_algorithm(path_to_data, aedat_file, timecode_file, disregarded_timesteps, ...
                    data_Descriptor, [time_start time_end time_resolution], angles, do_plot, negate_GT_Y));
            end
        end
        writetable(results_table,resultsTable_name,'Delimiter','\t','QuoteStrings',false)
        
    end


%% skateboard evaluation
    function evalSkateboard(do_plot)
        %
        path_to_data='../data/skateboard/';
        aedat_file='tony_hawk_DVS_Only.aedat';
        timecode_file='tony_hawk-timecode.txt';
        disregarded_timesteps=1;
        data_Descriptor='skateboard';
        negate_GT_Y =0;
        % %Create logfile
        % logfile_name=strcat(path_to_data,data_Descriptor,'_logfile_', DateString, '.txt');
        resultsTable_name=strcat(path_to_data,data_Descriptor,'_evalResults_', DateString, '.csv');
        %
        % % diary(logfile_name);
        %
        %
        
        %Dummy table for concatenation
        results_table = table({''}, 0, 0, 0, 0, 0, {''}, ...
            0, 0, 0, 0, 0, 0, ...
            0, 0, 0, 0, ...
            0, 0, 0, ...
            0, 0, 0, ...
            0, 0, 0, ...
            0, 0, 'VariableNames',{'Descriptor' 'retinaSizeX' 'retinaSizeY' 'time_start' 'time_end' 'time_resolution' 'angles' ...
            'Slice_Nr' 'Quantized_time' 'avgSpeed' 'avgSpeedAll' 'meanAngle' 'medianAngle' 'RMSE_directErr' 'AbsMean_directErr' 'Median_directErr' 'timeDiff_directErr' ...
            'RMSE_intpErr' 'AbsMean_intpErr' 'Median_intpErr' ...
            'avgRMSE_directErr' 'avgAbsMean_directErr' 'avgMedian_directErr' ...
            'avgRMSE_intpErr' 'avgAbsMean_intpErr' 'avgMedian_intpErr' ...
            'avgMeanAngle' 'avgMedianAngle'});
        % % evaluation for single parameter setting
        % [time_start time_end time_resolution] = get_time_settings(3);
        % angles = get_angles(1);
        % results_table = vertcat(results_table,evaluation_algorithm(path_to_data, aedat_file, timecode_file, disregarded_timesteps, ...
        %      data_Descriptor, [time_start time_end time_resolution], angles, do_plot, negate_GT_Y));
        
        % evaluation for multiple parameter settings
        for time_settings=3%1:6
            for angle_settings=1:2
                [time_start, time_end, time_resolution]=get_time_settings(time_settings);
                angles=get_angles(angle_settings);
                results_table = vertcat(results_table, evaluation_algorithm(path_to_data, aedat_file, timecode_file, disregarded_timesteps, ...
                    data_Descriptor, [time_start time_end time_resolution], angles, do_plot, negate_GT_Y));
            end
        end
        writetable(results_table,resultsTable_name,'Delimiter','\t','QuoteStrings',false)
        
        
    end

%% baelle_sim
    function evalBaelle(do_plot)
        
        
        %
        path_to_data='../data/Simulation_baelle/';
        aedat_file='baelle_sim.aedat';
        timecode_file='timestamps_fixed.txt';
        disregarded_timesteps=1;
        data_Descriptor='baelle';
        negate_GT_Y =0;
        %Create logfile
        %         logfile_name=strcat(path_to_data,data_Descriptor,'_logfile_', DateString, '.txt');
        resultsTable_name=strcat(path_to_data,data_Descriptor,'_evalResults_', DateString, '.csv');
        withFloReference = 1;
        %         diary(logfile_name);
        
        %Dummy table for concatenation
        results_table = table({''}, 0, 0, 0, 0, 0, {''}, ...
            0, 0, 0, 0, 0, 0, ...
            0, 0, 0, 0, ...
            0, 0, 0, ...
            0, 0, 0, ...
            0, 0, 0, ...
            0, 0, 'VariableNames',{'Descriptor' 'retinaSizeX' 'retinaSizeY' 'time_start' 'time_end' 'time_resolution' 'angles' ...
            'Slice_Nr' 'Quantized_time' 'avgSpeed' 'avgSpeedAll' 'meanAngle' 'medianAngle' 'RMSE_directErr' 'AbsMean_directErr' 'Median_directErr' 'timeDiff_directErr' ...
            'RMSE_intpErr' 'AbsMean_intpErr' 'Median_intpErr' ...
            'avgRMSE_directErr' 'avgAbsMean_directErr' 'avgMedian_directErr' ...
            'avgRMSE_intpErr' 'avgAbsMean_intpErr' 'avgMedian_intpErr' ...
            'avgMeanAngle' 'avgMedianAngle'});
        
        
        
        % evaluation for single parameter setting
        % [time_start time_end time_resolution] = get_time_settings(8);
        for time_settings=7%7:9
            for angle_settings=1:2
                [time_start, time_end, time_resolution]=get_time_settings(time_settings);
                angles=get_angles(angle_settings);
                
                results_table = vertcat(results_table, evaluation_algorithm(path_to_data, aedat_file, timecode_file, disregarded_timesteps, ...
                    data_Descriptor, [time_start time_end time_resolution], angles, do_plot, negate_GT_Y, withFloReference));
            end
        end
        
        writetable(results_table,resultsTable_name,'Delimiter','\t','QuoteStrings',false)
        
        
    end

%% square1.2
    function evalSquare12(do_plot)
        
        %
        path_to_data='../data/square1.2/';
        aedat_file='quadrat_1_2.aedat';
        timecode_file='timestampsGT_fixed.txt';
        disregarded_timesteps=1;
        data_Descriptor='square1.2';
        negate_GT_Y =1;
        %Create logfile
        %         logfile_name=strcat(path_to_data,data_Descriptor,'_logfile_', DateString, '.txt');
        resultsTable_name=strcat(path_to_data,data_Descriptor,'_evalResults_', DateString, '.csv');
        withFloReference = 1;
        % diary(logfile_name);
        
        %Dummy table for concatenation
        results_table = table({''}, 0, 0, 0, 0, 0, {''}, ...
            0, 0, 0, 0, 0, 0, ...
            0, 0, 0, 0, ...
            0, 0, 0, ...
            0, 0, 0, ...
            0, 0, 0, ...
            0, 0, 'VariableNames',{'Descriptor' 'retinaSizeX' 'retinaSizeY' 'time_start' 'time_end' 'time_resolution' 'angles' ...
            'Slice_Nr' 'Quantized_time' 'avgSpeed' 'avgSpeedAll' 'meanAngle' 'medianAngle' 'RMSE_directErr' 'AbsMean_directErr' 'Median_directErr' 'timeDiff_directErr' ...
            'RMSE_intpErr' 'AbsMean_intpErr' 'Median_intpErr' ...
            'avgRMSE_directErr' 'avgAbsMean_directErr' 'avgMedian_directErr' ...
            'avgRMSE_intpErr' 'avgAbsMean_intpErr' 'avgMedian_intpErr' ...
            'avgMeanAngle' 'avgMedianAngle'});
        %
        
        
        % evaluation for single parameter setting
        % [time_start time_end time_resolution] = get_time_settings(8);
        % for j=1:1
        %     angles=get_angles(j);
        %  results_table = vertcat(results_table, evaluation_algorithm(path_to_data, aedat_file, timecode_file, disregarded_timesteps, ...
        %     data_Descriptor, [time_start time_end time_resolution], angles, do_plot, negate_GT_Y, withFloReference));
        % end
        
        
        for time_settings=10%:12
            for angle_settings=1:2
                [time_start, time_end, time_resolution]=get_time_settings(time_settings);
                angles=get_angles(angle_settings);
                
                results_table = vertcat(results_table, evaluation_algorithm(path_to_data, aedat_file, timecode_file, disregarded_timesteps, ...
                    data_Descriptor, [time_start time_end time_resolution], angles, do_plot, negate_GT_Y, withFloReference));
            end
        end
        
        writetable(results_table,resultsTable_name,'Delimiter','\t','QuoteStrings',false)
        
        
        
    end


%% square2
    function evalSquare2(do_plot)
        
        %
        path_to_data='../data/square2/';
        aedat_file='quadrat_2.aedat';
        timecode_file='timestampsGT_fixed.txt';
        disregarded_timesteps=1;
        data_Descriptor='square2';
        negate_GT_Y =1;
        %Create logfile
        %         logfile_name=strcat(path_to_data,data_Descriptor,'_logfile_', DateString, '.txt');
        resultsTable_name=strcat(path_to_data,data_Descriptor,'_evalResults_', DateString, '.csv');
        withFloReference = 1;
        %         diary(logfile_name);
        
        %Dummy table for concatenation
        results_table = table({''}, 0, 0, 0, 0, 0, {''}, ...
            0, 0, 0, 0, 0, 0, ...
            0, 0, 0, 0, ...
            0, 0, 0, ...
            0, 0, 0, ...
            0, 0, 0, ...
            0, 0, 'VariableNames',{'Descriptor' 'retinaSizeX' 'retinaSizeY' 'time_start' 'time_end' 'time_resolution' 'angles' ...
            'Slice_Nr' 'Quantized_time' 'avgSpeed' 'avgSpeedAll' 'meanAngle' 'medianAngle' 'RMSE_directErr' 'AbsMean_directErr' 'Median_directErr' 'timeDiff_directErr' ...
            'RMSE_intpErr' 'AbsMean_intpErr' 'Median_intpErr' ...
            'avgRMSE_directErr' 'avgAbsMean_directErr' 'avgMedian_directErr' ...
            'avgRMSE_intpErr' 'avgAbsMean_intpErr' 'avgMedian_intpErr' ...
            'avgMeanAngle' 'avgMedianAngle'});
        %
        
        
        % evaluation for single parameter setting
        % [time_start time_end time_resolution] = get_time_settings(8);
        % for j=1:1
        %     angles=get_angles(j);
        %  results_table = vertcat(results_table, evaluation_algorithm(path_to_data, aedat_file, timecode_file, disregarded_timesteps, ...
        %     data_Descriptor, [time_start time_end time_resolution], angles, do_plot, negate_GT_Y, withFloReference));
        % end
        
        
        for time_settings=10
            for angle_settings=1:2
                [time_start, time_end, time_resolution]=get_time_settings(time_settings);
                angles=get_angles(angle_settings);
                
                results_table = vertcat(results_table, evaluation_algorithm(path_to_data, aedat_file, timecode_file, disregarded_timesteps, ...
                    data_Descriptor, [time_start time_end time_resolution], angles, do_plot, negate_GT_Y, withFloReference));
            end
        end
        
        writetable(results_table,resultsTable_name,'Delimiter','\t','QuoteStrings',false)
        
        
    end




    function [time_start, time_end, time_resolution] = get_time_settings(index)
        if (index==1)
            time_start=0;
            time_end=0.3;
            time_resolution=0.01;
            
        end
        if (index==2)
            time_start=0;
            time_end=0.5;
            time_resolution=0.01;
        end
        if (index==3)
            time_start=0;
            time_end=0.7;
            time_resolution=0.01;
        end
        if (index==4)
            time_start=0;
            time_end=0.3;
            time_resolution=0.005;
            
        end
        if (index==5)
            time_start=0;
            time_end=0.5;
            time_resolution=0.005;
        end
        if (index==6)
            time_start=0;
            time_end=0.7;
            time_resolution=0.005;
        end
        if (index==7) % For 'baelle'
            time_start=0;
            time_end=0.03;
            time_resolution=0.001;
        end
        if (index==8) % For 'baelle'
            time_start=0;
            time_end=0.05;
            time_resolution=0.001;
        end
        if (index==9) % For 'baelle'
            time_start=0;
            time_end=0.08;
            time_resolution=0.001;
        end
        if (index==10) %
            time_start=0;
            time_end=0.2;
            time_resolution=0.005;
        end
        if (index==11) %
            time_start=0;
            time_end=0.15;
            time_resolution=0.0025;
        end
        if (index==12) %
            time_start=0;
            time_end=0.1;
            time_resolution=0.0025;
        end
    end

    function [angles] = get_angles(index)
        if (index==1)
            angles = [0 45 90 135 180 225 270 315];
        end
        if (index==2)
            angles = [0 30 60 90 120 150 180 210 240 270 300 330];
        end
        if (index==3)
            angles = [0 20 40 60 80 100 120 140 160 180 200 240 260 280 300 320 340];
        end
    end


% diary off;

end