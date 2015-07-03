function [quantized, timestamps, allEvents] = quantize_events(events, time_resolution, retinaSize)
% QUANTIZE_EVENTS
% time_resolution is given in seconds
    
    time_resolution = time_resolution * 1e6;
    time_span = events(end, 1) - events(1, 1);
    time_steps = ceil(time_span / time_resolution);
    quantized = cell(time_steps, 1);
    allEvents = cell(time_steps, 1);
    timestamps=zeros(time_steps,1);
    
    time_end = events(1, 1) + time_resolution;
    current_step = 1;
    quantized{1} = zeros(retinaSize(1), retinaSize(2));
    
    numEvents = size(events, 1);
    i = 1;
%     waitbarHandle = waitbar(0, 'Quantizing events. Please wait...');
    while i <= numEvents
        quantized{current_step} = sparse(quantized{current_step});
        allEvents{current_step} = sparse(allEvents{current_step});

        if events(i, 1) > time_end
            time_end = time_end + time_resolution;
%             disp(['Slice number', num2str(current_step), ', at time ', num2str(time_end), ', with ', num2str(i), ' events']);
            timestamps(current_step)=time_end;
            current_step = current_step + 1;
            quantized{current_step} = zeros(retinaSize(1), retinaSize(2));
            allEvents{current_step} = zeros(retinaSize(1), retinaSize(2));
        end
        
        x = events(i, 2) + 1;
        y = events(i, 3) + 1;
%         quantized{current_step}(x, y)
        response = quantized{current_step}(x, y);
        response = response + events(i, 4);
        quantized{current_step}(x, y) = response;
        allEvents{current_step}(x, y) = 1;
        
        if mod(i, 100) == 0
%             waitbar(i / numEvents, waitbarHandle);
        end
        i = i + 1;        
    end
    timestamps(current_step)=events(end,1);
    quantized{current_step} = sparse(quantized{current_step});
%     close(waitbarHandle)
end