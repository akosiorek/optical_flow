function quantized = quantize_events(events, time_resolution)
% QUANTIZE_EVENTS
% time_resolution is given in seconds
    
    time_resolution = time_resolution * 1e6;
    time_span = events(end, 3) - events(1, 3);
    time_steps = ceil(time_span / time_resolution);
    quantized = cell(time_steps, 1);
    
    time_end = events(1, 3) + time_resolution;
    current_step = 1;
    quantized{1} = zeros(128, 128);
    
    numEvents = size(events, 1);
    i = 1;
    waitbarHandle = waitbar(0, 'Quantizing events. Please wait...');
    while i <= numEvents
        quantized{current_step} = sparse(quantized{current_step});
        if events(i, 3) > time_end
            time_end = time_end + time_resolution;
            current_step = current_step + 1;
            quantized{current_step} = zeros(128, 128);
        end
        
        x = events(i, 1) + 1;
        y = events(i, 2) + 1;
        response = quantized{current_step}(x, y) + events(i, 4);
        quantized{current_step}(x, y) = response;
        
        if mod(i, 100) == 0
            waitbar(i / numEvents, waitbarHandle);
        end
        i = i + 1;        
    end
    quantized{current_step} = sparse(quantized{current_step});
    close(waitbarHandle)
end