function optical_flow(eventFile, retinaSize, time_start_end, time_resolution)
    clc
    
    if nargin < 1
    eventFile='../data/events_medium.tsv';
    end
    if nargin <2
            retinaSize=[128 128];
    end
    if nargin <3
        time_start = 0;
        time_end = 0.7;
    else
        time_start=time_start_end(1);
        time_end=time_start_end(2);
    end
    if nargin <4
        time_resolution=0.01;
%         time_resolution=0.0025;

    end
    
    
%     time_resolution = 0.01;

    angles = [0 45 90 135 180 225 270 315];
%     retinaSize = [128 128];
    
    
    events = read_events(eventFile, 1);
    fprintf('Number of events:\t%d\n', size(events, 1));
    [quantized timestamps]= quantize_events(events, time_resolution, retinaSize);
    fprintf('Number of event slices:\t%d\n', size(quantized, 1));
    
%     % used to convert EventSlices to a displayable format
%     opticalFlow = zeros(128, 128, size(quantized, 1));
%     for i = 1:size(quantized, 1)
%         opticalFlow(:, :, i) = quantized{i};
%     end
    

%fix this here for a very low number of slices (aedat file)
    filterTimeSteps = numel(time_start:time_resolution:time_end);
    numEventSlices = numel(quantized);
    opticalFlowSize = [retinaSize numEventSlices-filterTimeSteps];
    opticalFlowX = zeros(opticalFlowSize);
    opticalFlowY = zeros(opticalFlowSize);
    
    for angle = angles
        filter = create_filter(angle, [time_start time_end], time_resolution);
        filterSize = size(filter);
        fourierFilter = transform_to_fourier(filter, retinaSize);
        fprintf('filter size = '), disp(size(filter))
        fprintf('fourier filter size = '), disp(size(fourierFilter))

        fourierEvents = transform_to_fourier(quantized, size(filter(:, :, 1)));
        fprintf('events size = '), disp(size(quantized))
        fprintf('fourier events size = '), disp(size(fourierEvents))

        tic
        responses = convolve(fourierEvents, fourierFilter, filterSize(1:2));
        toc
        
        % sum up stuff in X direction
        correctionOffset = -90;
        opticalFlowX = opticalFlowX + cos((correctionOffset+angle) / 180 * pi) * responses;
        % and in Y direction
        opticalFlowY = opticalFlowY - sin((correctionOffset+angle) / 180 * pi) * responses;
    end
    save('flow.mat', 'opticalFlowX', 'opticalFlowY','quantized', 'timestamps')
    
    
%     make_movie(opticalFlowX, 'flow_x.avi');  
%     make_movie2(opticalFlowX, opticalFlowY, quantized, 'flow_quivers.avi');
%     make_quiver_movie('flow_quivers.avi',opticalFlowX,opticalFlowY,quantized);
end


function padded = pad_to_fourier(data, theOtherSize)
    if iscell(data)
        pad = pad_to_fourier(full(data{1}), theOtherSize);
        padded = zeros([size(pad) size(data, 1)]);
        padded(:, :, 1) = pad;
        for i = 2:size(data, 1)
            padded(:, :, i) = pad_to_fourier(full(data{i}), theOtherSize);
        end  
        
    elseif numel(size(data)) == 2
        padded = padarray(data, theOtherSize - 1, 0, 'post'); 
    else
        padded = padarray(data, [theOtherSize-1 0], 0, 'post');
    end
end

function extracted = extract_from_fourier(data, filterSize)
   N = filterSize(1);
   from = floor(N / 2) + 1;
   to = N - from;
   extracted = data(from:end-to, from:end-to); 
end

function fourierFilter = transform_to_fourier(filter, theOtherSize)
   
   filter = pad_to_fourier(filter, theOtherSize);
   fourierFilter = zeros(size(filter));
   for i = 1:size(filter, 3)
       fourierFilter(:, :, i) = fft2(filter(:, :, i));
   end
end


function totalResponse = convolve_one(fourierEventSlices, fourierFilterBank, filterSize)
   
    assert(all(size(fourierEventSlices) == size(fourierFilterBank)),...
        'Event slices and filter bank have to be the same size')
    
    [N M K] = size(fourierEventSlices);
    totalResponse = zeros([N M]);
    for i = 1:K
        fourierResponse = fourierEventSlices(:, :, i) .* fourierFilterBank(:, :, i);
        totalResponse = totalResponse + fourierResponse;        
    end    
    totalResponse = extract_from_fourier(ifft2(totalResponse), filterSize);
end

function responses = convolve(fourierEventSlices, fourierFilterBank, filterSize)
    assert(size(fourierEventSlices, 1) == size(fourierFilterBank, 1));
    assert(size(fourierEventSlices, 2) == size(fourierFilterBank, 2));
    assert(size(fourierEventSlices, 3) >= size(fourierFilterBank, 3));
    
    filterDepth = size(fourierFilterBank, 3);
    [N M K] = size(fourierEventSlices);
    toCompute = K - filterDepth;
    
    waitbarHandle = waitbar(0, 'Computing OpticalFlow. Please wait...');
    responses = zeros([([N M] - filterSize + 1) toCompute]);
    for i = 1:toCompute
        responses(:, :, i) = convolve_one(fourierEventSlices(:, :, i:i+filterDepth-1),...
            fourierFilterBank, filterSize);
        if mod(i, 10) == 0
            waitbar(i / toCompute, waitbarHandle);
        end
    end
    close(waitbarHandle)
end

function make_movie(data, name)
    
    figureHandle = figure('units','normalized','outerposition',[0 0 1 1]);
    loops = size(data, 3);
    F(loops) = struct('cdata',[],'colormap',[]);
    for i = 1:loops
        show_filter(data, i)
        view([0 90]);
        drawnow
        F(i) = getframe(gcf);
    end
    movie2avi(F, name, 'compression', 'none');
    close(figureHandle);    
end

function make_movie2(opticalFlowX,opticalFlowY, quantized, name)
    
    figureHandle = figure('units','normalized','outerposition',[0 0 1 1]);
    loops = size(opticalFlowX,3);
    F(loops) = struct('cdata',[],'colormap',[]);
    for i = 1:loops
show_flow(i,opticalFlowX,opticalFlowY, quantized);
    xlim([-5 size(opticalFlowX,2)+5]);
    ylim([-5 size(opticalFlowY,1)+5]);
%         view([0 90]);
        drawnow
        F(i) = getframe(gcf);
    end
    movie2avi(F, name, 'compression', 'none');
    close(figureHandle);    
end