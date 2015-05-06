function optical_flow()
    clc, clear all
    time_resolution = 0.01;
    time_start = 0;
    time_end = 0.7;
    angle = 45;
    retinaSize = [128 128];
    
    
    events = read_events('../data/events.tsv', 1);
    quantized = quantize_events(events, time_resolution);
    
    filter = create_filter(angle, [time_start time_end], time_resolution);
    fourierFilter = transform_to_fourier(filter, retinaSize);
    fprintf('filter size = '), disp(size(filter))
    fprintf('fourier filter size = '), disp(size(fourierFilter))
    
    fourierEvents = transform_to_fourier(quantized, size(filter(:, :, 1)));
    fprintf('events size = '), disp(size(quantized))
    fprintf('fourier events size = '), disp(size(fourierEvents))
    
%     response = convolve_one(fourierEvents(:, :, 1:101), fourierFilter, [21 21])
%     size(response)
%     show_filter(response)

    responses = convolve(fourierEvents, fourierFilter, [21 21]);
    size(responses)
    for i = 1:size(responses, 3)
        show_filter(responses, i)
        pause
    end
    
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
   
    size(fourierEventSlices)
    size(fourierFilterBank)
    assert(all(size(fourierEventSlices) == size(fourierFilterBank)),...
        'Event slices and filter bank have to be the same size')
    
    [N M K] = size(fourierEventSlices);
    totalResponse = zeros([N M] - filterSize + 1);
    for i = 1:K
        fourierResponse = fourierEventSlices(:, :, i) .* fourierFilterBank(:, :, i);
        response = ifft2(fourierResponse);
        totalResponse = totalResponse + extract_from_fourier(response, filterSize);
    end    
end

function responses = convolve(fourierEventSlices, fourierFilterBank, filterSize)
    assert(size(fourierEventSlices, 1) == size(fourierFilterBank, 1));
    assert(size(fourierEventSlices, 2) == size(fourierFilterBank, 2));
    assert(size(fourierEventSlices, 3) >= size(fourierFilterBank, 3));
    
    filterDepth = size(fourierFilterBank, 3);
    [N M K] = size(fourierEventSlices);
    toCompute = K - filterDepth;
    
    responses = zeros([([N M] - filterSize + 1) toCompute]);
    for i = 1:toCompute
        responses(:, :, i) = convolve_one(fourierEventSlices(:, :, i:i+filterDepth-1),...
            fourierFilterBank, filterSize);
    end
end