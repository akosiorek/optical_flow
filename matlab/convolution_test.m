function convolution_test()
    clc, clear all 
    
%     %% 2d conv test
%     load 'filter.mat'
%     N = 128;
%     filter = f(:, :, 99);
%     size(filter);
%     nnz(filter);
%     
%     img = rand(N, N);
%     filteredImg = conv2(img, filter, 'same');
%     size(filteredImg);
%     
%     fourierFilter = fft2(padarray(filter, size(img) - 1, 0, 'post'));
%     fourierImg = fft2(padarray(img, size(filter) - 1, 0, 'post'));
%     fourierFilteredImg = fourierFilter .* fourierImg;
%     fourierFilteredImg = ifft2(fourierFilteredImg);
%   
%     
%     a = 12;
%     b = 22-a;
%     fourierFilteredImg = fourierFilteredImg(a:end-b, a:end-b);
%     size(fourierFilteredImg);
%     norm(fourierFilteredImg(:) - filteredImg(:))
    
    
    %% 3d conv test
    
    load 'filter.mat'
    A = 22;
    B = 188;
    filter = f(1:A, 1:A, 1:B);
    size(filter)
    N = 22;
    img = rand(N, N, N * 2);
    size(img)
    filteredImg = convn(img, filter, 'same');
    
    fourierFilter = fftn(padarray(filter, size(img) - 1, 0, 'post'));
    fourierImg = fftn(padarray(img, size(filter) - 1, 0, 'post'));
    fourierFilteredImg = fourierFilter .* fourierImg;
    fourierFilteredImg = ifftn(fourierFilteredImg);
    
    a = floor(A / 2) + 1;
    b = A - a;
    c = floor(B / 2) + 1;
    d = B - c;
    fourierFilteredImg = fourierFilteredImg(a:end-b, a:end-b, c:end-d);
    
    size(filteredImg)
    size(fourierFilteredImg)
    norm(fourierFilteredImg(:) - filteredImg(:))    
end