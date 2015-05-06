function optical_test()
    
    radius = 64;
    timeSpan = 1;
    timeResolution = 0.01;
   

    
    
    X = -radius:radius;
    Y = -radius:radius;
    T = -timeSpan:timeResolution:timeSpan;
    
    [x, y, t] = meshgrid(X, Y, T);
    
    f = filter(x, y, t);
    F = round(f * 1e5) * 1e-5;
    
    
    N = radius-2;
    for indI = 1:N
        for indJ = 1:N
            f = F(indI:end-indI, indI:end-indI, indJ:end-indJ);
            m(indI, indJ) = sum(abs(f(:))) / nnz(f);
            ss(indI, indJ) = sum(abs(f(:)));
        end
    end
%     max(m(:))
    approx = 5; 
    index = max(find(ss == max(ss(:))));
    [bestI, bestJ] = ind2sub(size(m), index);
    bestI = bestI + approx;
    bestJ = bestJ + approx;
    f = F(bestI:end-bestI, bestI:end-bestI, bestJ:end-bestJ);
    size(f)
    numel(f)
    nnz(f)
    
end

