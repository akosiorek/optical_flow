function f = create_filter(angle, timeSpan, varargin)
% CREATE_FILTER     creates a Gabor filter in (x,y,t) space for optical flow detection
%   F = CREATE_FILTER(angle, timeSpan) creates a filter rotated by angle w.r.t. x-axis
%                     spanning the timespan. timespan can be a scalar value or a vector.
%   F = CREATE_FILTER(angle, timeSpan, timeResolution) creates a filter rotated by 
%                     angle w.r.t x-axis spanning the timespan from timeSpan(1) to 
%                     timeSpan(2) with resolution timeResolution
%
%   Note: 
%   - angle is given in degrees
%   - all parameters are as in the paper
%   - response is very close to zero for timesteps < 0.3s and/or x, y > 10, therefore
%       the radius is set to 10
%
%   See also SHOW_FILTER
        
    
    radius = 10;    
    dir = [sqrt(2) * 0.08; 0];
    dir = rot(angle, dir);
    fx0 = dir(1);
    fy0 = dir(2);
    sigma = 25;
    s1 = 0.5;
    s2 = 0.75;
    mi_bi1 = 0.2;
    sigma_bi1 = mi_bi1 / 3;
    mi_bi2 = 2 * mi_bi1;
    sigma_bi2 = 1.5 * sigma_bi1;
    mi_mono = 0.2 * (1 + mi_bi1 * sqrt(36 + 10 * log(s1 / s2)));
    sigma_mono = mi_mono / 3;
    
    
    function s = spatial(x, y)
        s = 2 * pi / sigma^2;
        s = s * exp(2 * pi * i * (fx0 * x + fy0 * y));
        s = s .* exp( -2 * pi^2 * (x.^2 + y.^2) / sigma^2);
    end
    
    function T  = T_mono(t)
        T = gaus(sigma_mono, mi_mono, t);
    end
    
    function T = T_bi(t)
        T = -s1 * gaus(sigma_bi1, mi_bi1, t) + s2 * gaus(sigma_bi2, mi_bi2, t);
    end
    
    function F = filter(x, y, t)       
        s = spatial(x, y);
        re = real(s);
        im = imag(s);
        F = im .* T_mono(t) + re .* T_bi(t);
    end
    
    x = -radius:radius;
    y = -radius:radius;
    t = [];
    if nargin == 3
        timeResolution = varargin{1};
        t = timeSpan(1):timeResolution:timeSpan(2);
    else
        t = timeSpan;
    end
    
    [X, Y, T] = meshgrid(x, y, t);
    
    f = filter(X, Y, T);    
end

function g = gaus(sigma, mi, t)
    g = exp(-0.5 * (t - mi).^2 / sigma^2);
end

function v = rot(angle, x)    
    angle = angle / 180 * pi;   
    v = [cos(angle) -sin(angle); sin(angle) cos(angle)] * x;    
end