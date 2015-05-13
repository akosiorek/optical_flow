function show_filter(filter, varargin)
% SHOW_FILTER   creates 3D plot of a Gabor filter
%   SHOW_FILTER(filter) plots the Gabor filter. If numel(size(filter) == 2 it
%               plots the filter. If numel(size(filter) == 3 it plots the middle
%               timestep of the filter, that is floor(size(filter, 3) / 2).
%   SHOW_FILTER(filter, timestep) plots the Gabor filter at the given timestep.
%
%   See also CREATE_FILTER    
    
    s = size(filter);
    x = (s(1) - 1) / 2;
    x = -x:x;
    y = (s(2) - 1) / 2;
    y = -y:y;
    
    f = [];
    if numel(s) > 2
        if nargin > 1
            t = varargin{1};
        else
            t = floor(s(3) / 2);  
        end        
        f = filter(:, :, t);
    else
        t = 1;
        f = filter;
    end

    mesh(x, y, f);
    title_string = sprintf('Gabor filter, time step = %d', t);
    title(title_string);
    xlabel('x');
    ylabel('y');
    zlabel('response');    
end