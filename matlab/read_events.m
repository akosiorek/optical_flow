function [ events ] = read_events( file, response )
%READ_EVENTS Reads events from TSV
%   Parses a tsv event file. If an extended (3D) event-stream is used, the
%   additional 3d data is removed. A 3-dim matrix of x,y,t will be
%   returned.

% Get raw events
events_raw = dlmread(file,'\t');

% Reduce if extended file_size
if(size(events_raw,2) == 7)
    if(response)
        events = events_raw(:,4:7);
        response = events(:, end);
        response(response == 0) = -1;
        events(:, end) = response;
    else
        events = events_raw(:,4:6);
    end
else
    events = events_raw;
end    

end

