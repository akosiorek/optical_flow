function convert_aedat_to_tsv( aedat_file_path, tsv_file_path, cleaner_nr )
%CONVERT_AEDAT_TO_TSV read out events from .aedat file and save them to a .tsv file

tsv_matrix = events_from_aedat(aedat_file_path, cleaner_nr);
save_aedat_events(tsv_matrix,tsv_file_path);


end

function tsv_matrix = events_from_aedat(aedat_location, cleaner_nr)
if nargin < 1
    aedat_location='../data/aedat_events.aedat';
end
% [allAddr,allTs]=loadaerdat('../data/pushbot_DVS.aedat');
[x,y,pol,ts]=getDVSeventsDavis(aedat_location);
%write to text file

%comparable .tsv files use indexing from 0, aedat reader saves starting
%from 1 .. apparently not?
% x=x-1;
% y=y-1;

%clean defect pixels
if (cleaner_nr == 1 || cleaner_nr == 2 || cleaner_nr == 3)
    [x, y, pol, ts] = cleaner(x, y, pol, ts,cleaner_nr);
end


id = (1:size(x))';


tsv_matrix=[ts x y pol id];

end


%Cleaner by Florian Scherer
function [xVec,yVec,polVec,tsVec] = cleaner(xVec,yVec,polVec,tsVec, cleaner_nr)

% deletes events from broken pixels

nrEvents = size(xVec,1);


x = 0;
y = 0;

j = 0; % tracked how many rows got erased and adapts the index


if (cleaner_nr == 1)
    for i = 1 : nrEvents % erases all events from "broken" pixels
        
        x = xVec(i - j);
        y = yVec(i - j);
        
        
        % %     settings for 'square' data
        if ((x == 0)&&(y == 0)) || ((x == 23)&&(y == 95)) || ((x == 24)&&(y == 95)) || ((x == 24)&&(y == 94)) || ((x == 33)&&(y == 66))|| ((x == 168)&&(y == 91)) || ((x==132)&&(y==55))
            xVec(i - j) = [];
            yVec(i - j) = [];
            polVec(i - j) = [];
            tsVec(i - j) = [];
            
            j = j + 1;
        end
    end
    
elseif (cleaner_nr == 2)
    for i = 1 : nrEvents % erases all events from "broken" pixels
        
        x = xVec(i - j);
        y = yVec(i - j);
        
        
        % Settings for pushbot
        if ((x == 0)&&(y == 0)) || ((x == 27)&&(y == 110)) || ((x == 28)&&(y == 109)) || ((x == 12)&&(y == 125)) || ((x == 117)&&(y == 168))|| ((x == 160)&&(y == 110))
            xVec(i - j) = [];
            yVec(i - j) = [];
            polVec(i - j) = [];
            tsVec(i - j) = [];
            
            j = j + 1;
        end
    end
    
elseif (cleaner_nr == 3)
    for i = 1 : nrEvents % erases all events from "broken" pixels
        
        x = xVec(i - j);
        y = yVec(i - j);
        
        
        %Settings for skateboard
        if ((x == 0)&&(y == 0)) || ((x == 199)&&(y == 14)) || ((x == 122)&&(y == 9)) || ((x == 25)&&(y == 72)) || ((x == 147)&&(y == 43))|| ((x == 199)&&(y == 14)) || ((x == 64)&&(y == 80)) || ((x == 118)&&(y == 3))
            xVec(i - j) = [];
            yVec(i - j) = [];
            polVec(i - j) = [];
            tsVec(i - j) = [];
            
            j = j + 1;
        end
    end
    
end



end


function save_aedat_events(tsv_matrix,name_of_saved_tsv)
if nargin < 1
    name_of_saved_tsv='../data/aedat_events.tsv';
end
% dlmwrite(name_of_saved_tsv,tsv_matrix,'\t');
dlmwrite(name_of_saved_tsv,tsv_matrix, 'delimiter', '\t', 'precision', 16);

end