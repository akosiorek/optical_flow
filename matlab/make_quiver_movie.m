function make_quiver_movie(name, opticalFlowX,opticalFlowY, quantized_timestamps, quantized)
%MAKE_QUIVER_MOVIE Summary of this function goes here
%   Detailed explanation goes here


figureHandle = figure('units','normalized','outerposition',[0 0 1 1]);
loops = size(opticalFlowX,3);
F(loops) = struct('cdata',[],'colormap',[]);
for i = 1:loops
    if nargin <5
        show_flow(name, i,opticalFlowX,opticalFlowY, quantized_timestamps(i));
    else
        show_flow(name, i,opticalFlowX,opticalFlowY, quantized_timestamps(i), quantized);
    end
    xlim([-5 size(opticalFlowX,1)+5]);
    ylim([-5 size(opticalFlowY,2)+5]);
    %         view([0 90]);
    grid on
    drawnow
     F(i) = getframe(gcf);
end
path = '../data/';
fullname=strcat(path,name,'.avi');
vidObj = VideoWriter(fullname);
      open(vidObj);
      writeVideo(vidObj,F);
      close(vidObj);

close(figureHandle);


end

