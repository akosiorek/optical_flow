function make_quiver_movie(name, opticalFlowX,opticalFlowY, quantized)
%MAKE_QUIVER_MOVIE Summary of this function goes here
%   Detailed explanation goes here


figureHandle = figure('units','normalized','outerposition',[0 0 1 1]);
loops = size(opticalFlowX,3);
F(loops) = struct('cdata',[],'colormap',[]);
for i = 1:loops
    if nargin <4
        show_flow(i,opticalFlowX,opticalFlowY);
    else
        show_flow(i,opticalFlowX,opticalFlowY, quantized);
    end
    xlim([-5 size(opticalFlowX,1)+5]);
    ylim([-5 size(opticalFlowY,2)+5]);
    %         view([0 90]);
    drawnow
    F(i) = getframe(gcf);
end
vidObj = VideoWriter(name);
      open(vidObj);
      writeVideo(vidObj,F);
      close(vidObj);

close(figureHandle);


end

