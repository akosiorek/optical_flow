function make_quiver_movie(name, opticalFlowX,opticalFlowY, timestamps, quantized)
%MAKE_QUIVER_MOVIE Summary of this function goes here
%   Detailed explanation goes here


figureHandle = figure('units','normalized','outerposition',[0 0 1 1]);
loops = size(opticalFlowX,3);
F(loops) = struct('cdata',[],'colormap',[]);
path = '../data/';
fullname=strcat(path,name,'.avi');

for i = 1:loops
    if nargin <5
        show_flow(name, i,opticalFlowX,opticalFlowY, timestamps(i));
    else
        show_flow(name, i,opticalFlowX,opticalFlowY, timestamps(i), quantized);
    end
    xlim([-5 size(opticalFlowX,1)+5]);
    ylim([-5 size(opticalFlowY,2)+5]);
    %         view([0 90]);
    grid on
    drawnow
     F(i) = getframe(gcf);
     if (i==1 || mod(i,floor(0.25*loops))==0)
         tmpName = strrep (name,'_','-');
         tmpName = strrep (tmpName, '.', 'pnt');
         fullnamePic=strcat(path,'images/',tmpName,'-',num2str(timestamps(i)),'.png');
%         saveas(gcf,fullnamePic,'bmp'); %save first image
          export_fig(gcf, fullnamePic);
     end
end

vidObj = VideoWriter(fullname);
      open(vidObj);
      writeVideo(vidObj,F);
      close(vidObj);

close(figureHandle);


end

