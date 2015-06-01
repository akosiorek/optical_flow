
%run optical_flow first to save data to flow.mat
%then, run function, for example: movie_quivers('flow_quivers.avi')
function movieQuivers(name)
    load('flow.mat');

    figureHandle = figure('units','normalized','outerposition',[0 0 1 1]);
    loops = 294;
    F(loops) = struct('cdata',[],'colormap',[]);
    for i = 1:loops
        visualization(i,opticalFlowX, opticalFlowY, quantized)
%         view([0 90]);
        drawnow
        F(i) = getframe(gcf);
    end
    movie2avi(F, name, 'compression', 'none');
    close(figureHandle);    
end


function visualization(j, opticalFlowX, opticalFlowY, quantized)


% maxX=max(opticalFlowX(:));
% maxY=max(opticalFlowY(:));
% %for now, normalize the x- and y-components
% maxTotal=max(maxX, maxY);
% flowXNorm=opticalFlowX./maxTotal;
% flowYNorm=opticalFlowY/maxTotal;

maskedFlowX=opticalFlowX;
maskedFlowY=opticalFlowY;
quantizedOffset=size(quantized,1)-size(opticalFlowX,3);
mask=quantized(quantizedOffset+1:size(quantized,1));
maskArray=zeros(size(maskedFlowX));
for i=1:size(opticalFlowX,3)
    maskArray(:,:,i)=full(mask{i});
end
    maskedFlowX(maskArray==0)=0;
    maskedFlowY(maskArray==0)=0;

% figure
% hold on
[x,y] = meshgrid(1:1:128,1:1:128);
px = cos(x);
py = sin(y);
scaleFactor=10;
maskedFlowX(:,:,1);
quiver(x,y,maskedFlowX(:,:,j).*scaleFactor,maskedFlowY(:,:,j).*scaleFactor,'AutoScale','off');
% % quiver.LineWidth=1.25;
% quiver.MaxHeadSize=0.8;
end
