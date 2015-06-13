function show_flow(slice_nr,opticalFlowX,opticalFlowY, quantized)
if nargin <2
    load('flow.mat');
    
    %  else
    %           opticalFlowX=flow_file(:,:,1);
    %      opticalFlowY=flow_file(:,:,2);
end

if nargin <1
    j=1;
else
    j=slice_nr;
end



    maskedFlowX=opticalFlowX;
    maskedFlowY=opticalFlowY;
if nargin >3

    quantizedOffset=size(quantized,1)-size(opticalFlowX,3);
    mask=quantized(quantizedOffset+1:size(quantized,1));
    maskArray=zeros(size(maskedFlowX));
    for i=1:size(opticalFlowX,3)
        maskArray(:,:,i)=full(mask{i});
    end
    maskedFlowX(maskArray==0)=0;
    maskedFlowY(maskArray==0)=0;

end


% figure
% hold on
[x,y] = meshgrid(1:1:size(opticalFlowX,1),1:1:size(opticalFlowX,2 ));
px = cos(x);
py = sin(y);
% scaleFactor=5;
% f=quiver(x,y,opticalFlowX(:,:,j).*scaleFactor,opticalFlowY(:,:,j).*scaleFactor,'AutoScale','off');
scaleFactor=1;
% f=quiver(x,y,maskedFlowX(:,:,j).*scaleFactor,maskedFlowY(:,:,j).*scaleFactor)';%,'AutoScale','off');
merged=maskedFlowX(:,:,j)+maskedFlowY(:,:,j);   
%     s = size(merged);
%     x = 1:s(2);
%     y = 1:s(1);
merged=merged';
mesh(x, y, merged);

        view([0 90]);
%     xlim([-5 size(opticalFlowX,2)+5]);
%     ylim([-5 size(opticalFlowY,1)+5]);
%         drawnow

end