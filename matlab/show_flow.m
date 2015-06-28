function [f] = show_flow(title_string,slice_nr,opticalFlowX,opticalFlowY, quantized)
if nargin <4
    load('flow.mat');
    
    %  else
    %           opticalFlowX=flow_file(:,:,1);
    %      opticalFlowY=flow_file(:,:,2);
end

if nargin <2
    j=1;
else
    j=slice_nr;
end



    maskedFlowX=opticalFlowX;
    maskedFlowY=opticalFlowY;
if nargin >4

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
[x,y] = meshgrid(1:1:size(opticalFlowX,1),1:1:size(opticalFlowX,2));
angles = atan(opticalFlowY./opticalFlowX);
% py = sin(y);


% f=quiver(x,y,maskedFlowX(:,:,j)',maskedFlowY(:,:,j)');
%Bigger quivers
scaleFactor=2.5;
f=quiver(x,y,maskedFlowX(:,:,j)'.*scaleFactor,maskedFlowY(:,:,j)'.*scaleFactor,'AutoScale','off');
    xlabel('x');
    ylabel('y');
    title_string=strrep(title_string,'_','-');
    title(title_string);
    



end