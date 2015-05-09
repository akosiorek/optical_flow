function F = filter_animation( )
%FILTER_ANIMATION Summary of this function goes here
%   Detailed explanation goes here

tdf = 200;
cstFilter = cstf(15,1,tdf);

loops = tdf;
F(loops) = struct('cdata',[],'colormap',[]);
figure('units','normalized','outerposition',[0 0 1 1]);
for j = 1:loops
    subplot(1,2,2);
    surf(cstFilter(:,:,j));
    view([40 0]);
    xlim([0 31]);
    ylim([0 31]);
    zlim([-0.015 0.015]);
    
    subplot(1,2,1);
    surf(cstFilter(:,:,j));
    view([70 18]);
    xlim([0 31]);
    ylim([0 31]);
    zlim([-0.015 0.015]);
    drawnow
    F(j) = getframe(gcf);
end

end

