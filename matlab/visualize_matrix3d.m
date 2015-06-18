function [f] = visualize_matrix3d(matrix3d, framepause)


[x,y] = meshgrid(1:1:size(matrix3d,2),1:1:size(matrix3d,1));
% x=1:size(matrix3d,1);
% y=1:size(matrix3d,2);
figureHandle = figure('units','normalized','outerposition',[0 0 1 1]);
for index=1:size(matrix3d,3)
    f= mesh(x, y, matrix3d(:,:,index));
    xlim([-5 size(matrix3d,2)+5]);
    ylim([-5 size(matrix3d,1)+5]);
%     zlim([-3 3]);
%             view([0 90]);
    xlabel('x');
    ylabel('y');
    zlabel('response');    
    
    title_string = sprintf(strcat('visualization of ', inputname(1)));
    title(title_string);
        drawnow
        pause(framepause);
        delete(f)
end
close(figureHandle);

end