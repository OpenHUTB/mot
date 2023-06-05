

figure('visible','off');
%%
% [X,Y] = meshgrid(-8:.5:8);
% R = sqrt(X.^2 + Y.^2) + eps;
% Z = sin(R)./R;
% mesh(X,Y,Z)

axis
[X, Y] = meshgrid(0 : 0.2 : 1.0);
Z = [30.0,  33.9,   27.0,  16.0,  17.0,  15.0;
     27.0,  33.9,   27.3,  16.5,  18.0,  17.0;
     27.0,  33.9,   27.3,  17.4,  18.0,  17.0;
     33.5,  36.6,   32.0,  15.6,  17.0,  22.0;
     33.0,  36.8,   32.5,  33.0,  33.0,  17.0;
     36.2,  33.0,   32.0,  33.0,  29.0,  17.0];
 
% Z = mapminmax(Z-min(min(Z)), 0, 1);
Z = Z ./ max( max(Z) );
surf(X, Y, Z);

xlabel('\tau _a', 'Fontname', 'Times New Roman');
ylabel('\tau _s', 'Fontname', 'Times New Roman');
% set(gca,'Fontname','Monospaced');
zlabel('正则化的MOTA', 'Fontname', 'Monospaced');
xticks(0.0 : 0.2 : 1.0);
yticks(0.0 : 0.2 : 1.0);

% 0,0,0,0,0,0
% 0,0,0,0,0,0
% 0,0,0,0,0,0
% 0,36.6,0,0,0,0
% 0,0,0,0,0,0
% 0,0,0,0,0,0

%% export 
view([-8, 34]);  % change view point
filename = 'grid_search.pdf'; % 设定导出文件名
addpath('../../../../ai/tools/matlab');
savePDF(pwd, filename);  % crop and save as pdf
close(gcf)
