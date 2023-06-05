figure('visible','off');

%%
x=0:0.2:1.0;
tau_s=[36.8, 36.8, 32.5, 16.2, 15.6, 15.2];
tau_a=[33.9, 33.9, 33.9, 36.6, 36.8, 36.8];
tau_t=[36.6, 36.5, 36.6, 36.6, 36.5, 36.4];
tau_i=[35.3, 36.8, 36.3, 35.6, 34.8, 33.7];
tau_o=[36.5, 36.5, 36.2, 36.0, 35.5, 34.4];
tau_d=[36.8, 34.8, 36.7, 36.3, 36.3, 36.1];
max_tau = max([max(tau_s), max(tau_a), max(tau_t), max(tau_i), max(tau_o), max(tau_d)]);

grid
lineWidth = 4;
fontSize = 14;

tau_s = tau_s ./ max_tau;
hold on;
plot(x,tau_s, ...
    'Color', [65/255,111/255,166/255], ...
    'LineWidth', lineWidth);
% 's-',
% 'MarkerFaceColor', [65/255,111/255,166/255]
% 'MarkerEdgeColor',[65/255,111/255,166/255],
% 'MarkerSize',2, ...

tau_a = tau_a ./ max_tau;
hold on
plot(x,tau_a, ...
    'Color', [168/255,66/255,63/255], ...
    'LineWidth', lineWidth);


tau_t = tau_t ./ max_tau;
hold on
plot(x,tau_t, ...
    'Color', [134/255,164/255,74/255], ...
    'LineWidth', lineWidth);


tau_i = tau_i ./ max_tau;
hold on
plot(x, tau_i, ...
    'Color', [110/255,84/255,141/255], ...
    'LineWidth', lineWidth);

tau_o = tau_o ./ max_tau;
hold on
plot(x, tau_o, ...
    'Color', [61/255,150/255,174/255], ...
    'LineWidth', lineWidth);

tau_d = tau_d ./ max_tau;
hold on
plot(x, tau_d, ...
    'Color', [218/255,129/255,55/255], ...
    'LineWidth', lineWidth);

xticks(0.0 : 0.1 : 1.0);
yticks(0.0 : 0.1 : 1.0);
xlabel('参数值','fontsize', fontSize, 'FontName','Monospaced')   % 'FontWeight','bold',
ylabel('正则化的MOTA','fontsize', fontSize, 'FontName','Monospaced')   % 'FontWeight','bold',
h=legend('\tau_s','\tau_a','\tau_t','\tau_i','\tau_o', '\tau_d', ...
    'location', 'northeastoutside');
set(h,'Fontsize', fontSize, 'FontName','Times New Roman');  %'FontWeight','bold',
set(gca,'FontSize', fontSize, 'LineWid',2);%设置坐标轴字体打下以及网格粗细


%% export 
filename = 'parameter.pdf'; % 设定导出文件名
width=700;      %宽度，像素数
height=400;     %高度
left=200;       %距屏幕左下角水平距离
bottem=100;     %距屏幕左下角垂直距离
set(gcf,'position',[left,bottem,width,height])

addpath('../../../../ai/tools/matlab');
savePDF(pwd, filename);  % crop and save as pdf
close(gcf)
