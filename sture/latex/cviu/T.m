x=2:2:12;
MOTA=[36.8, 40.8, 42.5, 44.6, 43.6, 41.2];
% tau_a=[33.9, 33.9, 33.9, 36.6, 36.8, 36.8];
MOTP=[36.6, 36.5, 36.6, 36.6, 36.5, 36.4];
% tau_i=[35.3, 36.8, 36.3, 35.6, 34.8, 33.7];
% tau_o=[36.5, 36.5, 36.2, 36.0, 35.5, 34.4];
% tau_d=[36.8, 34.8, 36.7, 36.3, 36.3, 36.1];
max_tau = max([max(tau_s), max(tau_a), max(tau_t), max(tau_i), max(tau_o), max(tau_d)]);

grid
lineWidth = 4;
fontSize = 16;

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

xticks(0.0 : 0.2 : 1.0);
yticks(0.0 : 0.2 : 1.0);
xlabel('Parameter value','fontsize', fontSize, 'FontName','Times New Roman')   % 'FontWeight','bold',
ylabel('Normalized MOTA','fontsize', fontSize, 'FontName','Times New Roman')   % 'FontWeight','bold',
h=legend('\tau_s','\tau_a','\tau_t','\tau_i','\tau_o', '\tau_d', ...
    'location', 'northeastoutside');
set(h,'Fontsize', fontSize, 'FontName','Times New Roman');  %'FontWeight','bold',
set(gca,'FontSize', fontSize, 'LineWid',2);%设置坐标轴字体打下以及网格粗细

