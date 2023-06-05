function ts_plot(data)
% 绘制PCA成分数和神经相似性之间的关系图
% 原来位于dorsal/utils/ts_plot.m

data = load('comp_sim.mat');  % 加载已经生成的数据用于绘图(原来位于:../result/tmp/comp_sim.mat)
data = data.comp_sim;

orange = [ 0.91,0.41,0.17];
blue = [0,0.4470,0.7410];
figure('DefaultAxesFontSize',14);
ax = gca;

hold on
h1 = plot(data(:,1), data(:,2), '-', 'color',orange, 'linewidth',3, 'DisplayName','rmse');
h1 = plot(data(:,1), data(:,3), '-', 'color',blue, 'linewidth',3, 'DisplayName','rmse');
xlabel('PCA 成分数');
ylabel('正则化的值 [0, 1]');

ylim([-0.1 1.1])
legend('神经相似性', 'p 值', 'Location', 'east');

box on;
grid on;

ax.GridLineStyle = '-.';
set(gcf, 'PaperSize', [8 6]);
set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperPosition', [0 0 8 6]);

set(gca, 'XTick', 0:20:160)

%% 显示点的值
x_1 = 11;
y_1 = data(x_1, 2);
text(x_1, y_1+0.05, ['（',num2str(x_1),'，', num2str(sprintf("%0.2f", y_1)),'）'],'color', orange)  % 上移0.05

x_2 = 90;
y_2 = data(x_2, 2);
text(x_2, y_2-0.05, ['（',num2str(x_2),'，',num2str(sprintf("%0.2f", y_2)),'）'],'color', orange)  % 下移0.05

p_1 = 11;
y_p_1 = data(p_1, 3);
text(p_1, y_p_1+0.05, ['（',num2str(p_1),'，',num2str(sprintf("%0.2f", y_p_1)),'）'],'color', blue)  % 上移0.05

p_2 = 90;
y_p_2 = data(p_1, 3);
text(p_2, y_p_2-0.07, ['（',num2str(p_2),'，',num2str(sprintf("%0.2f", y_p_2)),'）'],'color', blue)  % 上移0.05

%%
saveas(gcf,'comp_sim.pdf','pdf');

close all
end