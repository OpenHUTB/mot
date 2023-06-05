% load fisheriris
rng default % for reproducibility

% 150*4 -> 150*2（减少数据的维度）：150个样本，每个样本包含4个属性
% Y = tsne(meas);
% gscatter(Y(:,1),Y(:,2), species)

% 只取前30个数据
num = 100;

load('vid_qf.mat')
load('ids.mat')

Y = tsne(vid_qf);

h1 = gscatter(Y(1:num, 1), Y(1:num, 2), vid_q_pids(1:num));
legend('off')  % 不显示图例
set(gca,'ytick',[])  % 隐藏y轴刻度
set(gca,'xtick',[])  % 隐藏x轴刻度

% gscatter(Y(1:num, 1), Y(1:num, 2), vid_q_pids(1:num)', 'doleg', 'off')