
base1_1 = 34.1;
base1_2 = 44.5;
base2_1 = 38.3;
base2_2 = 44.5;


y = [base1_1, base1_2-base1_1; base2_1, base2_2-base2_1];
x = 1:size(y, 2);
%% 
% Display Stacked Bars
bar(x, y, 'stacked');

set(gca,'XTickLabel',{'B1', 'B2'});%给坐标加标签 

set(gca,'YLim',[30 48]); %Y轴的数据显示范围
set(gca,'YTick', 30:5:50);%设置要显示坐标刻度



title('MOTA');
