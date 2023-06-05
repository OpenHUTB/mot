
:: todo
:: 命令行构建vs工程

:: 必须先切换到exe当前目录中执行
:: 工程在windows本地目录
:: cd D:\dong\ai\tools\latex\viso2pdf\viso2pdf\bin\Debug\

:: 需要先构建工程
:: 工程在远程服务器目录
:: 复制到该目录才能运行生成和裁剪命令？
:: cd Z:\data3\dong\ai\tools\latex\viso2pdf\viso2pdf\bin\Debug




:: 使用::进行注释
viso2pdf.exe mot\tracking\figures\C1Fig C:\texlive\2016\bin\win32\pdfcrop.exe

viso2pdf.exe mot\tracking\figures\C2Fig C:\texlive\2016\bin\win32\pdfcrop.exe

viso2pdf.exe mot\tracking\figures\C3Fig C:\texlive\2016\bin\win32\pdfcrop.exe

viso2pdf.exe mot\tracking\figures\C4Fig C:\texlive\2016\bin\win32\pdfcrop.exe

viso2pdf.exe mot\tracking\figures\C5Fig C:\texlive\2016\bin\win32\pdfcrop.exe

viso2pdf.exe mot\tracking\figures\C6Fig C:\texlive\2016\bin\win32\pdfcrop.exe


:: MATLAB生成PDF文件
:: matlab -nodesktop -nosplash -r run('D:/dong/mot/tracking/figures/C3Fig/grid_search_Chinese_export');run('D:/dong/mot/tracking/figures/C3Fig/hyperparameter_Chinese_export')

