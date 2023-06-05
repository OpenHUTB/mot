%% Rename the MOT17 tracking result of different detector
projectHome = fileparts( fileparts( mfilename('fullpath') ) );
resDir = fullfile(projectHome, 'results', 'mot17-SDP');
resFiles = dir(fullfile(resDir,'MOT17-*'))


eval(['cd ' resDir]);
