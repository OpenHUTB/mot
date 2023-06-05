%% init some directory
projectHome = fileparts(mfilename('fullpath'));
if ispc
    dataPath = fullfile(fileparts(projectHome), 'data');
else
    dataPath = fullfile(projectHome, 'data');
end


addpath(genpath(fullfile(projectHome, 'results', 'mot17-SDP')));
