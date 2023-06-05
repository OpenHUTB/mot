function run_evaluation(varargin)
% run_evaluation('MOT17_SDP')
if nargin >= 1
    dataset = varargin{1};
else
    dataset = 'MOT17_SDP';  % default validate dataset 
end

%% Environment init
dbclear if error
projectHome = fileparts( fileparts( mfilename('fullpath') ) );
addpath(projectHome);

% eval(['cd ' projectHome]);

%% Run tracking
% % !/home/d/anaconda2/envs/deepMOT/bin/python /home/d/workspace/MOT/DMAN_MOT/calculateSimilarity.py &



% run matlab tracking client
% DMAN_demo


% time statistics
% start:  10:25 - 12:52
% total time: 2h 33min

% reduce learning rate to re-train

%%
% benchmarkDir = 'data/MOT16/train/';
% allMets = evaluateTracking('c5-train.txt', 'results/', benchmarkDir);

benchmarkDir = fullfile(projectHome, 'data', dataset, 'train/');
resDir = fullfile(projectHome, 'results/');

if strcmp(dataset, 'MOT17_SDP') == 1
    seqmap = 'c7-train.txt';
else
    seqmap = 'c5-train.txt';
end
allMets = evaluateTracking(seqmap, resDir, benchmarkDir);

end
