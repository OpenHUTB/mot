%% 
% Python:
% nohup /home/d/anaconda2/envs/deepMOT/bin/python calculateSimilarityMOT17.py 1>results/calculateSimilarityMOT17.log 2>calculateSimilarityMOT17.err &
% Matlab: 
% nohup matlab -nodisplay -nosplash < DMAN_demo.m 1>results/DMAN_demo.log 2>results/DMAN_demo.err &

clear
clc
warning('off')
dbstop if error

%% mat file for sharing data between the matlab and python programs
if exist('mot_py.mat', 'file')
    delete('mot_py.mat');
end

% start the socket client
global client_tcp
client_tcp = tcpip('127.0.0.1', 65431, 'Timeout', 60*10, 'OutputBufferSize',10240,'InputBufferSize',10240);
fopen(client_tcp);

addpath(genpath('ECO/'));
addpath(genpath('devkit/'));

opt = globals();
if ~exist(opt.results_dir, 'dir')
    mkdir(opt.results_dir)
end

% determine whether running on the training set or test set
is_train = false;

% training and testing pairs
seq_num = numel(opt.mot2d_train_seqs);
test_time = 0;
for seq_idx = 1:seq_num
    % load tracker from file
    if is_train
        seq_name = opt.mot2d_train_seqs{seq_idx};
    else
        seq_name = opt.mot2d_test_seqs{seq_idx};     
    end
    tracker_file = sprintf('init_tracker/%s_tracker.mat', opt.mot2d_train_seqs{seq_idx});
    object = load(tracker_file);
    tracker = object.tracker;
    fprintf('load tracker from file %s\n', tracker_file);
    tracker = ECO_params(tracker);   
    fprintf('Testing on sequence: %s\n', seq_name); 
    if exist(['img_traj/' seq_name]) == 7    % name is folder
        rmdir(['img_traj/' seq_name], 's');
    end
    tic;
    if is_train
        track_seq(seq_idx, 'train', tracker, opt);  % call sequence: track_seq() -> MOT_track() -> track_frame() -> mot_py.mat
    else
        track_seq(seq_idx, 'test', tracker, opt);
    end
    test_time = test_time + toc;
end
fprintf('Total time for testing: %f\n', test_time);
pause(1)
fclose(client_tcp);
delete(client_tcp);

%%
% Test in "train/MOT16"
% Total time for testing: 8319.893212
