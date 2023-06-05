#!/usr/bin/env python
import os
import numpy as np
import scipy.io as sio
import cv2
from os.path import expanduser
import socket

from NL.util import utils
from NL.net import models
from NL.util.cmc import Video_Cmc
import parser
import sys
import random
from tqdm import tqdm
import numpy as np
import math

import torch
import torch.nn as nn
from torchvision.transforms import Compose,ToTensor,Normalize,Resize
import torch.backends.cudnn as cudnn
cudnn.benchmark=True
import os
import parser

import torch.nn.functional as F

import models

# parser.add_argument('--dataset', type=str, default='test')
# parser.add_argument('--detector', type=str, default='DPM')
# args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# determine whether runing on MOT training set or test set
dataset = 'test' # 'train' or 'test'
detector = 'SDP' # DPM, FRCNN, SDP
# dataset = args.dataset
# detector = args.detector


torch.multiprocessing.set_sharing_strategy('file_system')

HEIGHT = 256  # Similarity Model Input Image height
WIDTH = 128

modelPath = './ckpt/Similarity.pth'
network = torch.load(modelPath)
network.eval()  # fix dropout and batch normalization

use_gpu = True
img_arch = 'img_resnet50'
vid_arch = 'vid_nonlocalresnet50'
print("Initializing model: {} and {}".format(vid_arch, img_arch))
vid_model = models.init_model(name=vid_arch)
img_model = models.init_model(name=img_arch)
print("Video model size: {:.5f}M".format(sum(p.numel() for p in vid_model.parameters())/1000000.0))
print("Image model size: {:.5f}M".format(sum(p.numel() for p in img_model.parameters())/1000000.0))
# resume = 'TKP/log-mars/best_model.pth.tar'
resume = 'log/best_model.pth.tar'
print("Loading checkpoint from '{}'".format(resume))
checkpoint = torch.load(resume)
vid_model.load_state_dict(checkpoint['vid_model_state_dict'])
img_model.load_state_dict(checkpoint['img_model_state_dict'])
if use_gpu:
    vid_model = vid_model.cuda()
    img_model = img_model.cuda()
vid_model.eval()  # not use BatchNormalization and Dropout
img_model.eval()
print('load weights done!')

def extract_vid_feature(model, vids, use_gpu):
    test_frames = 32
    n, c, f, h, w = vids.size()
    # assert(n == 1)

    feat = torch.FloatTensor()
    if use_gpu:
        vids = vids.cuda()
    output = model(vids)
    feat = output.data.cpu()
    feat = feat.mean(1)
    # for i in range(f):
    #     clip = vids[:, :, i, :, :]  # require 1 * 3 * 32 * 256 * 128
    #     if use_gpu:
    #         clip = clip.cuda()
    #     output = model(clip)  # Expected 5-dimensional input for 5-dimensional weight 64 3 1 7 7, but got 4-dimensional input of size [2, 3, 256, 128]
    #     output = output.data.cpu()
    #     feat = torch.cat((feat, output), 1)

    # feat = feat.mean(1)

    return feat

# communicate with the matlab program using the socket
host = '127.0.0.1' 
port = 65431 
socket_tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
socket_tcp.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # release port right now
socket_tcp.bind((host, port))
socket_tcp.listen(5)  # start listen connection from client
print('The python socket server is ready. Waiting for the signal from the matlab socket client ...')
connection, adbboxess = socket_tcp.accept()  # enter loop, accept request from client
try:
    home = expanduser("~")  # change PATH that contains '~' to user directory
    while 1:
        flag = connection.recv(1024)  # receive data from client
        flag = str(flag)
        # flag = 'client ok'
        if not flag:
            break
        elif flag.find('client ok') < 0 :  #  flag != 'client ok'
            print(flag)
        else:
            print(flag)  # receive "client ok" from client(matlab)
            mat = sio.loadmat('mot_py.mat') # saved by the matlab program (extract tracklet from matlab)
            seq_name = mat['seq_name'][0]  # .encode('ascii', 'ignore')
            traj_dir = mat['traj_dir'][0]  # .encode('ascii', 'ignore')
            frame_id = int(mat['frame_id_double'][0, 0])
            target_id = traj_dir.split('/')[-2]
            x_det = mat['bboxes']['x'][0, 0]
            y_det = mat['bboxes']['y'][0, 0]
            w_det = mat['bboxes']['w'][0, 0]
            h_det = mat['bboxes']['h'][0, 0]
            num_det = x_det.shape[0]
            time_steps = 8
            frame_path = 'data/MOT20/' + dataset + '/' + seq_name + '/img1/' + '{:06d}.jpg'.format(frame_id)
            img_frame = cv2.imread(frame_path)
            img_h, img_w, _ = img_frame.shape
            # img_frame = image.load_img(frame_path)
            # img_w = img_frame.size[0]
            # img_h = img_frame.size[1]
            subfiles = os.listdir(traj_dir)
            subfiles.sort()
            img_traj_list = []
            for subfile in subfiles:
                # img_traj_list.append(subfile)
                subfile = str(subfile, encoding="utf-8")  # solve byte is not euql with str
                if subfile[-3:] == 'jpg':  # byte is not equal
                    img_traj_list.append(subfile)
            num_traj = len(img_traj_list)
            if num_traj < time_steps:
                tmp_list = img_traj_list[::-1]
                while len(img_traj_list) < time_steps:
                    img_traj_list += tmp_list
                img_traj_list = img_traj_list[0:time_steps]
            else:
                gap = num_traj // time_steps
                mod = num_traj % time_steps
                tmp_list = img_traj_list
                img_traj_list = []
                for i in range(mod, num_traj, gap):  # TypeError: 'float' object cannot be interpreted as an integer
                    img_traj_list.append(tmp_list[i])
            data = torch.zeros(num_det, time_steps+1, 3, HEIGHT, WIDTH)  # Similarity model input: bs * (8+1) * 3*256*128 (last row is invalid)
            for j in range(num_det):  # all raws are the same
                for i in range(time_steps):  # the length of tracklet: 8
                    img = cv2.imread(traj_dir + img_traj_list[i])
                    img_traj = torch.tensor(cv2.resize(img, (WIDTH, HEIGHT)))
                    data[j, i, :, :, :] = img_traj.permute(2, 0, 1)
            prediction = np.zeros(num_det, dtype = np.float32)
            for i in range(num_det):  # traverse all detection result in current frame
                x1 = int(x_det[i, 0])
                y1 = int(y_det[i, 0])
                w = int(w_det[i, 0])
                h = int(h_det[i, 0])
                x2 = x1 + w
                y2 = y1 + h
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(img_w, x2)
                y2 = min(img_h, y2)

                imgCropped = img_frame[y1:y2, x1:x2]
                img_det = torch.tensor(cv2.resize(imgCropped, (WIDTH, HEIGHT)))
                data[i, time_steps, :, :, :] = img_det.permute(2, 0, 1)
                # data = torch.cat((data, data), 0)  # expand batch size from 1 to 2

            vids = data[:, 0:8, :, :, :].transpose(2, 1)  # 1*8*3*256*128 -> 1*3*8*256*128
            vid_feat = extract_vid_feature(vid_model, vids, use_gpu)

            imgs = data[:, 8, :, :, :]
            if use_gpu:
                imgs = imgs.cuda()
            imgs_feat = img_model(imgs).data.cpu()

            vid_feat_norm = F.normalize(vid_feat)
            imgs_feat_norm = F.normalize(imgs_feat)
            similarity = vid_feat_norm.mm(imgs_feat_norm.t()).numpy()  # cosine similarity
            print(similarity[0])

            # prediction = network(data)  # invalid argument 0: Tensors must have same number of dimensions: got 1 and 2 at /pytorch/aten/src/THC/generic/THCTensorMath.cu:62
            # prediction = prediction.cpu().detach().numpy()  # GPU tensor -> CPU tensor -> Variable -> numpy
            # prediction = np.delete(prediction, [num_det], axis=0)  # remove last zeros line
            # prediction = np.delete(prediction, [0], axis=1)  # get the same probility
            # prediction = prediction.reshape(1, num_det)  # transpose to row vector
            sio.savemat('similarity.mat', {'similarity': similarity[0]})  # the similarity value between detections and the current tracklet.( one double value )
            connection.sendall(bytes('server ok', encoding="utf-8"))  # send TCP data fully before return
            print('server ok')
finally:
    connection.close()
    socket_tcp.close()
    print('python server closed.')
