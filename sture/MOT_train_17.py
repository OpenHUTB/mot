# modified file: MOT_train.py

from util import utils
import parser
from net import models
import sys
import random
from tqdm import tqdm
import numpy as np
import math
from util.loss import TripletLoss
from util.cmc import Video_Cmc

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
import torch.backends.cudnn as cudnn

cudnn.benchmark = True
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
torch.multiprocessing.set_sharing_strategy('file_system')


def validation(network,dataloader,args):
    network.eval()
    pbar = tqdm(total=len(dataloader),ncols=100,leave=True)  # progressbar
    pbar.set_description('Inference')

    right_sum = 0
    total_sum = 0
    with torch.no_grad():
        for c,data in enumerate(dataloader):
            seqs = data[0].cuda()
            label = data[1]
            cams = data[2]

            pred = network(seqs)#.cpu().numpy() #[xx,128]pred.cpu()

            batch_size = len(label)
            pred = pred.cpu().numpy()
            pred_label = np.zeros(batch_size)
            for i in range(batch_size):
                if pred[i, 1] > pred[i, 0]: # same
                    pred_label = 1
            right_sum += sum(pred_label == label.numpy())
            total_sum += batch_size

            pbar.update(1)
    pbar.close()

    network.train()

    return right_sum / total_sum


'''
python3 MOT_train_17.py 
小的学习率和小的步长
--train_txt ./MOT16_database/train_path.txt --train_info ./MOT16_database/train_info.npy  --batch_size 32 --test_txt ./MOT_database/test_path.txt  --test_info  ./MOT_database/test_info.npy   --query_info ./MARS_database/query_IDX.npy --n_epochs 20000 --lr 0.00001 --lr_step_size 2 --optimizer adam --ckpt ckpt_NL_0230 --log_path loss.txt --class_per_batch 1 --model_type resnet50_NL --num_workers 8 --track_per_class 4 --S 8 --latent_dim 2048 --temporal Done  --track_id_loss --non_layers  0 2 3 0
大的学习率0.0005
--train_txt ./MOT17_database/train_path.txt --train_info ./MOT17_database/train_info.npy  --batch_size 32 --test_txt ./MOT_database/test_path.txt  --test_info  ./MOT_database/test_info.npy   --query_info ./MARS_database/query_IDX.npy --n_epochs 20000 --lr 0.0005 --lr_step_size 50 --optimizer adam --ckpt ckpt_NL_0230 --log_path loss.txt --class_per_batch 1 --model_type resnet50_NL --num_workers 8 --track_per_class 4 --S 8 --latent_dim 2048 --temporal Done  --track_id_loss --non_layers  0 2 3 0
--train_txt ./MOT_database/train_path.txt --train_info ./MOT_database/train_info.npy  --batch_size 32 --test_txt ./MOT_database/test_path.txt  --test_info  ./MOT_database/test_info.npy   --query_info ./MARS_database/query_IDX.npy --n_epochs 2000 --lr 0.0001 --lr_step_size 50 --optimizer adam --ckpt ckpt_NL_0230 --log_path loss.txt --class_per_batch 1 --model_type resnet50_NL --num_workers 8 --track_per_class 4 --S 8 --latent_dim 2048 --temporal Done  --track_id_loss --non_layers  0 2 3 0
  
'''

if __name__ == '__main__':
    # Parse args
    args = parser.parse_args()

    # set transformation (H flip is inside dataset)
    train_transform = Compose(
        [Resize((256, 128)), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    test_transform = Compose(
        [Resize((256, 128)), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    print('Start dataloader...')
    train_dataloader = utils.Get_MOT16_train_DataLoader(args.train_txt, args.train_info, train_transform, shuffle=True,
                                                        num_workers=args.num_workers, \
                                                        S=args.S, track_per_class=args.track_per_class,
                                                        class_per_batch=args.class_per_batch)
    print('End dataloader...')

    # network = nn.DataParallel(models.CNN(args.latent_dim,model_type=args.model_type,num_class=num_class,non_layers=args.non_layers,stripes=args.stripes,temporal=args.temporal).cuda())
    modelPath = './ckpt/Similarity_MOT17.pth'
    network = torch.load(modelPath)
    # if args.load_ckpt is not None:
    #     state = torch.load(args.load_ckpt)
    #     network.load_state_dict(state,strict=False)
    # log
    os.system('mkdir -p %s' % (args.ckpt))
    f = open(os.path.join(args.ckpt, args.log_path), 'a')
    f.close()

    # Train loop
    # 1. Criterion
    criterion_triplet = TripletLoss('soft', True)

    critetion_id = nn.CrossEntropyLoss().cuda()
    # 2. Optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(network.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    else:
        optimizer = optim.Adam(network.parameters(), lr=args.lr, weight_decay=5e-5)
    if args.lr_step_size != 0:
        scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_step_size, 0.1)

    id_loss_list = []
    trip_loss_list = []
    track_id_loss_list = []
    best_acc = 0
    for e in range(args.n_epochs):
        print('epoch', e)

        total_id_loss = 0
        total_trip_loss = 0
        total_track_id_loss = 0
        pbar = tqdm(total=len(train_dataloader), ncols=100, leave=True)
        for i, data in enumerate(train_dataloader):
            seqs = data[0]  # .cuda()
            labels = data[1].cuda()
            B, T, C, H, W = seqs.shape
            sililarity = network(seqs)

            total_loss = critetion_id(sililarity, labels)  # Cross-entropy Loss
            #####################
            optimizer.zero_grad()
            total_loss.backward()  # total_loss = trip_loss + frame_id_loss + track_id_loss
            optimizer.step()
            pbar.update(1)
        pbar.close()

        if args.lr_step_size != 0:
            scheduler.step()

        print('%.4f' % total_loss)
        modelCheckpoint = os.path.join('ckpt', 'Similarity_MOT17_.pth')
        torch.save(network, modelCheckpoint)


'''
CMC: 0.7177, mAP : 0.5587
'''