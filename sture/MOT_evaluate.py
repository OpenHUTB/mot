from util import utils
from util.cmc import Video_Cmc
from net import models
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

os.environ['CUDA_VISIBLE_DEVICES']='0'
torch.multiprocessing.set_sharing_strategy('file_system')


'''
python evaluate.py 
--test_txt ./MARS_database/test_path.txt  --test_info  ./MARS_database/test_info.npy   --query_info ./MARS_database/query_IDX.npy --batch_size 32 --model_type resnet50_NL --num_workers 8  --S 8 --latent_dim 2048 --temporal Done  --non_layers  0 2 3 0 --load_ckpt ./ckpt/Similarity.pth
--test_txt ./MARS_database/test_path.txt  --test_info  ./MARS_database/test_info.npy   --query_info ./MARS_database/query_IDX.npy --batch_size 32 --model_type resnet50_NL --num_workers 8  --S 8 --latent_dim 2048 --temporal Done  --non_layers  0 2 3 0 --load_ckpt ./ckpt/NVAN.pth
'''


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

if __name__ == '__main__':
    #Parse args
    print(torch.__version__) # 1.2.0
    args = parser.parse_args()

    test_transform = Compose([Resize((256,128)),ToTensor(),Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])
    print('Start dataloader...')
    num_class = 625
    test_dataloader = utils.Get_MOT_test_DataLoader(args.test_txt,args.test_info,args.query_info,test_transform,batch_size=args.batch_size,
                                                 shuffle=False,num_workers=args.num_workers,S=args.S,distractor=True)
    print('End dataloader...')



    # latent_dim = 2048
    # non_layers = [0, 2, 3, 0]
    # network = nn.DataParallel(models.Similarity(args.latent_dim, model_type=args.model_type, num_class=num_class, \
    #                                      non_layers=args.non_layers, stripes=args.stripes, temporal=args.temporal).cuda())
    modelPath = './ckpt/Similarity.pth'
    network = torch.load(modelPath)


    # with torch.no_grad():
    #     for c,data in enumerate(test_dataloader):
    #         seqs = data[0].cuda()
    #         label = data[1]
    #         cams = data[2]
    #         if args.model_type != 'resnet50_s1':
    #             B,C,H,W = seqs.shape
    #             seqs = seqs.reshape(B//args.S,args.S,C,H,W)
    #
    #         detection = seqs[:, 1, :, :]  # 8->9
    #         detection = detection.reshape(32, 1, C, H, W)
    #         seqs = torch.cat((seqs, detection), 1)  # 32* 9 * 3*256*128
    #         feat = network(seqs)  #input: 32 * 9 * 3*256*128;    feat: 32 * 2048 -> 32 * 2 (same, different)
    #         print(feat)

            # torch.save(network, modelPath)
            # network = torch.load(modelPath)
            # network.eval()  # fix dropout and batch normalization





    # network = nn.DataParallel(models.CNN(args.latent_dim, model_type=args.model_type, num_class=num_class, \
    #                                      non_layers=args.non_layers, stripes=args.stripes, temporal=args.temporal).cuda())

    # if args.load_ckpt is  None:
    #     print('No ckpt!')
    #     exit()
    # else:
    #     state = torch.load(args.load_ckpt)
    #     network.load_state_dict(state, strict=True)

        # network.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(args.load_ckpt).items()})

        # try:
        #     state_dict = torch.load(args.load_ckpt)
        #     from collections import OrderedDict
        #
        #     new_state_dict = OrderedDict()
        #     for k, v in state_dict.items():
        #         name = 'module.' + k  # add `module.`
        #         new_state_dict[name] = v
        #     # load params
        #     # model.load_state_dict(new_state_dict)
        #     network.load_state_dict(new_state_dict)
        # except Exception as e:
        #     print(e)

    acc = validation(network, test_dataloader, args)

    print('acc : %.4f'%(acc))

'''
NVAN in MARS 
CMC : 0.8995 , mAP : 0.8275

Model init
acc : 0.5008
'''

