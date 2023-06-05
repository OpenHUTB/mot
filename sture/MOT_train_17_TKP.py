# modified file: MOT_train.py
import argparse
import time
import datetime

from PIL import Image

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
from torch.optim import lr_scheduler

cudnn.benchmark = True
import os
import os.path as osp
import cv2

from matplotlib import pyplot as plt

import utils.data_manager as data_manager
from utils.video_loader import VideoDataset, ImageDataset
import transforms.spatial_transforms as ST
import transforms.temporal_transforms as TT
import torchvision.transforms as T
import models
from utils.losses import FeatureBasedTKP, SimilarityBasedTKP, HeterogeneousTripletLoss
from utils.utils import AverageMeter, Logger, save_checkpoint
from utils.eval_metrics import evaluate
from utils.samplers import RandomIdentitySampler

from utils.gpu import get_gpu

import torch.nn.functional as F
import torchvision.transforms as transforms

os.environ['CUDA_VISIBLE_DEVICES'] = get_gpu()
# os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
# memory_gpu = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
# torch.cuda.set_device(np.argmax(memory_gpu))
# os.system('rm tmp')


torch.multiprocessing.set_sharing_strategy('file_system')




############################################################################################
i=0##testing in what
from torchvision import models as torch_models
resnet = torch_models.resnet50(pretrained=True)#这里单独加载一个包含全连接层的resnet50模型
image = []
class FeatureExtractor():
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
    	self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules['base']._modules.items():##resnet50没有.feature这个特征，直接删除用就可以。
            x = module(x)
            #print('name=',name)
            #print('x.size()=',x.size())
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
            #print('outputs.size()=',x.size())
        #print('len(outputs)',len(outputs))
        return outputs, x

class ModelOutputs():
	""" Class for making a forward pass, and getting:
	1. The network output.
	2. Activations from intermeddiate targetted layers.
	3. Gradients from intermeddiate targetted layers. """
	def __init__(self, model, target_layers,use_cuda):
		self.model = model
		self.feature_extractor = FeatureExtractor(self.model, target_layers)
		self.cuda = use_cuda
	def get_gradients(self):
		return self.feature_extractor.gradients

	def __call__(self, x):
		target_activations, output  = self.feature_extractor(x)
		# output = output.view(output.size(0), -1)
		#print('classfier=',output.size())
		if self.cuda:
			output = output.cpu()
			output = resnet.fc(output).cuda()##这里就是为什么我们多加载一个resnet模型进来的原因，因为后面我们命名的model不包含fc层，但是这里又偏偏要使用。#
		else:
			output = resnet.fc(output)##这里对应use-cuda上更正一些bug,不然用use-cuda的时候会导致类型对不上,这样保证既可以在cpu上运行,gpu上运行也不会出问题.
		return target_activations, output

def preprocess_image(img):
	means=[0.485, 0.456, 0.406]
	stds=[0.229, 0.224, 0.225]

	preprocessed_img = img.copy()[: , :, ::-1]
	for i in range(3):
		preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
		preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
	preprocessed_img = \
		np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
	preprocessed_img = torch.from_numpy(preprocessed_img)
	preprocessed_img.unsqueeze_(0)
	input = preprocessed_img
	input.requires_grad = True
	return input


def show_cam_on_image(img, mask,name):
	heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
	heatmap = np.float32(heatmap) / 255
	cam = heatmap + np.float32(img)
	cam = cam / np.max(cam)
    # plt.imshow( np.uint8(255 * cam)   )
	cv2.imwrite("results/cam_{}.jpg".format(name), np.uint8(255 * cam))


class GradCam:
	def __init__(self, model, target_layer_names, use_cuda):
		self.model = model
		self.model.eval()
		self.cuda = use_cuda
		if self.cuda:
			self.model = model.cuda()

		self.extractor = ModelOutputs(self.model, target_layer_names, use_cuda)

	def forward(self, input):
		return self.model(input)

	def __call__(self, input, index = None):
		if self.cuda:
			features, output = self.extractor(input.cuda())
		else:
			features, output = self.extractor(input)

		if index == None:
			index = np.argmax(output.cpu().data.numpy())

		one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
		one_hot[0][index] = 1
		one_hot = torch.Tensor(torch.from_numpy(one_hot))
		one_hot.requires_grad = True
		if self.cuda:
			one_hot = torch.sum(one_hot.cuda() * output)
		else:
			one_hot = torch.sum(one_hot * output)

		self.model.zero_grad()##features和classifier不包含，可以重新加回去试一试，会报错不包含这个对象。
		#self.model.zero_grad()
		one_hot.backward(retain_graph=True)##这里适配我们的torch0.4及以上，我用的1.0也可以完美兼容。（variable改成graph即可）

		grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()
		#print('grads_val',grads_val.shape)
		target = features[-1]
		target = target.cpu().data.numpy()[0, :]

		weights = np.mean(grads_val, axis = (2, 3))[0, :]
		#print('weights',weights.shape)
		cam = np.zeros(target.shape[1 : ], dtype = np.float32)
		#print('cam',cam.shape)
		#print('features',features[-1].shape)
		#print('target',target.shape)
		for i, w in enumerate(weights):
			cam += w * target[i, :, :]

		cam = np.maximum(cam, 0)
		cam = cv2.resize(cam, (224, 224))
		cam = cam - np.min(cam)
		cam = cam / np.max(cam)
		return cam

class GuidedBackpropReLUModel:
	def __init__(self, model, use_cuda):
		self.model = model#这里同理，要的是一个完整的网络，不然最后维度会不匹配。
		self.model.eval()
		self.cuda = use_cuda
		if self.cuda:
			self.model = model.cuda()
		for module in self.model.named_modules():
			module[1].register_backward_hook(self.bp_relu)

	def bp_relu(self, module, grad_in, grad_out):
		if isinstance(module, nn.ReLU):
			return (torch.clamp(grad_in[0], min=0.0),)
	def forward(self, input):
		return self.model(input)

	def __call__(self, input, index = None):
		if self.cuda:
			output = self.forward(input.cuda())
		else:
			output = self.forward(input)
		if index == None:
			index = np.argmax(output.cpu().data.numpy())
		#print(input.grad)
		one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
		one_hot[0][index] = 1
		one_hot = torch.from_numpy(one_hot)
		one_hot.requires_grad = True
		if self.cuda:
			one_hot = torch.sum(one_hot.cuda() * output)
		else:
			one_hot = torch.sum(one_hot * output)
		#self.model.classifier.zero_grad()
		one_hot.backward(retain_graph=True)
		output = input.grad.cpu().data.numpy()
		output = output[0,:,:,:]

		return output
############################################################################################


def transform_invert(img_, transform_train):
    """
    将data 进行反transfrom操作
    :param img_: tensor
    :param transform_train: torchvision.transforms
    :return: PIL image
    """
    if 'Normalize' in str(transform_train):
        norm_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), transform_train.transforms))
        mean = torch.tensor(norm_transform[0].mean, dtype=img_.dtype, device=img_.device)
        std = torch.tensor(norm_transform[0].std, dtype=img_.dtype, device=img_.device)
        img_.mul_(std[:, None, None]).add_(mean[:, None, None])

    img_ = img_.transpose(0, 2).transpose(0, 1)  # C*H*W --> H*W*C
    img_ = np.array(img_) * 255

    if img_.shape[2] == 3:
        img_ = Image.fromarray(img_.astype('uint8')).convert('RGB')
    elif img_.shape[2] == 1:
        img_ = Image.fromarray(img_.astype('uint8').squeeze())
    else:
        raise Exception("Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(img_.shape[2]))

    return img_


def extract_vid_feature(model, vids, use_gpu):
    test_frames = 32
    n, c, f, h, w = vids.size()

    feat = torch.FloatTensor()
    if use_gpu:
        vids = vids.cuda()
    output = model(vids)
    feat = output.data.cpu()
    feat = feat.mean(1)

    return feat


def validation(vid_model, img_model, classifier, dataloader):
    vid_model.eval()
    img_model.eval()
    classifier.eval()
    # pbar = tqdm(total=len(dataloader),ncols=100,leave=True)  # progressbar
    # pbar.set_description('Inference')

    right_sum = 0
    total_sum = 0
    is_second = False
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            if is_second is False:
                tmp_data = data
                is_second = True
                continue
            if data[1][0] == tmp_data[1][0]:  # and data[0].size(1) != tmp_data[0].size(1):  # remove same label data
                continue
            # seq_1 = data[0][:, 0:4, :, :, :]
            # seq_2 = tmp_data[0][:, 0:4, :, :, :]

            vids = data[0][:, 0:4, :, :, :].transpose(2, 1)
            imgs = data[0][:, 4, :, :, :]
            imgs_neg = tmp_data[0][:, 4, :, :, :]
            label = data[1]

            vid_feat = extract_vid_feature(vid_model, vids, use_gpu)
            if use_gpu:
                imgs = imgs.cuda()
                imgs_neg = imgs_neg.cuda()
            imgs_feat = img_model(imgs).data.cpu()
            imgs_neg_feat = img_model(imgs_neg).data.cpu()

            vid_feat_norm = F.normalize(vid_feat)
            imgs_feat_norm = F.normalize(imgs_feat)
            imgs_neg_feat_norm = F.normalize(imgs_neg_feat)
            similarity = vid_feat_norm.mm(imgs_feat_norm.t()).numpy()  # cosine similarity
            disimilarity = vid_feat_norm.mm(imgs_neg_feat_norm.t()).numpy()
            if i % 40 == 0:
                print('Similarity: {}'.format(np.mean(similarity)))
                print('Disimilarity: {}'.format(np.mean(disimilarity)))
                print()

    #         pbar.update(1)
    # pbar.close()

    vid_model.train()
    img_model.train()
    classifier.train()

    return similarity[0]


def train(epoch, vid_model, img_model, classifier, criterion, criterion_tkp_f, criterion_tkp_d, criterion_i2v,
          optimizer, trainloader, use_gpu, train_transform, args):
    batch_vid_loss = AverageMeter()
    batch_img_loss = AverageMeter()
    batch_TKP_F_loss = AverageMeter()
    batch_TKP_D_loss = AverageMeter()
    batch_i2v_loss = AverageMeter()
    batch_v2i_loss = AverageMeter()
    batch_i2i_loss = AverageMeter()
    batch_v2v_loss = AverageMeter()
    batch_vid_corrects = AverageMeter()
    batch_img_corrects = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    vid_model.train()
    img_model.train()
    classifier.train()

    end = time.time()
    is_second = False
    for i, data in enumerate(trainloader):  # batch_idx, (vids, pids, _)
        if i % 20 == 0: print(i)
        # print(i)  # crash: 334, 113. shuffle=False: 414, 414
        if is_second is False:
            tmp_data = data
            is_second = True
            continue
        if data[1][0] == tmp_data[1][0]:  # and data[0].size(1) != tmp_data[0].size(1):  # remove same label data
            continue
        if args.debug:
            # img = transform_invert(data[0][0, 0, :, :, :], train_transform)
            # plt.imshow(img)
            # plt.show()
            # plt.pause(0.5)
            # plt.close()
            viz = visdom.Visdom()
            viz.close()
            batch_size = data[0].size(0)
            for i in range(batch_size):
                viz.images(data[0][i, :, :, :, :],
                           opts={
                               'title': str(data[1][i].numpy())
                           })
            tmp_data_batch_size = tmp_data[0].size(0)
            for i in range(tmp_data_batch_size):
                viz.images(tmp_data[0][i, :, :, :, :],
                           opts={
                               'title': str(tmp_data[1][i].numpy())
                           })
            # viz.images(data[0][0,:,:,:,:],
            #            opts={
            #                'title': 'data:[4,5,3,256,128]'
            #            })

            pass
        seq_1 = data[0][:, 0:4, :, :, :]
        seq_2 = tmp_data[0][:, 0:4, :, :, :]
        # labels_1 = data[1]
        # labels_2 = tmp_data[1]
        seqs = torch.cat([seq_1, seq_2], dim=0)
        labels = torch.cat([data[1], tmp_data[1]], dim=0).cuda()
        is_second = False
        # seqs = data[0]
        # labels = data[1]
        pids = labels  # TODO: replace with true identity label

        vids = seqs[:, 0:4, :, :, :].transpose(2, 1)  # permute(2,1,0) vs transpose(0,2)
        b, c, t, h, w = vids.size()
        img_pids = pids.unsqueeze(1).repeat(1, t).view(-1)  # 8 -*4-> 32

        if use_gpu:
            vids, pids, img_pids = vids.cuda(), pids.cuda(), img_pids.cuda()

        # measure data loading time
        data_time.update(time.time() - end)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        vid_features, frame_features = vid_model(vids)  # all frame (contains in video) features: (8*4)*2048
        vid_outputs = classifier(vid_features)  # predicted video label: 8*625

        imgs = vids.permute(0, 2, 1, 3, 4).contiguous().view(b * t, c, h, w)  # imgs: 32*3*256*128
        img_features = img_model(imgs)  # image feature: 32*2048
        img_outputs = classifier(img_features)  # predicted image label: 32*625

        if args.debug:
            import cv2

            img_path = '/home/d/workspace/MOT/DMAN_MOT/MOT16_database/MOT16-09/4/000409F0446.jpg'
            pedestrian_img = cv2.imread(img_path)

            model = torch_models.resnet50(
                pretrained=True)  # 这里相对vgg19而言我们处理的不一样，这里需要删除fc层，因为后面model用到的时候会用不到fc层，只查到fc层之前的所有层数。
            del model.fc
            print(model)
            # modules = list(resnet.children())[:-1]
            # model = torch.nn.Sequential(*modules)

            # print(model)
            grad_cam = GradCam(model, \
                               target_layer_names=["layer4"],
                               use_cuda=True)  ##这里改成layer4也很简单，我把每层name和size都打印出来了，想看哪层自己直接嵌套就可以了。（最后你会在终端看得到name的）
            image_path = './study/pytorch-gradcam-resnet50/examples/both.png'
            img = cv2.imread(img_path)
            img = np.float32(cv2.resize(img, (224, 224))) / 255
            input = preprocess_image(img)
            input.required_grad = True
            print('input.size()=', input.size())
            target_index = None
            # mask = grad_cam(input, target_index)
            # show_cam_on_image(img, mask, 0)


            img_cam = GradCam(img_model, \
                              target_layer_names=["7"],
                              use_cuda=True)  # 7, layer4
            img_mask = img_cam(input, target_index)  # {RuntimeError}size mismatch, m1: [1 x 401408], m2: [2048 x 1000] at /pytorch/aten/src/TH/generic/THTensorMath.cpp:197
            show_cam_on_image(img, img_mask, 0)

            print('end cam')

            # x = os.walk(image_path)
            # for root, dirs, filename in x:
            #     # print(type(grad_cam))
            #     print(filename)
            # for s in filename:
            #     image.append(cv2.imread(image_path + s, 1))
            # # img = cv2.imread(filename, 1)
            # for img in image:
            #     img = np.float32(cv2.resize(img, (224, 224))) / 255
            #     input = preprocess_image(img)
            #     input.required_grad = True
            #     print('input.size()=', input.size())
            #     # If None, returns the map for the highest scoring category.
            #     # Otherwise, targets the requested index.
            #     target_index = None
            #
            #     mask = grad_cam(input, target_index)
            #     i = i + 1
            #     show_cam_on_image(img, mask, i)


        # compute loss
        vid_loss = criterion(vid_outputs, pids)  # CrossEntropyLoss (Classification loss).
        img_loss = criterion(img_outputs, img_pids)  # img_pids(pids): true identity label
        TKP_F_loss = criterion_tkp_f(img_features, frame_features)  # Feature
        TKP_D_loss = criterion_tkp_d(img_features, frame_features)  # Distance
        i2v_loss = criterion_i2v(img_features, vid_features, img_pids, pids)  # image to video loss
        v2i_loss = criterion_i2v(vid_features, img_features, pids, img_pids)  # video to image loss
        i2i_loss = criterion_i2v(img_features, img_features, img_pids, img_pids)  # image to image loss
        v2v_loss = criterion_i2v(vid_features, vid_features, pids, pids)  # video to video loss
        loss = vid_loss + img_loss + i2v_loss + i2i_loss + v2v_loss + v2i_loss + TKP_F_loss + TKP_D_loss

        # backward + optimize
        loss.backward()
        optimizer.step()

        # statistics
        _, vid_preds = torch.max(vid_outputs.data, 1)
        batch_vid_corrects.update(torch.sum(vid_preds == pids.data).float() / b, b)

        _, img_preds = torch.max(img_outputs.data, 1)
        batch_img_corrects.update(torch.sum(img_preds == img_pids.data).float() / (b * t), b * t)

        batch_vid_loss.update(vid_loss.item(), b)
        batch_img_loss.update(img_loss.item(), b * t)
        batch_TKP_F_loss.update(TKP_F_loss.item(), b * t)
        batch_TKP_D_loss.update(TKP_D_loss.item(), b * t)
        batch_i2v_loss.update(i2v_loss.item(), b * t)
        batch_i2i_loss.update(i2i_loss.item(), b * t)
        batch_v2v_loss.update(v2v_loss.item(), b)
        batch_v2i_loss.update(v2i_loss.item(), b)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print('Epoch{0} '
          'Time:{batch_time.sum:.1f}s '
          'Data:{data_time.sum:.1f}s '
          'vXent:{vid_xent.avg:.4f} '
          'iXent:{img_xent.avg:.4f} '
          'TKP_F:{TKP_F.avg:.4f} '
          'TKP_D:{TKP_D.avg:.4f} '
          'i2v:{i2v.avg:.4f} '
          'v2i:{v2i.avg:.4f} '
          'i2i:{i2i.avg:.4f} '
          'v2v:{v2v.avg:.4f} '
          'vAcc:{vid_acc.avg:.2%} '
          'iAcc:{img_acc.avg:.2%} '.format(
        epoch + 1, batch_time=batch_time, data_time=data_time,
        vid_xent=batch_vid_loss,
        img_xent=batch_img_loss,
        TKP_F=batch_TKP_F_loss, TKP_D=batch_TKP_D_loss,
        i2v=batch_i2v_loss, v2i=batch_v2i_loss, i2i=batch_i2i_loss, v2v=batch_v2v_loss,
        vid_acc=batch_vid_corrects, img_acc=batch_img_corrects))


'''
python3 MOT_train_17.py 
小的学习率和小的步长
--train_txt ./MOT16_database/train_path.txt --train_info ./MOT16_database/train_info.npy  --batch_size 32 --test_txt ./MOT_database/test_path.txt  --test_info  ./MOT_database/test_info.npy   --query_info ./MARS_database/query_IDX.npy --n_epochs 20000 --lr 0.00001 --lr_step_size 2 --optimizer adam --ckpt ckpt_NL_0230 --log_path loss.txt --class_per_batch 1 --model_type resnet50_NL --num_workers 8 --track_per_class 4 --S 8 --latent_dim 2048 --temporal Done  --track_id_loss --non_layers  0 2 3 0
大的学习率0.0005
--train_txt ./MOT17_database/train_path.txt --train_info ./MOT17_database/train_info.npy  --batch_size 32 --test_txt ./MOT_database/test_path.txt  --test_info  ./MOT_database/test_info.npy   --query_info ./MARS_database/query_IDX.npy --n_epochs 20000 --lr 0.0005 --lr_step_size 50 --optimizer adam --ckpt ckpt_NL_0230 --log_path loss.txt --class_per_batch 1 --model_type resnet50_NL --num_workers 8 --track_per_class 4 --S 8 --latent_dim 2048 --temporal Done  --track_id_loss --non_layers  0 2 3 0
--train_txt ./MOT_database/train_path.txt --train_info ./MOT_database/train_info.npy  --batch_size 32 --test_txt ./MOT_database/test_path.txt  --test_info  ./MOT_database/test_info.npy   --query_info ./MARS_database/query_IDX.npy --n_epochs 2000 --lr 0.0001 --lr_step_size 50 --optimizer adam --ckpt ckpt_NL_0230 --log_path loss.txt --class_per_batch 1 --model_type resnet50_NL --num_workers 8 --track_per_class 4 --S 8 --latent_dim 2048 --temporal Done  --track_id_loss --non_layers  0 2 3 0


--train_txt ./MOT16_database/train_path.txt --train_info ./MOT16_database/train_info.npy  --batch_size 32 --test_txt ./MOT_database/test_path.txt  --test_info  ./MOT_database/test_info.npy   --query_info ./MARS_database/query_IDX.npy --n_epochs 20000 --lr 0.00001 --lr_step_size 2 --optimizer adam --ckpt ckpt_NL_0230 --log_path loss.txt --class_per_batch 1 --model_type resnet50_NL --num_workers 8 --track_per_class 4 --S 8 --latent_dim 2048 --temporal Done --eval_step 2
--train_txt ./MOT17_database/train_path.txt --train_info ./MOT17_database/train_info.npy  --batch_size 32 --test_txt ./MOT_database/test_path.txt  --test_info  ./MOT_database/test_info.npy   --query_info ./MARS_database/query_IDX.npy --n_epochs 20000 --lr 0.00001 --lr_step_size 2 --optimizer adam --ckpt ckpt_NL_0230 --log_path loss.txt --class_per_batch 1 --model_type resnet50_NL --num_workers 8 --track_per_class 4 --S 8 --latent_dim 2048 --temporal Done --eval_step 1 --debug True
--train_txt ./MOT17_database/train_path.txt --train_info ./MOT17_database/train_info.npy  --batch_size 32 --test_txt ./MOT_database/test_path.txt  --test_info  ./MOT_database/test_info.npy   --query_info ./MARS_database/query_IDX.npy --n_epochs 20000 --lr 0.00001 --lr_step_size 2 --optimizer adam --ckpt ckpt_NL_0230 --log_path loss.txt --class_per_batch 1 --model_type resnet50_NL --num_workers 8 --track_per_class 4 --S 8 --latent_dim 2048 --temporal Done --eval_step 1
'''


# if __name__ == '__main__':
    # Parse args
parser = argparse.ArgumentParser(description='Train img to video model')
# NAAN
parser.add_argument('--train_txt', type=str, default='./MOT17_database/train_path.txt')
parser.add_argument('--train_info', type=str, default='../MOT17_database/train_info.npy')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--test_txt', type=str, default='./MOT_database/test_path.txt')
parser.add_argument('--test_info', type=str, default='./MOT_database/test_info.npy')
parser.add_argument('--query_info', type=str, default='./MARS_database/query_IDX.npy')
parser.add_argument('--n_epochs', type=int, default=2000)
parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--lr_step_size', type=int, default=50)
parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--ckpt', type=str, default='ckpt_NL_0230')
# parser.add_argument('--ckpt', type=str, default='loss.txt')
parser.add_argument('--class_per_batch', type=int, default=1)
parser.add_argument('--model_type', type=str, default='resnet50_NL')
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--track_per_class', type=int, default=4)
parser.add_argument('--S', type=int, default=8)
# parser.add_argument('--S', type=int, default=2048)
parser.add_argument('--temporal', type=str, default='Done')
parser.add_argument('--log_path', type=str, default='loss.txt')
parser.add_argument('--latent_dim', type=int, default=2048)
# parser.add_argument('--track_id_loss', type=str, default='none')
# parser.add_argument('--non_layers', type=str, default='0 2 3 0')

# Datasets
parser.add_argument('--root', type=str, default='/data/datasets/')
parser.add_argument('-d', '--dataset', type=str, default='mars',
                    choices=data_manager.get_names())
parser.add_argument('-j', '--workers', default=4, type=int)
parser.add_argument('--height', type=int, default=256)
parser.add_argument('--width', type=int, default=128)
# Augment
parser.add_argument('--seq_len', type=int, default=4, help="the length of video clips")
parser.add_argument('--sample_stride', type=int, default=8, help="sampling stride of video clips")
# Optimization options
parser.add_argument('--max_epoch', default=150, type=int)
parser.add_argument('--start_epoch', default=0, type=int)
parser.add_argument('--train_batch', default=16, type=int)
parser.add_argument('--test_batch', default=128, type=int)
parser.add_argument('--img_test_batch', default=512, type=int)
# parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float)
parser.add_argument('--stepsize', default=[60, 120], nargs='+', type=int)
parser.add_argument('--gamma', default=0.1, type=float, help="learning rate decay")
parser.add_argument('--weight_decay', default=5e-04, type=float)
parser.add_argument('--num_instances', type=int, default=4, help="number of instances per identity")
# Architecture
parser.add_argument('--vid_arch', type=str, default='vid_nonlocalresnet50')
parser.add_argument('--img_arch', type=str, default='img_resnet50')
# Loss
parser.add_argument('--bp_to_vid', action='store_true', help="weather the TKP loss BP to vid model")
# Miscs
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--resume', type=str, default='', metavar='PATH')
parser.add_argument('--evaluate', action='store_true', help="evaluation only")
parser.add_argument('--eval_step', type=int, default=10)
parser.add_argument('--start_eval', type=int, default=0, help="start to evaluate after specific epoch")
parser.add_argument('--save_dir', type=str, default='log')
parser.add_argument('--gpu_devices', default='0', type=str)
parser.add_argument('--debug', default=False, type=bool, help="debug model include plot graph")
args = parser.parse_args()

if args.debug:
    import visdom

use_gpu = torch.cuda.is_available()

# set transformation (H flip is inside dataset)
# ToTensor: Python Imaging Library -> Tensor
train_transform = Compose(
    [Resize((256, 128)), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)
test_transform = Compose(
    [Resize((256, 128)), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)

print('Start dataloader...')
track_per_class = 4
train_dataloader = utils.Get_MOT16_train_DataLoader(args.train_txt, args.train_info, train_transform, shuffle=True,
                                                    num_workers=args.num_workers, \
                                                    S=args.S, track_per_class=track_per_class,
                                                    class_per_batch=args.class_per_batch)  # args.track_per_class
print('End dataloader...')

# %% Load Model
# network = nn.DataParallel(models.CNN(args.latent_dim,model_type=args.model_type,num_class=num_class,non_layers=args.non_layers,stripes=args.stripes,temporal=args.temporal).cuda())
# modelPath = './ckpt/Similarity_MOT17.pth'
# network = torch.load(modelPath)
vid_model = models.init_model(name=args.vid_arch)
img_model = models.init_model(name=args.img_arch)
# dataset = data_manager.init_dataset(name=args.dataset, root=args.root)
# RuntimeError: CUDA error: device-side assert triggered: label中有些指不在[0, num classes)， 区间左闭右开。比如类别数num_class=3， 你的label出现了-1或者3， 4， 5等！
classifier = models.init_model(name='classifier',
                               num_classes=train_dataloader.dataset.id_count)  # TODO: change num_classes to true person numbers 1450+1

resume = 'TKP/log-mars/best_model.pth.tar'
print("Loading checkpoint from '{}'".format(resume))
checkpoint = torch.load(resume)
vid_model.load_state_dict(checkpoint['vid_model_state_dict'])
img_model.load_state_dict(checkpoint['img_model_state_dict'])
# classifier.load_state_dict(checkpoint['classifier_state_dict'])
if use_gpu:
    vid_model = vid_model.cuda()
    img_model = img_model.cuda()
    classifier = classifier.cuda()

# if args.load_ckpt is not None:
#     state = torch.load(args.load_ckpt)
#     network.load_state_dict(state,strict=False)
# log
os.system('mkdir -p %s' % (args.ckpt))
f = open(os.path.join(args.ckpt, args.log_path), 'a')
f.close()

# %%  Criterion
criterion = nn.CrossEntropyLoss()
criterion = nn.CrossEntropyLoss()
criterion_tkp_f = FeatureBasedTKP(bp_to_vid=args.bp_to_vid)
criterion_tkp_d = SimilarityBasedTKP(distance='euclidean', bp_to_vid=args.bp_to_vid)
criterion_i2v = HeterogeneousTripletLoss(margin=0.3, distance='euclidean')

criterion_triplet = TripletLoss('soft', True)

critetion_id = nn.CrossEntropyLoss().cuda()
# 2. Optimizer
optimizer = torch.optim.Adam([
    {'params': vid_model.parameters(), 'lr': args.lr},
    {'params': img_model.parameters(), 'lr': args.lr},
    {'params': classifier.parameters(), 'lr': args.lr}
], weight_decay=args.weight_decay)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args.stepsize, gamma=args.gamma)
start_epoch = args.start_epoch

id_loss_list = []
trip_loss_list = []
track_id_loss_list = []
best_acc = 0
train_time = 0
start_time = time.time()

# %% Train loop
for epoch in range(start_epoch, args.max_epoch):
    print('epoch', epoch)
    # scheduler.step()
    # validation(vid_model, img_model, classifier, train_dataloader)
    start_train_time = time.time()
    train(epoch, vid_model, img_model, classifier, criterion, criterion_tkp_f, criterion_tkp_d, criterion_i2v,
          optimizer, train_dataloader, use_gpu, train_transform, args)
    torch.cuda.empty_cache()
    # train_time += round(time.time() - start_train_time)

    if (epoch + 1) >= args.start_eval and args.eval_step > 0 and (epoch + 1) % args.eval_step == 0 or (
            epoch + 1) == args.max_epoch:
        print("==> No Test")

        validation(vid_model, img_model, classifier, train_dataloader)

        vid_model_state_dict = vid_model.state_dict()
        img_model_state_dict = img_model.state_dict()
        classifier_state_dict = classifier.state_dict()

        is_best = True
        rank1 = 1.0
        save_checkpoint({
            'vid_model_state_dict': vid_model_state_dict,
            'img_model_state_dict': img_model_state_dict,
            'classifier_state_dict': classifier_state_dict,
            'rank1': rank1,
            'epoch': epoch,
        }, is_best, osp.join(args.save_dir, 'checkpoint_ep' + str(epoch + 1) + '.pth.tar'))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))

