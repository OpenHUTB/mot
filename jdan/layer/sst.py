"""
SST tracker net

Thanks to ssd pytorch implementation (see https://github.com/amdegroot/ssd.pytorch)
copyright: shijie Sun (shijieSun@chd.edu.cn)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from config.config import config
import numpy as np
import os

# Detection & Re-ID Network
# from models.model import create_model, load_model
from layer.models.model import create_model, load_model
from layer.models.decode import mot_decode
from layer.models.utils import _tranpose_and_gather_feat
from outils.post_process import ctdet_post_process

#todo: add more extra columns to use the mogranic method
#todo: label the mot17 data and train the detector.
#todo: Does the inherient really work
#todo: add achors to extract features
#todo: think about how to represent the motion model
#todo: debug every feature step and see the feature change of each objects [do]
#todo: change the output of the SST.
#todo: add relu to extra net


class SST(nn.Module):
    #new: combine two vgg_net
    def __init__(self, phase, use_gpu=config['cuda']):
        super(SST, self).__init__()
        self.phase = phase  # train,

        # Detection & Re-ID network
        print('Creating Detection & Re-ID network model...')
        heads = {'hm': 1,
                 'wh': 2,
                 'id': 512,
                 'reg': 2}
        self.model = create_model('dla_34', heads, 256)
        self.model = load_model(self.model, '/data2/whd/workspace/MOT/FairMOT/SST/weights/all_dla34.pth')
        self.model = self.model.to(torch.device('cuda'))
        self.model.eval()

        self.stacker2_bn = nn.BatchNorm2d(512)
        self.final_dp = nn.Dropout(0.5)
        self.final_net = nn.ModuleList(add_final([1024, 512, 256, 128, 64, 1]))  # the compression network in paper -------*big change!*

        self.image_size = config['sst_dim']
        self.max_object = config['max_object']  # the number of max tracking object (Nm: 80)
        self.selector_channel = config['selector_channel']

        self.false_objects_column = None
        self.false_objects_row = None
        self.false_constant = config['false_constant']
        self.use_gpu = use_gpu

    def forward(self, x_pre, x_next, index_pre, index_next, current_valid_boxes, next_valid_boxes, extra=None):
        '''
        the sst net forward stream
        :param x_pre:  the previous image, (1, 3, 900, 900) FT
        :param x_next: the next image,  (1, 3, 900, 900) FT
        :param l_pre: the previous box center, (1, 60, 1, 1, 2) FT
        :param l_next: the next box center, (1, 60, 1, 1, 2) FT
        :param valid_pre: the previous box mask, (1, 1, 61) BT
        :param valid_next: the next box mask, (1, 1, 61) BT
        :return: the similarity matrix
        '''
        # pedestrian features, bounding boxes, {heatmap, width/height(size:1*2*W*H), identity embedding(512), regression(offset:1*2*W*H) }
        x_pre, dets_pre, out_pre = self.forward_feature_extracter(x_pre, current_valid_boxes)  # 1*n*512, n*5, {hm:1*1*152*272, wh:1*2*152*272, id:1*512*152*272, reg:1*2*152*272} = 1*3*608*1088
        x_next, dets_next, out_next = self.forward_feature_extracter(x_next, next_valid_boxes)
        # folder == '/data2/whd/workspace/MOT/SST/results/MOT17/train/MOT17-10-SDP' and key == 45

        # check data
        if extra is not None:
            import cv2
            current_img_path = extra[0]
            next_img_path = extra[1]
            current_img = cv2.imread(current_img_path)
            next_img = cv2.imread(next_img_path)
            for i in range(19):
                x1 = int(dets_pre[i][0])  # * width
                y1 = int(dets_pre[i][1])  # * height
                x2 = int(dets_pre[i][2])
                y2 = int(dets_pre[i][3])
                if x1 < 0: x1 = 0
                if y1 < 0 : y1 = 0
                if x2 < 0 : y2 = 0
                if y2 < 0 : y2 = 0
                point_color = (0, 0, 255)  # BGR
                cv2.rectangle(current_img, (x1, y1), (x2, y2), point_color, 2)
                cv2.putText(current_img, str(i), (x1, y1), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
            cv2.imshow('current image(network output)', current_img)
            cv2.waitKey(0)

        x = self.forward_stacker_features(x_pre, x_next, index_pre, index_next, False)  # process two frame features
        # x = torch.from_numpy(x)
        net={}
        net['sim']=x
        net['out_pre']=out_pre
        net['out_next'] = out_next
        return net

    def forward_feature_extracter(self, x, valid_boxes=None):
        '''
        extract features from the DLA net
        :param x: input image
        :return: the features
        '''

        # Detection & Re-ID network (replace with vgg and extras in SST)
        with torch.no_grad():
            output = self.model(x)[-1]
            # print("output:",output)
            hm = output['hm'].sigmoid_()
            wh = output['wh']
            id_feature = output['id']  # 1*512*152*272
            reg = output['reg']
            dets, inds = mot_decode(hm, wh, reg=reg, cat_spec_wh=False, K=128)
            id_feature = _tranpose_and_gather_feat(id_feature, inds)  # 1*512*152*272 -- > 1*128*512

        # check model parameter is changed or not
        '''
        torch.save(self.model.state_dict(), 'results/model_sst.pth')  # cmp -l model_fairmot.pth model_sst.pth
        np.savetxt('results/x_sst.txt', x.detach().cpu()[0, 1, :, :],
                           fmt='%.10f', delimiter=' ')
        for name, param in self.model.named_parameters():
            if name == 'base.level3.tree1.tree1.conv1.weight':
                np.savetxt('results/' + name + '.txt', param.detach().cpu()[0, 0, :, :],
                           fmt='%.10f', delimiter=' ')
                print(param.shape)  # [27,64,3,3]
        '''

            # id_feature = id_feature.squeeze(0)  # --> 128*512
            # id_feature = id_feature.cpu().numpy()
        width = 1920
        height = 1080
        inp_height = 608
        inp_width = 1088
        down_ratio = 4
        c = np.array([width / 2., height / 2.], dtype=np.float32)  # [960, 540]
        s = max(float(inp_width) / float(inp_height) * height, width) * 1.0  # 1932.63
        meta = {'c': c, 's': s,
                'out_height': inp_height // down_ratio,  # 152
                'out_width': inp_width // down_ratio}    # 272
        dets = self.post_process(dets, meta)  # 1*128*6  -> dict: 128*5
        dets = self.merge_outputs([dets])[1]  # --> 128*5
        remain_inds = dets[:, 4] > 0.4
        # remain_inds = dets[:, :, 4] > 0.2
        dets = dets[remain_inds]  # 12*5
        # id_feature = id_feature[remain_inds]
        id_feature = id_feature[:, remain_inds, :]  # id_feature[remain_inds] -> 12*512;  IndexError: The shape of the mask [128] at index 0does not match the shape of the indexed tensor [4, 128, 512] at index 0

        # filter with current_valid_boxes and next_valid_boxes to solve detection number is greater than tracks number
        if valid_boxes is not None:
            remain_dets = np.zeros(dets.shape[0], dtype=bool)
            for i in range(dets.shape[0]):
                if i in valid_boxes:
                    remain_dets[i] = True
            dets = dets[remain_dets]
            id_feature = id_feature[:, remain_dets, :]

        return id_feature, dets, output

    def merge_outputs(self, detections):
        results = {}
        for j in range(1, 1 + 1):
            results[j] = np.concatenate(
                [detection[j] for detection in detections], axis=0).astype(np.float32)

        scores = np.hstack(
            [results[j][:, 4] for j in range(1, 1 + 1)])
        if len(scores) > 128:
            kth = len(scores) - 128
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, 1 + 1):
                keep_inds = (results[j][:, 4] >= thresh)
                results[j] = results[j][keep_inds]
        return results

    def post_process(self, dets, meta):
        dets = dets.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])  # 1*128*6
        dets = ctdet_post_process(
            dets.copy(), [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'], 1)  # 1*128*6 -> list: 128*5
        for j in range(1, 1 + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
        return dets[0]

    def get_similarity(self, image1, detection1, image2, detection2):
        feature1 = self.forward_feature_extracter(image1, detection1)
        feature2 = self.forward_feature_extracter(image2, detection2)
        return self.forward_stacker_features(feature1, feature2, False)

    # resize the detection feature (1*14*520) into fix size feature matrix (1*80*520)
    def resize_index_dim(self, x, index, added_size, dim=1, constant=0):
        if added_size <= 0:
            return x
        shape = list(x.shape)
        shape[dim] = added_size  # 1*14*520 -> 1*(80-14)*520
        if self.use_gpu:
            new_data = Variable(torch.ones(shape)*constant).cuda()
        else:
            new_data = Variable(torch.ones(shape) * constant)
        return torch.cat([x, new_data], dim=dim)[:, index, :].squeeze(0)  # append other position of detection feature with full zero

    def resize_dim(self, x, added_size, dim=1, constant=0):
        if added_size <= 0:
            return x
        shape = list(x.shape)
        shape[dim] = added_size  # 1*14*520 -> 1*(80-14)*520
        if self.use_gpu:
            new_data = Variable(torch.ones(shape)*constant).cuda()
        else:
            new_data = Variable(torch.ones(shape) * constant)
        return torch.cat([x, new_data], dim=dim)

    def forward_stacker_features(self, xp, xn, index_pre, index_next, fill_up_column=True):  # get affinity matrix between previous frame (xp:1*14*520) and current frame (xn:1*15*520)
        pre_rest_num = self.max_object - xp.shape[1]  # 66 = 80-14
        next_rest_num = self.max_object - xn.shape[1]  # 65 = 80-15
        pre_num = xp.shape[1]    # 14
        next_num = xn.shape[1]   # 15
        x = self.forward_stacker2(
            self.resize_index_dim(xp, index_pre, pre_rest_num, dim=1),  # 1*17*512 --pad with 0> 1*80*512 --forward_stacker2():repeat row> 1*80*80*512
            self.resize_index_dim(xn, index_next, next_rest_num, dim=1)
        )  # stack two frame feature into feature matrix (1*1024*80*80)

        x = self.final_dp(x)  # drop out
        # [B, N, N, 1]
        x = self.forward_final(x, self.final_net)  # compression network 1*1024*80*80 --> 1*1*80*80
        x = self.add_unmatched_dim(x)  # 1*1*80*80 --> 1*1*81*81


        # x = x.contiguous()
        # # add zero
        # if next_num < self.max_object:
        #     x[0, 0, :, next_num:] = 0  # set the other detection position column in matrix M to 0
        # if pre_num < self.max_object:
        #     x[0, 0, pre_num:, :] = 0
        # x = x[0, 0, :]  # --> 80*80
        # # add false unmatched row and column
        # x = self.resize_dim(x, 1, dim=0, constant=self.false_constant)  # add one row (full 10): 80*80 -> 81*80
        # x = self.resize_dim(x, 1, dim=1, constant=self.false_constant)  # add one column: 81*80 --> 81*81
        #
        # x_f = F.softmax(x, dim=1)  # disappear: column-wise softmax: --> 81*81
        # x_t = F.softmax(x, dim=0)  # occur: row-wise softmax
        # # slice
        # last_row, last_col = x_f.shape
        # row_slice = list(range(pre_num)) + [last_row-1]
        # col_slice = list(range(next_num)) + [last_col-1]
        # x_f = x_f[row_slice, :]  # 81*81 --> 17*81
        # x_f = x_f[:, col_slice]  # 17*81 --> 17*16
        # x_t = x_t[row_slice, :]
        # x_t = x_t[:, col_slice]  # --> 17*16
        #
        # x = Variable(torch.zeros(pre_num, next_num + 1))
        # # x[0:pre_num, 0:next_num] = torch.max(x_f[0:pre_num, 0:next_num], x_t[0:pre_num, 0:next_num])
        # x[0:pre_num, 0:next_num] = (x_f[0:pre_num, 0:next_num] + x_t[0:pre_num, 0:next_num]) / 2.0
        # x[:, next_num:next_num + 1] = x_f[:pre_num, next_num:next_num + 1]
        # if fill_up_column and pre_num > 1:
        #     x = torch.cat([x, x[:, next_num:next_num + 1].repeat(1, pre_num - 1)], dim=1)


        # if self.use_gpu:
        #     y = x.data.cpu().numpy()
        #     # del x, x_f, x_t
        #     # torch.cuda.empty_cache()
        # else:
        #     y = x.data.numpy()

        return x  # 16*16, similarity matrix

    # for tracking
    def forward_stacker_final(self, xp, xn, fill_up_column=True):
        pre_rest_num = self.max_object - xp.shape[1]
        next_rest_num = self.max_object - xn.shape[1]
        pre_num = xp.shape[1]
        next_num = xn.shape[1]
        x = self.forward_stacker2(
            self.resize_dim(xp, pre_rest_num, dim=1),
            self.resize_dim(xn, next_rest_num, dim=1)
        )

        x = self.final_dp(x)
        # [B, N, N, 1]
        x = self.forward_final(x, self.final_net)
        x = x.contiguous()
        # add zero
        if next_num < self.max_object:
            x[0, 0, :, next_num:] = 0
        if pre_num < self.max_object:
            x[0, 0, pre_num:, :] = 0
        x = x[0, 0, :]
        # add false unmatched row and column
        x = self.resize_dim(x, 1, dim=0, constant=self.false_constant)
        x = self.resize_dim(x, 1, dim=1, constant=self.false_constant)

        x_f = F.softmax(x, dim=1)
        x_t = F.softmax(x, dim=0)
        # slice
        last_row, last_col = x_f.shape
        row_slice = list(range(pre_num)) + [last_row-1]
        col_slice = list(range(next_num)) + [last_col-1]
        x_f = x_f[row_slice, :]
        x_f = x_f[:, col_slice]
        x_t = x_t[row_slice, :]
        x_t = x_t[:, col_slice]

        x = Variable(torch.zeros(pre_num, next_num+1))
        # x[0:pre_num, 0:next_num] = torch.max(x_f[0:pre_num, 0:next_num], x_t[0:pre_num, 0:next_num])
        x[0:pre_num, 0:next_num] = (x_f[0:pre_num, 0:next_num] + x_t[0:pre_num, 0:next_num]) / 2.0
        x[:, next_num:next_num+1] = x_f[:pre_num, next_num:next_num+1]
        if fill_up_column and pre_num > 1:
            x = torch.cat([x, x[:, next_num:next_num+1].repeat(1, pre_num-1)], dim=1)

        if self.use_gpu:
            y = x.data.cpu().numpy()
            # del x, x_f, x_t
            # torch.cuda.empty_cache()
        else:
            y = x.data.numpy()

        return y

    def forward_vgg(self, x, vgg, sources):
        for k in range(16):
            x = vgg[k](x)
        sources.append(x)

        for k in range(16, 23):
            x = vgg[k](x)
        sources.append(x)

        for k in range(23, 35):
            x = vgg[k](x)
        sources.append(x)
        return x

    def forward_extras(self, x, extras, sources):
        for k, v in enumerate(extras):
            x = v(x) #x = F.relu(v(x), inplace=True)        #done: relu is unnecessary.
            if k % 6 == 3:                  #done: should select the output of BatchNormalization (-> k%6==2)
                sources.append(x)
        return x

    def forward_selector_stacker1(self, sources, labels, selector):
        '''
        :param sources: [B, C, H, W]
        :param labels: [B, N, 1, 1, 2]
        :return: the connected feature
        '''
        sources = [
            F.relu(net(x), inplace=True) for net, x in zip(selector, sources)
        ]

        res = list()
        for label_index in range(labels.size(1)):  # 80
            label_res = list()
            for source_index in range(len(sources)):  # 9
                # [N, B, C, 1, 1]
                label_res.append(
                    # [B, C, 1, 1]
                    F.grid_sample(sources[source_index],  # [B, C, H, W]
                                  labels[:, label_index, :]  # [B, 1, 1, 2
                                  ).squeeze(2).squeeze(2) # 2*60 * 225*225, 2* 1*1 *2 -> 2*60*1*1
                )
            res.append(torch.cat(label_res, 1))  # 9(2*60, 2*80, ..., 2*20) -> 2*520

        return torch.stack(res, 1)  # 2*80*520

    def forward_stacker2(self, stacker1_pre_output, stacker1_next_output):
        stacker1_pre_output = stacker1_pre_output.unsqueeze(2).repeat(1, 1, self.max_object, 1).permute(0, 3, 1, 2)
        stacker1_next_output = stacker1_next_output.unsqueeze(1).repeat(1, self.max_object, 1, 1).permute(0, 3, 1, 2)

        stacker1_pre_output = self.stacker2_bn(stacker1_pre_output.contiguous())
        stacker1_next_output = self.stacker2_bn(stacker1_next_output.contiguous())

        # stacker1_pre_output = stacker1_pre_output.squeeze()   # 1*512*96*80; solve: RuntimeError: invalid argument 0: Sizes of tensors must match except in dimension 1. Got 80 and 96 in dimension 2 at
        # stacker1_next_output = stacker1_next_output.squeeze() # 1*512*80*87

        output = torch.cat(
            [stacker1_pre_output, stacker1_next_output],
            1
        )  # max_object is set 150(80).  MOT17-03-FRCNN.txt:700, 46, 0_0_4_0_3_3:   RuntimeError: invalid argument 0: Sizes of tensors must match except in dimension 1. Got 80 and 95 in dimension 2 at

        return output

    def forward_final(self, x, final_net):
        x_c = x.contiguous()
        input = x_c
        for f in final_net:
            output = f(input).clone()  # RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation: [torch.cuda.FloatTensor [1, 1, 80, 80]], which is output 0 of ReluBackward1, is at version 3; expected version 1 instead. Hint: the backtrace further above shows the operation that failed to compute its gradient. The variable in question was changed in there or anywhere later. Good luck!
            input = output
        return output

    def add_unmatched_dim(self, x):
        if self.false_objects_column is None:
            self.false_objects_column = Variable(torch.ones(x.shape[0], x.shape[1], x.shape[2], 1)) * self.false_constant
            if self.use_gpu:
                self.false_objects_column = self.false_objects_column.cuda()
        x = torch.cat([x, self.false_objects_column], 3)  # 1*1*80*1

        if self.false_objects_row is None:
            self.false_objects_row = Variable(torch.ones(x.shape[0], x.shape[1], 1, x.shape[3])) * self.false_constant
            if self.use_gpu:
                self.false_objects_row = self.false_objects_row.cuda()
        x = torch.cat([x, self.false_objects_row], 2)  # 1*1 * 1*81
        return x  # 1*1*81*81

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(
                torch.load(base_file,
                           map_location=lambda storage, loc: storage)
            )
            print('Finished')
        else:
            print('Sorry only .pth and .pkl files supported.')

def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers

def add_extras(cfg, i, batch_norm=True):
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                conv2d = nn.Conv2d(in_channels, cfg[k+1],
                                     kernel_size=(1, 3)[flag],
                                     stride=2,
                                     padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(cfg[k+1]), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
            else:
                conv2d = nn.Conv2d(in_channels, v,
                                     kernel_size=(1, 3)[flag])
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
            flag = not flag
        in_channels = v
    return layers


# construct every layer of compression network
# input: [1024, 512, 256, 128, 64, 1]
def add_final(cfg, batch_normal=True):
    layers = []
    in_channels = int(cfg[0])
    layers += []
    # 1. add the 1:-2 layer with BatchNorm
    for v in cfg[1:-2]:
        conv2d = nn.Conv2d(in_channels, v, kernel_size=1, stride=1)  # dimension reduction (kernel size = 1)
        if batch_normal:
            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
        else:
            layers += [conv2d, nn.ReLU(inplace=True)]
        in_channels = v
    # 2. add the -2: layer without BatchNorm for BatchNorm would make the output value normal distribution.
    for v in cfg[-2:]:
        conv2d = nn.Conv2d(in_channels, v, kernel_size=1, stride=1)
        layers += [conv2d, nn.ReLU(inplace=True)]
        in_channels = v

    return layers

def selector(vgg, extra_layers, batch_normal=True):
    '''
    batch_normal must be same to add_extras batch_normal
    '''
    selector_layers = []
    vgg_source = config['vgg_source']

    for k, v in enumerate(vgg_source):
         selector_layers += [nn.Conv2d(vgg[v-1].out_channels,
                              config['selector_channel'][k],
                              kernel_size=3,
                              padding=1)]
    if batch_normal:
        for k, v in enumerate(extra_layers[3::6], 3):
            selector_layers += [nn.Conv2d(v.out_channels,
                                 config['selector_channel'][k],
                                 kernel_size=3,
                                 padding=1)]
    else:
        for k, v in enumerate(extra_layers[3::4], 3):
            selector_layers += [nn.Conv2d(v.out_channels,
                                 config['selector_channel'][k],
                                 kernel_size=3,
                                 padding=1)]

    return vgg, extra_layers, selector_layers

def build_sst(phase, size=900, use_gpu=config['cuda']):
    '''
    create the SSJ Tracker Object
    :return: ssj tracker object
    '''
    if phase != 'test' and phase != 'train':
        print('Error: Phase not recognized')
        return

    # if size != 900:
    #     print('Error: Sorry only SST{} is supported currently!'.format(size))
    #     return
    #
    # base = config['base_net']
    # extras = config['extra_net']
    # final = config['final_net']  # '900': [1040, 512, 256, 128, 64, 1],
    # final[str(size)][0] = 1024  # change 1040 to 1024

    return SST(phase,use_gpu)
