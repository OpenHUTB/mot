import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from config.config import config
######################################################
import math
from models.decode import mot_decode
from layer.losses import FocalLoss
from layer.losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss
from layer.utils import _sigmoid, _tranpose_and_gather_feat
from outils.post_process import ctdet_post_process
from opts import opts

class SSTLoss(nn.Module):
    def __init__(self, opt, use_gpu=config['cuda']):
        super(SSTLoss, self).__init__()
        self.use_gpu = use_gpu
        self.max_object = config['max_object']
####################################################################################
        self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
        self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
            RegLoss() if opt.reg_loss == 'sl1' else None
        self.crit_wh = torch.nn.L1Loss(reduction='sum') if opt.dense_wh else \
            NormRegL1Loss() if opt.norm_wh else \
                RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg
        self.opt = opt
        self.emb_dim = opt.reid_dim
        self.nID = opt.nID
        self.classifier = nn.Linear(self.emb_dim, self.nID)
        self.IDLoss = nn.CrossEntropyLoss(ignore_index=-1)
        # self.TriLoss = TripletLoss()
        self.emb_scale = math.sqrt(2) * math.log(self.nID - 1)
        self.s_det = nn.Parameter(-1.85 * torch.ones(1))
        self.s_id = nn.Parameter(-1.05 * torch.ones(1))

    # merge FairMOT loss and SST loss
    # import numpy as np
    # np.savetxt('results/input.txt', input.detach().cpu().squeeze(), fmt='%.2f', delimiter=' ')
    # np.savetxt('results/target.txt', target.detach().cpu().squeeze(), fmt='%.2f', delimiter=' ')
    # np.savetxt('results/mask0.txt', mask0.detach().cpu().squeeze(), fmt='%.2f', delimiter=' ')
    # np.savetxt('results/mask1.txt', mask1.detach().cpu().squeeze(), fmt='%.2f', delimiter=' ')
    def forward(self, input, target, mask0, mask1, outputs, current_fair_labels, next_fair_labels):  # 4*1*81*81, 4*1*81*81, 4*1*81, 4*1*81
        # current_fair_loss = self.lossf(outputs["out_pre"], current_fair_labels)
        # next_fair_loss = self.lossf(outputs["out_next"], next_fair_labels)

        mask_pre = mask0[:, :, :]
        mask_next = mask1[:, :, :]
        mask0 = mask0.unsqueeze(3).repeat(1, 1, 1, self.max_object+1)  # 4*1*81 -> 4*1*81*1 -> 4*1*81*81
        mask1 = mask1.unsqueeze(2).repeat(1, 1, self.max_object+1, 1)  # 4*1*81 -> 4*1*1*81 -> 4*1*81*81
        mask0 = Variable(mask0.data)
        mask1 = Variable(mask1.data)
        target = Variable(target.byte().data)

        if self.use_gpu:
            mask0 = mask0.cuda()
            mask1 = mask1.cuda()

        mask_region = (mask0 * mask1).float() # the valid position mask 4*1*81*81
        mask_region_pre = mask_region.clone() #note: should use clone (fix this bug)
        mask_region_pre[:, :, self.max_object, :] = 0  # set last row to 0
        mask_region_next = mask_region.clone() #note: should use clone (fix this bug)
        mask_region_next[:, :, :, self.max_object] = 0  # set last column to 0
        mask_region_union = mask_region_pre*mask_region_next  # remove last row and column

        input_pre = nn.Softmax(dim=3)(mask_region_pre*input)  # forward A1^: row-wise softmax -> column trimming
        input_next = nn.Softmax(dim=2)(mask_region_next*input)  # backward A2^: column-wise softmax -> row trimming
        input_all = input_pre.clone()
        input_all[:, :, :self.max_object, :self.max_object] = torch.max(input_pre, input_next)[:, :, :self.max_object, :self.max_object]  # max(A1^, A2^)
        # input_all[:, :, :self.max_object, :self.max_object] = ((input_pre + input_next)/2.0)[:, :, :self.max_object, :self.max_object]
        target = target.float()
        target_pre = mask_region_pre * target
        target_next = mask_region_next * target
        target_union = mask_region_union * target  # 4*1*81*81
        target_num = target.sum()  # target number (69)
        target_num_pre = target_pre.sum()  # 67
        target_num_next = target_next.sum()  # 67
        target_num_union = target_union.sum()  # 65
        #todo: remove the last row negative effect
        if int(target_num_pre.item()):  # Forward-direction loss (forward association loss) Lf.   pytorch version problem:  .data[0] --> .item()
            loss_pre = - (target_pre * torch.log(input_pre)).sum() / target_num_pre
        else:
            loss_pre = - (target_pre * torch.log(input_pre)).sum()
        if int(target_num_next.item()):  # Backward-direction (backward association loss) Lb
            loss_next = - (target_next * torch.log(input_next)).sum() / target_num_next
        else:
            loss_next = - (target_next * torch.log(input_next)).sum()
        if int(target_num_pre.item()) and int(target_num_next.item()):  # Consistency loss Lc: to rebuff any inconsistency between Lf and Lb
            loss = -(target_pre * torch.log(input_all)).sum() / target_num_pre
        else:
            loss = -(target_pre * torch.log(input_all)).sum()

        if int(target_num_union.item()):  # Assemble Loss (overlap loss) La: suppresse non-maximum forwad/backward association for affinity predictions
            loss_similarity = (target_union * (torch.abs((1-input_pre) - (1-input_next)))).sum() / target_num
        else:
            loss_similarity = (target_union * (torch.abs((1-input_pre) - (1-input_next)))).sum()

        _, indexes_ = target_pre.max(3)
        indexes_ = indexes_[:, :, :-1]
        _, indexes_pre = input_all.max(3)
        indexes_pre = indexes_pre[:, :, :-1]
        mask_pre_num = mask_pre[:, :, :-1].sum().item()
        if mask_pre_num:  # forward matching accuracy
            accuracy_pre = (indexes_pre[mask_pre[:, :, :-1].bool()] == indexes_[mask_pre[:,:, :-1].bool()]).float().sum() / mask_pre_num
        else:
            accuracy_pre = (indexes_pre[mask_pre[:, :, :-1].bool()] == indexes_[mask_pre[:, :, :-1].bool()]).float().sum() + 1

        _, indexes_ = target_next.max(2)
        indexes_ = indexes_[:, :, :-1]
        _, indexes_next = input_next.max(2)
        indexes_next = indexes_next[:, :, :-1]
        mask_next_num = mask_next[:, :, :-1].sum().item()
        if mask_next_num:  # backward matching accuracy
            accuracy_next = (indexes_next[mask_next[:, :, :-1].bool()] == indexes_[mask_next[:, :, :-1].bool()]).float().sum() / mask_next_num
        else:
            accuracy_next = (indexes_next[mask_next[:, :, :-1].bool()] == indexes_[mask_next[:, :, :-1].bool()]).float().sum() + 1
######################################################################################################################################
        # + current_fair_loss + next_fair_loss
        return loss_pre, loss_next, loss_similarity, \
               (loss_pre + loss_next + loss + loss_similarity )/4.0 , accuracy_pre, accuracy_next, (accuracy_pre + accuracy_next)/2.0, indexes_pre

    def lossf(self,outputs,batch):
        opt = self.opt
        hm_loss, wh_loss, off_loss, id_loss = 0, 0, 0, 0
        for s in range(opt.num_stacks): # =1
            # output = outputs[s]
            output = outputs
            if not opt.mse_loss:
                output['hm'] = _sigmoid(output['hm'])

            hm_loss += self.crit(output['hm'], torch.from_numpy(batch['hm']).cuda()) / opt.num_stacks  # FocalLoss
            if opt.wh_weight > 0:
                if opt.dense_wh:
                    mask_weight = batch['dense_wh_mask'].sum() + 1e-4
                    wh_loss += (
                                       self.crit_wh(output['wh'] * batch['dense_wh_mask'],
                                                    batch['dense_wh'] * batch['dense_wh_mask']) /
                                       mask_weight) / opt.num_stacks
                else:
                    # RegL1Loss(1*2*152*272, 128, 128, 128*2), 6*2*152*272, 6*128, 6*128, 6*128*2
                    wh_loss += self.crit_reg(
                        output['wh'], torch.from_numpy(batch['reg_mask']).cuda().unsqueeze(0),
                        torch.from_numpy(batch['ind']).cuda().unsqueeze(0), torch.from_numpy(batch['wh']).cuda().unsqueeze(0)) / opt.num_stacks
                    # print(output['wh'].shape, batch['reg_mask'].shape, batch['ind'].shape, batch['wh'].shape)

            if opt.reg_offset and opt.off_weight > 0:
                off_loss += self.crit_reg(output['reg'], torch.from_numpy(batch['reg_mask']).cuda().unsqueeze(0),
                                          torch.from_numpy(batch['ind']).cuda().unsqueeze(0), torch.from_numpy(batch['reg']).cuda().unsqueeze(0)) / opt.num_stacks

            if opt.id_weight > 0:
                id_head = _tranpose_and_gather_feat(output['id'], torch.from_numpy(batch['ind']).cuda().unsqueeze(0))  # 1*128*512
                id_head = id_head[torch.from_numpy(batch['reg_mask']).cuda().unsqueeze(0) > 0].contiguous()  # 128
                id_head = self.emb_scale * F.normalize(id_head)
                id_target = batch['ids'][batch['reg_mask'] > 0]
                id_output = self.classifier(id_head.cpu()).contiguous()
                id_loss += self.IDLoss(id_output, torch.from_numpy(id_target).cpu())  # TypeError: 'int' object is not callable
                # id_loss += self.IDLoss(id_output, id_target) + self.TriLoss(id_head, id_target)

        # loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + opt.off_weight * off_loss + opt.id_weight * id_loss

        det_loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + opt.off_weight * off_loss

        loss = torch.exp(-self.s_det) * det_loss + torch.exp(-self.s_id) * id_loss + (self.s_det + self.s_id)
        loss *= 0.5
        return loss

    def getProperty(self, input, target, mask0, mask1):
        return self.forward(input, target, mask0, mask1)
