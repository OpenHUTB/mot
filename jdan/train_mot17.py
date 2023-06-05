import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import argparse
from torch.autograd import Variable
import torch.utils.data as data
from torch.optim import lr_scheduler

import numpy as np
import cv2
from data.mot import MOTTrainDataset
from config.config import config
from layer.sst import build_sst
from outils.augmentations import SSJAugmentation, collate_fn
from layer.sst_loss import SSTLoss
import time
import torchvision.utils as vutils
from outils.operation import show_circle, show_batch_circle_image
#################################################################
from torchvision.transforms import transforms as T
from opts import opts
from models.model import create_model, load_model, save_model


from tracker import SSTTracker, TrackerConfig, Track
from outils.timer import Timer
from data.mot_data_reader import MOTDataReader

from tracking_utils.evaluation import Evaluator
import motmetrics as mm

from trains.train_factory import train_factory

# is_debug = True
is_debug = False

np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)  # find "inplace operation" error in model

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)  # solve: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tupl

# close UserWarning: /opt/conda/conda-bld/pytorch_1565272279342/work/aten/src/ATen/native/IndexingUtils.h:20: UserWarning: indexing with dtype torch.uint8 is now deprecated, please use a dtype torch.bool instead.

str2bool = lambda v: v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description='Single Shot Joint Tracker Train')
parser.add_argument('--version', default='v1', help='current version')
parser.add_argument('--basenet', default=config['base_net_folder'], help='pretrained base model')
parser.add_argument('--matching_threshold', default=0.5, type=float, help='Min Jaccard index for matching')
parser.add_argument('--batch_size', default=config['batch_size'], type=int, help='Batch size for training')
parser.add_argument('--resume', default=config['resume'], type=str, help='Resume from checkpoint')
parser.add_argument('--num_workers', default=config['num_workers'], type=int, help='Number of workers used in dataloading')
parser.add_argument('--iterations', default=config['iterations'], type=int, help='Number of training iterations')
parser.add_argument('--start_iter', default=config['start_iter'], type=int, help='Begin counting iterations starting from this value (used with resume)')
parser.add_argument('--cuda', default=config['cuda'], type=str2bool, help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate', default=config['learning_rate'], type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--log_iters', default=True, type=bool, help='Print the loss at each iteration')
parser.add_argument('--tensorboard',default=True, type=str2bool, help='Use tensor board x for loss visualization')
parser.add_argument('--port', default=6006, type=int, help='set vidom port')
parser.add_argument('--send_images', type=str2bool, default=True, help='Sample a random image from each 10th batch, send it to visdom after augmentations step')
parser.add_argument('--save_folder', default=config['save_folder'], help='Location to save checkpoint models')
parser.add_argument('--mot_root', default=config['mot_root'], help='Location of VOC root directory')
###########################################################################################################################################
parser.add_argument('task', default='mot', help='mot')
parser.add_argument('--dataset', default='jde', help='jde')
parser.add_argument('--exp_id', default='default')
parser.add_argument('--test', action='store_true')
parser.add_argument('--load_model', default='',help='path to pretrained model')
parser.add_argument('--gpus', default='0, 1',help='-1 for CPU, use comma for multiple gpus')
args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

if 'save_images_folder' in config and not os.path.exists(config['save_images_folder']):
    os.mkdir(config['save_images_folder'])

sst_dim = config['sst_dim']
means = config['mean_pixel']
batch_size = args.batch_size
max_iter = args.iterations
weight_decay = args.weight_decay

if 'learning_rate_decay_by_epoch' in config:
    stepvalues = list((config['epoch_size'] * i for i in config['learning_rate_decay_by_epoch']))
    save_weights_iteration = config['save_weight_every_epoch_num'] * config['epoch_size']
else:
    stepvalues = (90000, 95000)
    save_weights_iteration = 5000

gamma = args.gamma
momentum = args.momentum

from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

# data prefetch
class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


if args.tensorboard:
    from tensorboardX import SummaryWriter
    if not os.path.exists(config['log_folder']):
        os.mkdir(config['log_folder'])
    writer = SummaryWriter(log_dir=config['log_folder'])

sst_net = build_sst('train')

if args.cuda:
    # net = torch.nn.DataParallel(sst_net)
    cudnn.benchmark = True

if args.resume:
    print('Resuming training, loading {}...'.format(args.resume))
    # sst_net.load_weights('./weights/ddt_0.pth')
    # sst_net.load_weights(args.resume)
    # sst_net.model.load_state_dict()
else:
    pass
    # vgg_weights = torch.load(args.basenet)

    # print('Loading the base network...')
    # sst_net.vgg.load_state_dict(vgg_weights)

if args.cuda:
    sst_net = sst_net.cuda()

def xavier(param):
    init.xavier_uniform_(param)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()

# resume parameter from SST parameter
resume_model = './weights/ddt_sst.pth'
# sst_net.load_state_dict(torch.load(resume_model))
if not args.resume:
    print('Initializing weights...')
    # sst_net.extras.apply(weights_init)
    # sst_net.selector.apply(weights_init)
sst_net.final_net.apply(weights_init)

net = sst_net

# optimizer = optim.SGD(net.parameters(), lr=args.lr,momentum=args.momentum,weight_decay=args.weight_decay)
optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr,momentum=args.momentum,weight_decay=args.weight_decay)
# scheduler = lr_scheduler.StepLR(optimizer, 100, 0.1)

# add FairMOT loss
opt = opts().parse()
transforms = T.Compose([T.ToTensor()])
dataset = MOTTrainDataset(mot_root=args.mot_root, transform=SSJAugmentation(config['sst_dim_width'], config['sst_dim_height'], means),transforms=transforms,augment=True )
opt = opts().update_dataset_info_and_set_heads(opt, dataset)
model = create_model(opt.arch, opt.heads, opt.head_conv)
criterion = SSTLoss(opt,use_gpu=args.cuda)

def train():
    net.train()
    net.model.eval()
    current_lr = config['learning_rate']
    print('Loading Dataset...')
###################################  dataset  #############################################
    #dataset = Dataset(opt, dataset_root, trainset_paths, (1088, 608), augment=True, transforms=transforms)
    #opt = opts().parse()
    #dataset = MOTTrainDataset(opt,args.mot_root,SSJAugmentation(config['sst_dim_width'], config['sst_dim_height'], means) )

    epoch_size = len(dataset) // args.batch_size
    print('Training SSJ on', dataset.dataset_name)
    step_index = 0

    batch_iterator = None

    batch_size = 1  # sst.py -> forward_feature_extracter() -> post_process()
#################################  data_loader  ##############################################data.DataLoader
    data_loader = DataLoaderX(dataset, batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True,
                                  collate_fn=collate_fn,
                                  pin_memory=True)

    # Trainer = train_factory['mot']
    # trainer = Trainer(opt, net, optimizer)

    for iteration in range(args.start_iter, max_iter):
        if (not batch_iterator) or (iteration % epoch_size == 0):
            # create batch iterator
            batch_iterator = iter(data_loader)
            all_epoch_loss = []

        if iteration in stepvalues:
            step_index += 1
            current_lr = adjust_learning_rate(optimizer, args.gamma, step_index)

        # load train data (1*3*608*1088, 16*4, 1*3*608*1088, 14*4, 16*14)
        img_pre, img_next, boxes_pre, boxes_next, labels, valid_pre, index_pre, valid_next, index_next, \
            current_fair_labels, next_fair_labels, cur_dla_input, next_dla_input, \
            current_img_path, next_img_path, \
            current_valid_boxes, next_valid_boxes \
            = next(batch_iterator)  # back the "batch_iterator" one by one

        # check data
        if is_debug:
            current_img = cv2.imread(current_img_path)
            next_img = cv2.imread(next_img_path)
            for i in range(19):
                box_pre = boxes_pre.numpy()
                x1 = int(box_pre[0][i][0] * current_img.shape[1])  # * width
                y1 = int(box_pre[0][i][1] * current_img.shape[0])  # * height
                x2 = int(box_pre[0][i][2] * current_img.shape[1])
                y2 = int(box_pre[0][i][3] * current_img.shape[0])
                point_color = (0, 0, 255)  # BGR
                cv2.rectangle(current_img, (x1, y1), (x2, y2), point_color, 2)
                cv2.putText(current_img, str(i), (x1, y1), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
            cv2.imshow('current image', current_img)
            cv2.waitKey(0)
            extra = [current_img_path, next_img_path]
            # cv2.imshow('next image', next_img)
            # cv2.waitKey(0)

        if args.cuda:
            with torch.no_grad():
                img_pre = Variable(img_pre.cuda())
                img_next = Variable(img_next.cuda())
                boxes_pre = Variable(boxes_pre.cuda())
                boxes_next = Variable(boxes_next.cuda())
                valid_pre = Variable(valid_pre.cuda())
                valid_next = Variable(valid_next.cuda())
                labels = Variable(labels.cuda())
                cur_dla_input = torch.from_numpy(cur_dla_input).cuda().unsqueeze(0)
                next_dla_input = torch.from_numpy(next_dla_input).cuda().unsqueeze(0)

        else:
            img_pre = Variable(img_pre)
            img_next = Variable(img_next)
            boxes_pre = Variable(boxes_pre)
            boxes_next = Variable(boxes_next)
            valid_pre = Variable(valid_pre)
            valid_next = Variable(valid_next)
            labels = Variable(labels)
            cur_dla_input = Variable(cur_dla_input)
            next_dla_input = Variable(next_dla_input)

        # for mod in net.final_net.modules():
        #     print(mod)

        # forward
        t0 = time.time()
        if is_debug:
            out = net(cur_dla_input, next_dla_input, index_pre,
                      index_next, current_valid_boxes, next_valid_boxes, extra)  # return similarity matrix (23*23); why pytorch can create computing graph with mixing numpy arrays and pytorch variables
        else:
            out = net(cur_dla_input, next_dla_input, index_pre, index_next, current_valid_boxes, next_valid_boxes)  # return similarity matrix (23*23); why pytorch can create computing graph with mixing numpy arrays and pytorch variables

        optimizer.zero_grad()
        if valid_pre.ndim == 2:
            valid_pre = valid_pre.unsqueeze(0)  # 1*81 -> 1*1*81
            valid_next = valid_next.unsqueeze(0)
        # np.savetxt('results/out.txt',out["sim"].detach().squeeze().cpu(), fmt='%.2f', delimiter=' ')
        # np.savetxt('results/labels.txt',labels.squeeze().cpu(), fmt='%d', delimiter=' ')
        loss_pre, loss_next, loss_similarity, loss, accuracy_pre, accuracy_next, accuracy, predict_indexes = \
            criterion(out["sim"].cuda(), labels, valid_pre, valid_next, out, current_fair_labels, next_fair_labels) #add para that the loss of FairMOT needs
                                                                                                                    #  4*1*81*81,4*1*81*81, 4*1*81,    4*1*81     (valid position, others is zero) --> 1,1,1,1,1,1,1, 4*1*80
        # loss.requires_grad = True  # RuntimeError: you can only change requires_grad flags of leaf variables.
        # loss.retain_grad()  # 1
        a = list(net.parameters())[0].clone()
        # RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation: [torch.cuda.FloatTensor [1, 1, 80, 80]], which is output 0 of ReluBackward1, is at version 3; expected version 1 instead. Hint: enable anomaly detection to find the operation that failed to compute its gradient, with torch.autograd.set_detect_anomaly(True).
        loss.backward()  # (retain_graph=True, backward twice) only leaf node can save grad
        # nn.utils.clip_grad_norm_(net.parameters(), max_norm=20, norm_type=2)  # Gradient Clipping
        optimizer.step()
        # scheduler.step()
        b = list(net.parameters())[0].clone()
        # print(torch.equal(a.data, b.data))  # whether model parameter is update
        t1 = time.time()

        # for Debug: [x.grad for x in optimizer.param_groups[0]['params']]
        if iteration == 0 or iteration == 30:  # diff final_net.0.weight_0.txt final_net.0.weight_100.txt
            for name, param in net.named_parameters():
                if name == 'model.base.base_layer.0.weight' or name == 'model.ida_up.node_2.conv.weight' or name == 'model.hm.0.weight':
                    np.savetxt('results/' + name + '_' + str(iteration) + '.txt', param.detach().cpu()[0,0,:,:],
                               fmt='%.10f', delimiter=' ')
                    print(param.requires_grad)
                if name == 'final_net.0.weight':
                    print(param.requires_grad)
                    np.savetxt('results/final_net.0.weight_' + str(iteration) + '.txt', param.detach().squeeze().cpu(),
                               fmt='%.10f', delimiter=' ')

        all_epoch_loss += [loss.data.cpu()]

        if iteration % 10 == 0:
            print('Timer: %.4f sec.' % (t1 - t0))
            print('iter ' + repr(iteration) + ', ' + repr(epoch_size) + ' || epoch: %.4f ' % (iteration/(float)(epoch_size)) + ' || Loss: %.4f ||' % (loss.item()), end=' ')
        if iteration % 999 == 0:
            torch.save(net.state_dict(), './weights/ddt_' + str(iteration) + '.pth')
            c = (0, 0, 4, 0, 3, 3)
            # validate(c)

        if args.tensorboard:
            if len(all_epoch_loss) > 30:
                writer.add_scalar('data/epoch_loss', float(np.mean(all_epoch_loss)), iteration)
            writer.add_scalar('data/learning_rate', current_lr, iteration)

            writer.add_scalar('loss/loss', loss.data.cpu(), iteration)
            writer.add_scalar('loss/loss_pre', loss_pre.data.cpu(), iteration)
            writer.add_scalar('loss/loss_next', loss_next.data.cpu(), iteration)
            writer.add_scalar('loss/loss_similarity', loss_similarity.data.cpu(), iteration)
            writer.add_scalar('accuracy/accuracy', accuracy.data.cpu(), iteration)
            writer.add_scalar('accuracy/accuracy_pre', accuracy_pre.data.cpu(), iteration)
            writer.add_scalar('accuracy/accuracy_next', accuracy_next.data.cpu(), iteration)

            # add weights
            if iteration % 1000 == 0:
                for name, param in net.named_parameters():
                    writer.add_histogram(name, param.clone().cpu().data.numpy(), iteration)

            # add images
            # if args.send_images and iteration % 1000 == 0:
            #     result_image = show_batch_circle_image(img_pre, img_next, boxes_pre, boxes_next, valid_pre, valid_next, predict_indexes, iteration)
            #
            #     writer.add_image('WithLabel/ImageResult', vutils.make_grid(result_image, nrow=2, normalize=True, scale_each=True), iteration)

        if iteration % save_weights_iteration == 0:
            print('Saving state, iter:', iteration)
            torch.save(sst_net.state_dict(),
                       os.path.join(
                           args.save_folder,
                           'sst300_0712_' + repr(iteration) + '.pth'))

    torch.save(sst_net.state_dict(), args.save_folder + '' + args.version + '.pth')


def validate(choice=None):
    # dataset_index = [2, 4, 5, 9, 10, 11, 13]
    dataset_index = [2]
    # dataset_detection_type = {'-DPM', '-FRCNN', '-SDP'}
    dataset_detection_type = {'-SDP'}

    type = 'train'
    mot_version = 17
    log_folder = config['log_folder']
    dataset_image_folder_format = os.path.join(args.mot_root, type+'/MOT'+str(mot_version)+'-{:02}{}/img1')
    detection_file_name_format=os.path.join(args.mot_root, type+'/MOT'+str(mot_version)+'-{:02}{}/det/det.txt')

    if not os.path.exists(log_folder):
        os.mkdir(log_folder)

    save_folder = ''
    choice_str = ''
    if not choice is None:
        choice_str = TrackerConfig.get_configure_str(choice)
        save_folder = os.path.join(log_folder, choice_str)
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        # else:
        #     return

    saved_file_name_format = os.path.join(save_folder, 'MOT'+str(mot_version)+'-{:02}{}.txt')
    save_video_name_format = os.path.join(save_folder, 'MOT'+str(mot_version)+'-{:02}{}.avi')

    f = lambda format_str: [format_str.format(index, type) for type in dataset_detection_type for index in dataset_index]

    timer = Timer()
    accs = []
    for image_folder, detection_file_name, saved_file_name, save_video_name in zip(f(dataset_image_folder_format), f(detection_file_name_format), f(saved_file_name_format), f(save_video_name_format)):
        tracker = SSTTracker()
        print('start processing '+saved_file_name)
        reader = MOTDataReader(image_folder = image_folder,
                      detection_file_name =detection_file_name,
                               min_confidence=0.0)
        result = list()
        result_str = saved_file_name
        first_run = True
        for i, item in enumerate(reader):
            if i > len(reader):
                break

            if item is None:
                continue

            img = item[0]
            det = item[1]

            if img is None or det is None or len(det)==0:
                continue

            if len(det) > config['max_object']:
                det = det[:config['max_object'], :]

            h, w, _ = img.shape

            save_video = True
            show_image = True

            if first_run and save_video:
                vw = cv2.VideoWriter(save_video_name, cv2.VideoWriter_fourcc('M','J','P','G'), 10, (w, h))
                first_run = False

            det[:, [2,4]] /= float(w)
            det[:, [3,5]] /= float(h)
            timer.tic()
            image_org = tracker.update(img, show_image, i)  # SST tracker entrance
            timer.toc()
            print('{}:{}, {}, {}\r'.format(os.path.basename(saved_file_name), i, int(i*100/len(reader)), choice_str))
            if show_image and not image_org is None:
                cv2.imshow('res', image_org)
                cv2.waitKey(1)

            if save_video and not image_org is None:
                vw.write(image_org)

            # save result
            for t in tracker.tracks:
                n = t.nodes[-1]
                if t.age == 1:
                    b = n.get_box(tracker.frame_index-1, tracker.recorder)
                    result.append(
                        [i+1] + [t.id+1] + [b[0]*w] + [b[1]*h] + [b[2]*w] + [b[3]*h] + [-1] + [-1] + [-1] + [-1]
                    )
        # save data
        np.savetxt(saved_file_name, np.array(result), fmt="%d,%d,%s,%s,%s,%s,%d,%d,%d,%d", delimiter=',')
        print(result_str)
    print('Tracking total time: ', timer.total_time)    # 200.08386325836182
    print('Tracking average time: ', timer.average_time)  # 0.33347310543060305 (Idea: 6.3 Hz)

    # compute acc
    timer = Timer()
    timer.tic()
    seqs_str = '''MOT17-02-SDP'''
    seqs = [seq.strip() for seq in seqs_str.split()]
    result_dir = '/data2/whd/workspace/MOT/SST/results/log_folder/train/0_0_4_0_3_3'
    accs = []
    for seq in seqs:
        print("Evaluating " + seq)
        saved_file_name = os.path.join(result_dir, seq + '.txt')
        data_type = 'mot'
        data_root = os.path.join(args.mot_root, 'train')
        evaluator = Evaluator(data_root, seq, data_type)
        accs.append(evaluator.eval_file(saved_file_name))

    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, seqs, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    timer.toc()
    print('Evaluation time: ', timer.total_time)
    print(strsummary)


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


# mot --exp_id all_dla34 --gpus 0,1 --batch_size 2 --load_model ../models/ctdet_coco_dla_2x.pth --num_workers 20
if __name__ == '__main__':
    # c = (0, 0, 4, 0, 3, 3)
    # validate(c)
    train()


'''
iter 79710, 1329 || epoch: 59.9774  || Loss: 0.1366 || Timer: 0.4418 sec.
10:00-17:16: cost 19 hours
'''