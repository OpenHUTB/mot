from layer.sst import build_sst
from config.config import config
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

from torchvision import transforms
from outils.augmentations import *


class TrackUtil:
    @staticmethod
    def letterbox(img, height=608, width=1088,
                  color=(127.5, 127.5, 127.5)):  # resize a rectangular image to a padded rectangular
        shape = img.shape[:2]  # shape = [height, width]
        ratio = min(float(height) / shape[0], float(width) / shape[1])
        new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))  # new_shape = [width, height]
        dw = (width - new_shape[0]) / 2  # width padding
        dh = (height - new_shape[1]) / 2  # height padding
        top, bottom = round(dh - 0.1), round(dh + 0.1)
        left, right = round(dw - 0.1), round(dw + 0.1)
        img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded rectangular
        return img, ratio, dw, dh

    @staticmethod
    def convert_detection(detection):
        '''
        transform the current detection center to [-1, 1]
        :param detection: detection
        :return: translated detection
        '''
        # get the center, and format it in (-1, 1)
        center = (2 * detection[:, 0:2] + detection[:, 2:4]) - 1.0
        center = torch.from_numpy(center.astype(float)).float()
        center.unsqueeze_(0)
        center.unsqueeze_(2)
        center.unsqueeze_(3)

        if TrackerConfig.cuda:
            return Variable(center.cuda())
        return Variable(center)

    @staticmethod
    def convert_image(image):
        '''
        transform image to the FloatTensor (1, 3,size, size)
        :param image: same as update parameter
        :return: the transformed image FloatTensor (i.e. 1x3x900x900)
        '''
        image = cv2.resize(image, TrackerConfig.image_size).astype(np.float32)
        image -= TrackerConfig.mean_pixel
        image = torch.FloatTensor(image)
        image = image.permute(2, 0, 1)
        image.unsqueeze_(dim=0)
        if TrackerConfig.cuda:
            return Variable(image.cuda())
        return Variable(image)

    @staticmethod
    def get_iou(pre_boxes, next_boxes):
        h = len(pre_boxes)
        w = len(next_boxes)
        if h == 0 or w == 0:
            return []

        iou = np.zeros((h, w), dtype=float)
        for i in range(h):
            b1 = np.copy(pre_boxes[i, :])
            b1[2:] = b1[:2] + b1[2:]
            for j in range(w):
                b2 = np.copy(next_boxes[j, :])
                b2[2:] = b2[:2] + b2[2:]
                delta_h = min(b1[2], b2[2]) - max(b1[0], b2[0])
                delta_w = min(b1[3], b2[3])-max(b1[1], b2[1])
                if delta_h < 0 or delta_w < 0:
                    expand_area = (max(b1[2], b2[2]) - min(b1[0], b2[0])) * (max(b1[3], b2[3]) - min(b1[1], b2[1]))
                    area = (b1[2] - b1[0]) * (b1[3] - b1[1]) + (b2[2] - b2[0]) * (b2[3] - b2[1])
                    iou[i,j] = -(expand_area - area) / area
                else:
                    overlap = delta_h * delta_w
                    area = (b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - max(overlap, 0)
                    iou[i,j] = overlap / area

        return iou

    @staticmethod
    def get_node_similarity(n1, n2, frame_index, recorder):
        if n1.frame_index > n2.frame_index:
            n_max = n1
            n_min = n2
        elif n1.frame_index < n2.frame_index:
            n_max = n2
            n_min = n1
        else: # in the same frame_index
            return None

        f_max = n_max.frame_index
        f_min = n_min.frame_index

        # not recorded in recorder
        if frame_index - f_min >= TrackerConfig.max_track_node:
            return None

        return recorder.all_similarity[f_max][f_min][n_min.id, n_max.id]

    @staticmethod
    def get_merge_similarity(t1, t2, frame_index, recorder):
        '''
        Get the similarity between two tracks
        :param t1: track 1
        :param t2: track 2
        :param frame_index: current frame_index
        :param recorder: recorder
        :return: the similairty (float value). if valid, return None
        '''
        merge_value = []
        if t1 is t2:
            return None

        all_f1 = [n.frame_index for n in t1.nodes]
        all_f2 = [n.frame_index for n in t2.nodes]

        for i, f1 in enumerate(all_f1):
            for j, f2 in enumerate(all_f2):
                compare_f = [f1 + 1, f1 - 1]
                for f in compare_f:
                    if f not in all_f1 and f == f2:
                        n1 = t1.nodes[i]
                        n2 = t2.nodes[j]
                        s = TrackUtil.get_node_similarity(n1, n2, frame_index, recorder)
                        if s is None:
                            continue
                        merge_value += [s]

        if len(merge_value) == 0:
            return None
        return np.mean(np.array(merge_value))

    @staticmethod
    def merge(t1, t2):
        '''
        merge t2 to t1, after that t2 is set invalid
        :param t1: track 1
        :param t2: track 2
        :return: None
        '''
        all_f1 = [n.frame_index for n in t1.nodes]
        all_f2 = [n.frame_index for n in t2.nodes]

        for i, f2 in enumerate(all_f2):
            if f2 not in all_f1:
                insert_pos = 0
                for j, f1 in enumerate(all_f1):
                    if f2 < f1:
                        break
                    insert_pos += 1
                t1.nodes.insert(insert_pos, t2.nodes[i])

        # remove some nodes in t1 in order to keep t1 satisfy the max nodes
        if len(t1.nodes) > TrackerConfig.max_track_node:
            t1.nodes = t1.nodes[-TrackerConfig.max_track_node:]
        t1.age = min(t1.age, t2.age)
        t2.valid = False

class TrackerConfig:
    max_record_frame = 30
    max_track_age = 30
    max_track_node = 30
    max_draw_track_node = 30

    max_object = config['max_object']
    sst_model_path = config['resume']
    cuda = config['cuda']
    mean_pixel = config['mean_pixel']
    image_size = (config['sst_dim'], config['sst_dim'])

    min_iou_frame_gap = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    min_iou = [0.3, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -7.0]
    # min_iou = [pow(0.3, i) for i in min_iou_frame_gap]

    min_merge_threshold = 0.9

    max_bad_node = 0.9

    decay = 0.995

    roi_verify_max_iteration = 2
    roi_verify_punish_rate = 0.6

    @staticmethod
    def set_configure(all_choice):

        min_iou_frame_gaps = [
            # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
            ]
        min_ious = [
            # [0.4, 0.3, 0.25, 0.2, 0.1, 0.0, -1.0, -2.0, -3.0, -4.0, -4.5, -5.0, -5.5, -6.0, -6.5, -7.0],
            [0.3, 0.1, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0],
            [0.3, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -7.0],
            [0.2, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -7.0],
            [0.1, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -7.0],
            [-1.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0],
            [0.4, 0.3, 0.25, 0.2, 0.1, 0.0, -1.0, -2.0, -3.0, -4.0, -4.5, -5.0, -5.5, -6.0, -6.5, -7.0],
        ]

        decays = [1-0.01*i for i in range(11)]

        roi_verify_max_iterations = [2, 3, 4, 5, 6]

        roi_verify_punish_rates = [0.6, 0.4, 0.2, 0.1, 0.0, 1.0]

        max_track_ages = [i*3 for i in range(1,11)]
        max_track_nodes = [i*3 for i in range(1,11)]

        if all_choice is None:
            return
        TrackerConfig.min_iou_frame_gap = min_iou_frame_gaps[all_choice[0]]
        TrackerConfig.min_iou = min_ious[all_choice[0]]
        TrackerConfig.decay = decays[all_choice[1]]
        TrackerConfig.roi_verify_max_iteration = roi_verify_max_iterations[all_choice[2]]
        TrackerConfig.roi_verify_punish_rate = roi_verify_punish_rates[all_choice[3]]
        TrackerConfig.max_track_age = max_track_ages[all_choice[4]]
        TrackerConfig.max_track_node = max_track_nodes[all_choice[5]]

    @staticmethod
    def get_configure_str(all_choice):
        return "{}_{}_{}_{}_{}_{}".format(all_choice[0], all_choice[1], all_choice[2], all_choice[3], all_choice[4], all_choice[5])

    @staticmethod
    def get_all_choices():
        # return [(1, 1, 0, 0, 4, 2)]
        return [(i1, i2, i3, i4, i5, i6) for i1 in range(5) for i2 in range(5) for i3 in range(5) for i4 in range(5) for i5 in range(5) for i6 in range(5)]

    @staticmethod
    def get_all_choices_decay():
        return [(1, i2, 0, 0, 4, 2) for i2 in range(11)]

    @staticmethod
    def get_all_choices_max_track_node():
        return [(1, i2, 0, 0, 4, 2) for i2 in range(11)]

    @staticmethod
    def get_choices_age_node():
        return [(0, 0, 0, 0, a, n) for a in range(10) for n in range(10)]

    @staticmethod
    def get_ua_choice():
        return (5, 0, 4, 1, 5, 5)


class FeatureRecorder:
    '''
    Record features and boxes every frame
    '''

    def __init__(self):
        self.max_record_frame = TrackerConfig.max_record_frame
        self.all_frame_index = np.array([], dtype=int)
        self.all_features = {}
        self.all_boxes = {}
        self.all_similarity = {}
        self.all_iou = {}

    def update(self, sst, frame_index, features, boxes):
        # if the coming frame in the new frame
        if frame_index not in self.all_frame_index:
            # if the recorder have reached the max_record_frame.
            if len(self.all_frame_index) == self.max_record_frame:  # max record frame number: 30
                del_frame = self.all_frame_index[0]
                del self.all_features[del_frame]
                del self.all_boxes[del_frame]
                del self.all_similarity[del_frame]
                del self.all_iou[del_frame]
                self.all_frame_index = self.all_frame_index[1:]

            # add new item for all_frame_index, all_features and all_boxes. Besides, also add new similarity
            self.all_frame_index = np.append(self.all_frame_index, frame_index)
            self.all_features[frame_index] = features
            self.all_boxes[frame_index] = boxes

            self.all_similarity[frame_index] = {}
            for pre_index in self.all_frame_index[:-1]:  # [:-1]: remove the last element in 'all_frame_index'; [::-1] reverse
                delta = pow(TrackerConfig.decay, (frame_index - pre_index)/3.0)
                pre_similarity = sst.forward_stacker_final(Variable(self.all_features[pre_index]), Variable(features), fill_up_column=False)  # get similarity with previous frame
                self.all_similarity[frame_index][pre_index] = pre_similarity*delta  # save similarity matrix according to time distance

            self.all_iou[frame_index] = {}
            for pre_index in self.all_frame_index[:-1]:
                iou = TrackUtil.get_iou(self.all_boxes[pre_index], boxes)
                self.all_iou[frame_index][pre_index] = iou
        else:
            self.all_features[frame_index] = features
            self.all_boxes[frame_index] = boxes
            index = self.all_frame_index.__index__(frame_index)

            for pre_index in self.all_frame_index[:index+1]:
                if pre_index == self.all_frame_index[-1]:
                    continue

                pre_similarity = sst.forward_stacker_features(Variable(self.all_features[pre_index]), Variable(self.all_features[-1]))
                self.all_similarity[frame_index][pre_index] = pre_similarity

                iou = TrackUtil.get_iou(self.all_boxes[pre_index], boxes)
                self.all_similarity[frame_index][pre_index] = iou

    def get_feature(self, frame_index, detection_index):
        '''
        get the feature by the specified frame index and detection index
        :param frame_index: start from 0
        :param detection_index: start from 0
        :return: the corresponding feature at frame index and detection index
        '''

        if frame_index in self.all_frame_index:
            features = self.all_features[frame_index]
            if len(features) == 0:
                return None
            if detection_index < len(features):
                return features[detection_index]

        return None

    def get_box(self, frame_index, detection_index):
        if frame_index in self.all_frame_index:
            boxes = self.all_boxes[frame_index]
            if len(boxes) == 0:
                return None

            if detection_index < len(boxes):
                return boxes[detection_index]
        return None

    def get_features(self, frame_index):
        if frame_index in self.all_frame_index:
            features = self.all_features[frame_index]
        else:
            return None
        if len(features) == 0:
            return None
        return features

    def get_boxes(self, frame_index):
        if frame_index in self.all_frame_index:
            boxes = self.all_boxes[frame_index]
        else:
            return None

        if len(boxes) == 0:
            return None
        return boxes

class Node:
    '''
    The Node is the basic element of a track. it contains the following information:
    1) extracted feature (it'll get removed when it isn't active
    2) box (a box (l, t, r, b)
    3) label (active label indicating keeping the features)
    4) detection, the formated box
    '''
    def __init__(self, frame_index, id):
        self.frame_index = frame_index
        self.id = id

    def get_box(self, frame_index, recoder):
        if frame_index - self.frame_index >= TrackerConfig.max_record_frame:
            return None
        return recoder.all_boxes[self.frame_index][self.id, :]

    def get_iou(self, frame_index, recoder, box_id):
        if frame_index - self.frame_index >= TrackerConfig.max_track_node:
            return None
        return recoder.all_iou[frame_index][self.frame_index][self.id, box_id]

class Track:
    '''
    轨迹 包含所有的节点
    Track is the class of track. it contains all the node and manages the node. it contains the following information:
    1) all the nodes
    2) track id. it is unique it identify each track
    3) track pool id. it is a number to give a new id to a new track
    4) age. age indicates how old is the track
    5) max_age. indicates the dead age of this track
    '''
    _id_pool = 0

    def __init__(self):
        self.nodes = list()
        self.id = Track._id_pool
        Track._id_pool += 1
        self.age = 0
        self.valid = True   # indicate this track is merged
        self.color = tuple((np.random.rand(3) * 255).astype(int).tolist())

    def __del__(self):
        for n in self.nodes:
            del n

    def add_age(self):
        self.age += 1

    def reset_age(self):
        self.age = 0

    def add_node(self, frame_index, recorder, node):
        # iou judge
        if len(self.nodes) > 0:
            n = self.nodes[-1]
            iou = n.get_iou(frame_index, recorder, node.id)
            delta_frame = frame_index - n.frame_index
            if delta_frame in TrackerConfig.min_iou_frame_gap:
                iou_index = TrackerConfig.min_iou_frame_gap.index(delta_frame)
                # if iou < TrackerConfig.min_iou[iou_index]:
                if iou < TrackerConfig.min_iou[-1]:
                    return False
        self.nodes.append(node)
        self.reset_age()
        return True

    def get_similarity(self, frame_index, recorder):
        similarity = []
        for n in self.nodes:
            f = n.frame_index  # previous frame index
            id = n.id  # 12
            if frame_index - f >= TrackerConfig.max_track_node:
                continue
            similarity += [recorder.all_similarity[frame_index][f][id, :]]
            # sim_mat = recorder.all_similarity[frame_index][f]
            # sim_mat = sim_mat.squeeze()
            # similarity += [sim_mat[id, :]]  # 12*11 todo: IndexError: index 12 is out of bounds for axis 0 with size 12

        if len(similarity) == 0:
            return None
        a = np.array(similarity)
        return np.sum(np.array(similarity), axis=0)

    def verify(self, frame_index, recorder, box_id):
        for n in self.nodes:
            delta_f = frame_index - n.frame_index
            if delta_f in TrackerConfig.min_iou_frame_gap:
                iou_index = TrackerConfig.min_iou_frame_gap.index(delta_f)
                iou = n.get_iou(frame_index, recorder, box_id)
                if iou is None:
                    continue
                if iou < TrackerConfig.min_iou[iou_index]:
                    return False
        return True

class Tracks:
    '''
    轨迹的集合 包含并管理所有的轨迹
    Track set. It contains all the tracks and manage the tracks. it has the following information
    1) tracks. the set of tracks
    2) keep the previous image and features
    '''
    def __init__(self):
        self.tracks = list() # the set of tracks
        self.max_drawing_track = TrackerConfig.max_draw_track_node


    def __getitem__(self, item):
        return self.tracks[item]

    def append(self, track):
        self.tracks.append(track)
        self.volatile_tracks()

    def volatile_tracks(self):
        if len(self.tracks) > TrackerConfig.max_object:
            # start to delete the most oldest tracks
            all_ages = [t.age for t in self.tracks]
            oldest_track_index = np.argmax(all_ages)
            del self.tracks[oldest_track_index]

    def get_track_by_id(self, id):
        for t in self.tracks:
            if t.id == id:
                return t
        return None

    def get_similarity(self, frame_index, recorder):
        ids = []
        similarity = []
        for t in self.tracks:   # current tracked object 14;
            s = t.get_similarity(frame_index, recorder)  # historical recorder number 16 (14*16)
            if s is None:
                continue
            similarity += [s]
            ids += [t.id]

        similarity = np.array(similarity)  # 21*81

        track_num = similarity.shape[0]
        if track_num > 0:
            box_num = similarity.shape[1]  # 1->0 {IndexError}tuple index out of range
        else:
            box_num = 0

        if track_num == 0 :
            return np.array(similarity), np.array(ids)

        similarity = np.repeat(similarity, [1]*(box_num-1)+[track_num], axis=1)  # numpy.AxisError: axis 1 is out of bounds for array of dimension 1
        return np.array(similarity), np.array(ids)

    def one_frame_pass(self):
        keep_track_set = list()
        for i, t in enumerate(self.tracks):
            t.add_age()
            if t.age > TrackerConfig.max_track_age:
                continue
            keep_track_set.append(i)

        self.tracks = [self.tracks[i] for i in keep_track_set]

    def merge(self, frame_index, recorder):
        t_l = len(self.tracks)
        res = np.zeros((t_l, t_l), dtype=float)
        # get track similarity matrix
        for i, t1 in enumerate(self.tracks):
            for j, t2 in enumerate(self.tracks):
                s = TrackUtil.get_merge_similarity(t1, t2, frame_index, recorder)
                if s is None:
                    res[i, j] = 0
                else:
                    res[i, j] = s

        # get the track pair which needs merged
        used_indexes = []
        merge_pair = []
        for i, t1 in enumerate(self.tracks):
            if i in used_indexes:
                continue
            max_track_index = np.argmax(res[i, :])
            if i != max_track_index and res[i, max_track_index] > TrackerConfig.min_merge_threshold:
                used_indexes += [max_track_index]
                merge_pair += [(i, max_track_index)]

        # start merge
        for i, j in merge_pair:
            TrackUtil.merge(self.tracks[i], self.tracks[j])

        # remove the invalid tracks
        self.tracks = [t for t in self.tracks if t.valid]


    def show(self, frame_index, recorder, image):
        h, w, _ = image.shape

        # draw rectangle
        for t in self.tracks:
            if len(t.nodes) > 0 and t.age < 2:
                b = t.nodes[-1].get_box(frame_index, recorder)
                if b is None:
                    continue
                txt = '({}, {})'.format(t.id, t.nodes[-1].id)
                image = cv2.putText(image, txt, (int(b[0]*w),int((b[1])*h)), cv2.FONT_HERSHEY_SIMPLEX, 1, t.color, 3)
                image = cv2.rectangle(image, (int(b[0]*w),int((b[1])*h)), (int((b[0]+b[2])*w), int((b[1]+b[3])*h)), t.color, 2)

        # draw line
        for t in self.tracks:
            if t.age > 1:
                continue
            if len(t.nodes) > self.max_drawing_track:
                start = len(t.nodes) - self.max_drawing_track
            else:
                start = 0
            for n1, n2 in zip(t.nodes[start:], t.nodes[start+1:]):
                b1 = n1.get_box(frame_index, recorder)
                b2 = n2.get_box(frame_index, recorder)
                if b1 is None or b2 is None:
                    continue
                c1 = (int((b1[0] + b1[2]/2.0)*w), int((b1[1] + b1[3])*h))
                c2 = (int((b2[0] + b2[2] / 2.0) * w), int((b2[1] + b2[3]) * h))
                image = cv2.line(image, c1, c2, t.color, 2)

        return image

# The tracker is compatible with pytorch (cuda)
class SSTTracker:
    def __init__(self):
        Track._id_pool = 0
        self.first_run = True
        self.image_size = TrackerConfig.image_size
        self.model_path = TrackerConfig.sst_model_path
        self.cuda = TrackerConfig.cuda
        self.mean_pixel = TrackerConfig.mean_pixel
        self.max_object = TrackerConfig.max_object
        self.frame_index = 0
        self.load_model()
        self.recorder = FeatureRecorder()
        self.tracks = Tracks()

    def load_model(self):
        # load the model [ 0(512*1024*1*1), 3(256*512*1*1), 6(128*256*1*1), 9(64*128*1*1), 11(1*64*1*1) ]
        self.sst = build_sst('test', 900)  # final_net.0.weight: 512*1024*1*1 (512*1040*1*1); final_net.0.bias: 512
        if self.cuda:
            cudnn.benchmark = True
            # self.sst = torch.nn.DataParallel(self.sst)
            resume_model = './weights/ddt_27972.pth'  # ddt_sst.pth, ddt_999(150), ddt_21978(80)
            self.sst.load_state_dict(torch.load(resume_model))

            # load sst final net parameters (except final_net.0.)
            # model_dict = self.sst.state_dict()
            # sst_pretrained = torch.load('./weights/sst300_0712_3320.pth')  # copy from '/data2/whd/workspace/MOT/FairMOT/SST/results/train/save_folder/1031-E120-M80-G30-weights/sst300_0712_3320.pth'
            # final_net_dict = dict((key, value) for key, value in sst_pretrained.items() if key.startswith('final_net.') > 0 and key != 'final_net.0.bias' and key != 'final_net.0.weight')  # filter
            # model_dict.update(final_net_dict)
            # self.sst.load_state_dict(model_dict)

            # pretrained_dict = torch.load(config['resume'])  # load sst weight from 'sst.pth'
            # model_dict = self.sst.state_dict()
            # model_dict.update(pretrained_dict)
            # torch.save(self.sst.state_dict(), './weights/ddt_sst.pth')  # save resume model for training
            self.sst = self.sst.cuda()
        else:
            self.sst.load_state_dict(torch.load(config['resume'], map_location='cpu'))
        self.sst.eval()

    def update(self, image, show_image, frame_index, force_init=False):
        '''
        Update the state of tracker, the following jobs should be done:
        1) extract the features
        2) stack the features together
        3) get the similarity matrix
        4) do assignment work
        5) save the previous image
        :param image: the opencv readed image, format is hxwx3
        :param detections: detection array. numpy array (l, r, w, h) and they all formated in (0, 1)
        '''

        self.frame_index = frame_index

        # format the image and detection (xxx)
        # h, w, _ = image.shape
        h = 1080  # fix the MOT17-06: 640*480, the output is percentage in the current frame.
        w = 1920
        image_org = np.copy(image)
        # image = TrackUtil.convert_image(image)
        # detection_org = np.copy(detection)
        # detection = TrackUtil.convert_detection(detection)  # get the center, and format it in (-1, 1): 16*4 --> 1*16*1*1*2

        # format the input image of DLA network
        dla_width = 1088
        dla_height = 608
        mean_pixel = (104, 117, 123)
        # Padded resize
        dla_input, _, _, _ = TrackUtil.letterbox(image_org, height=dla_height, width=dla_width)
        # Normalize RGB
        dla_input = dla_input[:, :, ::-1].transpose(2, 0, 1)
        dla_input = np.ascontiguousarray(dla_input, dtype=np.float32)
        dla_input /= 255.0

        # features can be (1, 10, 450)
        dla_input = torch.from_numpy(dla_input).cuda().unsqueeze(0)

        # # for test (SST input is wrong for DLA network)
        # trans = SingleCompose([
        #     SingleConvertFromInts(),
        #     SingleResize(dla_width, dla_height),
        #     SingleSubtractMeans(mean_pixel),
        #     SingleToTensor()
        # ])
        # dla_input = trans(image_org)
        # dla_input = dla_input.unsqueeze(0).cuda()  # np.expand_dims(dla_input, axis=0)
        outputs, dets, features = self.sst.forward_feature_extracter(dla_input)  # (1*3*900*900, 1*14*1*1*2) --> 1*14*520 (1*12*512)
        # features = self.sst.forward_feature_extracter(image, detection)

        # update recorder
        # dets = dets.cpu().numpy()
        dets[:, 2] = dets[:, 2] - dets[:, 0]  # (x1, y1, x2, y2) -> (x1, y1, w, h)
        dets[:, 3] = dets[:, 3] - dets[:, 1]
        dets = dets[:, 0:4]
        dets[:, [0, 2]] /= float(w)  # / (x1, width) width
        dets[:, [1, 3]] /= float(h)  # / (y1, height) height
        # detection_org = dets
        self.recorder.update(self.sst, self.frame_index, outputs, dets)  # features.data: 1*14*520, 14*4

        if self.frame_index == 0 or force_init or len(self.tracks.tracks) == 0:  # first frame (SST: 1*12*512, DLA: )
            for i in range(dets.shape[0]):  # use the size of DLA detection result
                t = Track()
                n = Node(self.frame_index, i)
                t.add_node(self.frame_index, self.recorder, n)
                self.tracks.append(t)
            self.tracks.one_frame_pass()
            # self.frame_index += 1
            return self.tracks.show(self.frame_index, self.recorder, image_org)

        # get tracks similarity between current frame index (self.frame_index) and history recorders (self.recorder)
        y, ids = self.tracks.get_similarity(self.frame_index, self.recorder)

        if len(y) > 0:
            #3) find the corresponding by the similar matrix
            row_index, col_index = linear_sum_assignment(-y)  # ndarray:14*29; ValueError: expected a matrix (2-d array), got a (1,) array
            col_index[col_index >= dets.shape[0]] = -1

            # verification by iou
            verify_iteration = 0
            while verify_iteration < TrackerConfig.roi_verify_max_iteration:
                is_change_y = False
                for i in row_index:
                    box_id = col_index[i]
                    track_id = ids[i]

                    if box_id < 0:
                        continue
                    t = self.tracks.get_track_by_id(track_id)
                    if not t.verify(self.frame_index, self.recorder, box_id):
                        y[i, box_id] *= TrackerConfig.roi_verify_punish_rate
                        is_change_y = True
                if is_change_y:
                    row_index, col_index = linear_sum_assignment(-y)
                    col_index[col_index >= dets.shape[0]] = -1
                else:
                    break
                verify_iteration += 1

            print(verify_iteration)

            #4) update the tracks
            for i in row_index:
                track_id = ids[i]
                t = self.tracks.get_track_by_id(track_id)
                col_id = col_index[i]
                if col_id < 0:
                    continue
                node = Node(self.frame_index, col_id)
                t.add_node(self.frame_index, self.recorder, node)

            #5) add new track
            for col in range(len(dets)):
                if col not in col_index:
                    node = Node(self.frame_index, col)
                    t = Track()
                    t.add_node(self.frame_index, self.recorder, node)
                    self.tracks.append(t)

        # remove the old track
        self.tracks.one_frame_pass()

        # merge the tracks
        # if self.frame_index % 20 == 0:
        #     self.tracks.merge(self.frame_index, self.recorder)

        # if show_image:
        image_org = self.tracks.show(self.frame_index, self.recorder, image_org)
        # self.frame_index += 1
        return image_org

        # self.frame_index += 1
        # image_org = cv2.resize(image_org, (320, 240))
        # vw.write(image_org)

        # plt.imshow(image_org)
