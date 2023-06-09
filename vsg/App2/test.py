# from __future__ import print_function
import numpy as np
from munkres import Munkres
import torch.nn.functional as F
import time, os, shutil
from global_set import test_gt_det, \
    tau_threshold, gap, f_gap, show_recovering, decay, u_update, mot_dataset_dir, model_dir
from test_dataset import DatasetFromFolder
from mot_model import *

torch.manual_seed(123)
np.random.seed(123)


def deleteDir(del_dir):
    shutil.rmtree(del_dir)


year = 17

type = ''
t_dir = ''  # the dir of tracking results
sequence_dir = ''  # the dir of the testing video sequence

seqs = [2, 4, 5, 9, 10, 11, 13]  # the set of sequences
lengths = [600, 1050, 837, 525, 654, 900, 750]  # the length of the sequence

test_seqs = [1, 3, 6, 7, 8, 12, 14]
test_lengths = [450, 1500, 1194, 500, 625, 900, 750]

# test on train dataset and test dataset
test_seqs = test_seqs + seqs
test_lengths = test_lengths + lengths

tt_tag = 1  # 1 - test, 0 - train

tau_conf_score = 0.0


class GN():
    def __init__(self, seq_index, seq_len, type, cuda=True):
        '''
        Evaluating with the MotMetrics
        :param seq_index: the number of the sequence
        :param seq_len: the length of the sequence
        :param cuda: True - GPU, False - CPU
        '''
        self.seq_index = seq_index
        self.type = type

        self.hungarian = Munkres()
        self.device = torch.device("cuda" if cuda else "cpu")
        self.seq_len = seq_len
        self.missingCounter = 0
        self.sideConnection = 0

        print('     Loading the model...')
        self.loadModel()

        self.out_dir = t_dir
        # self.out_dir = t_dir + 'motmetrics_%s/' % (type)

        print(self.out_dir)
        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)
        # else:
        #     deleteDir(self.out_dir)
        #     os.mkdir(self.out_dir)
        self.initOut()

    def initOut(self):
        print('     Loading Data...')
        self.train_set = DatasetFromFolder(sequence_dir, mot_dataset_dir + 'MOT16/test/MOT16-%02d' % self.seq_index,
                                           tau_conf_score)

        detection_dir = self.out_dir + 'det_MOT%s-%02d-%s.txt' % (year, self.seq_index, self.type)  #
        # MOT17-01-DPM.txt
        res_training = self.out_dir + 'MOT%s-%02d-%s.txt' % (year, self.seq_index, self.type)  # the tracking results
        self.createTxt(detection_dir)
        self.createTxt(res_training)
        self.copyLines(self.seq_index, 1, detection_dir, self.seq_len, 1)

        self.evaluation(1, self.seq_len, detection_dir, res_training)

    def getSeqL(self, info):
        # get the length of the sequence
        f = open(info, 'r')
        f.readline()
        for line in f.readlines():
            line = line.strip().split('=')
            if line[0] == 'seqLength':
                seqL = int(line[1])
        f.close()
        return seqL

    def copyLines(self, seq, head, gt_seq, tail=-1, tag=0):
        '''
        Copy the groun truth within [head, head+num]
        :param seq: the number of the sequence
        :param head: the head frame number
        :param tail: the number the clipped sequence
        :param gt_seq: the dir of the output file
        :return: None
        '''
        if tt_tag:
            basic_dir = mot_dataset_dir + 'MOT%d/test/MOT%d-%02d-%s/' % (year, year, seq, type)
        else:
            basic_dir = mot_dataset_dir + 'MOT%d/train/MOT%d-%02d-%s/' % (year, year, seq, type)
        print('     Testing on', basic_dir, 'Length:', self.seq_len)
        seqL = tail if tail != -1 else self.getSeqL(basic_dir + 'seqinfo.ini')

        det_dir = 'gt/gt_det.txt' if test_gt_det else 'det/det.txt'
        seq_dir = basic_dir + ('gt/gt.txt' if tag == 0 else det_dir)
        inStream = open(seq_dir, 'r')

        outStream = open(gt_seq, 'w')
        for line in inStream.readlines():
            line = line.strip()
            attrs = line.split(',')
            f_num = int(attrs[0])
            if f_num >= head and f_num <= seqL:
                outStream.write(line + '\n')
        outStream.close()

        inStream.close()
        return seqL

    def createTxt(self, out_file):
        f = open(out_file, 'w')
        f.close()

    def loadModel(self):
        tail = 13
        self.Uphi = torch.load(model_dir + 'uphi_%02d.pth' % tail).to(self.device)
        self.Vphi = torch.load(model_dir + 'vphi_%02d.pth' % tail).to(self.device)
        self.Ephi1 = torch.load(model_dir + 'ephi1_%02d.pth' % tail).to(self.device)
        self.Ephi2 = torch.load(model_dir + 'ephi2_%02d.pth' % tail).to(self.device)
        self.u = torch.load(model_dir + 'u_%02d.pth' % tail).to(self.device)

    def swapFC(self):
        self.cur = self.cur ^ self.nxt
        self.nxt = self.cur ^ self.nxt
        self.cur = self.cur ^ self.nxt

    def linearModel(self, out, attr1, attr2):
        # print 'I got you! *.*'
        t = attr1[-1]
        self.sideConnection += 1
        if t > f_gap:
            return
        frame = int(attr1[0])
        x1, y1, w1, h1 = float(attr1[2]), float(attr1[3]), float(attr1[4]), float(attr1[5])
        x2, y2, w2, h2 = float(attr2[2]), float(attr2[3]), float(attr2[4]), float(attr2[5])

        x_delta = (x2 - x1) / t
        y_delta = (y2 - y1) / t
        w_delta = (w2 - w1) / t
        h_delta = (h2 - h1) / t

        for i in range(1, t):
            frame += 1
            x1 += x_delta
            y1 += y_delta
            w1 += w_delta
            h1 += h_delta
            attr1[0] = str(frame)
            attr1[2] = str(x1)
            attr1[3] = str(y1)
            attr1[4] = str(w1)
            attr1[5] = str(h1)
            line = ''
            for attr in attr1[:-1]:
                line += attr + ','
            if show_recovering:
                line += '1'
            else:
                line = line[:-1]
            out.write(line + '\n')
        self.missingCounter += t - 1

    def evaluation(self, head, tail, gtFile, outFile):
        '''
        Evaluation on dets
        :param head: the head frame number
        :param tail: the tail frame number
        :param gtFile: the ground truth file name
        :param outFile: the name of output file
        :return: None
        '''
        with torch.no_grad():
            gtIn = open(gtFile, 'r')
            self.cur, self.nxt = 0, 1
            line_con = [[], []]
            id_con = [[], []]
            id_step = 1

            step = head + self.train_set.setBuffer(head)
            while step < tail:
                # print '     ', step
                print(step, end=' ')
                if step % 100 == 0:
                    print('')
                t_gap = self.train_set.loadNext()
                step += t_gap
                # print 'F', 't_gap=%d'%t_gap,
                # print 'Fo',

                m = self.train_set.m
                n = self.train_set.n
                # print m, n,
                if n == 0:
                    print('There is no detection in the rest of sequence!')
                    break

                if id_step == 1:
                    out = open(outFile, 'a')
                    i = 0
                    while i < m:
                        attrs = gtIn.readline().strip().split(',')
                        if float(attrs[6]) >= tau_conf_score:
                            attrs.append(1)
                            attrs[1] = str(id_step)
                            line = ''
                            for attr in attrs[:-1]:
                                line += attr + ','
                            if show_recovering:
                                line += '0'
                            else:
                                line = line[:-1]
                            out.write(line + '\n')
                            line_con[self.cur].append(attrs)
                            id_con[self.cur].append(id_step)
                            id_step += 1
                            i += 1
                    out.close()
                # print '0.0',
                i = 0
                while i < n:
                    attrs = gtIn.readline().strip().split(',')
                    if float(attrs[6]) >= tau_conf_score:
                        attrs.append(1)
                        line_con[self.nxt].append(attrs)
                        id_con[self.nxt].append(-1)
                        i += 1

                # update the edges
                # print 'T',
                candidates = []
                E_CON, V_CON = [], []
                ret = self.train_set.getRet()

                decay_tag = [0 for i in range(m)]
                for i in range(m):
                    for j in range(n):
                        if ret[i][j] == 0:
                            decay_tag[i] += 1

                for edge in self.train_set.candidates:
                    e, vs_index, vr_index = edge
                    e = e.view(1, -1).to(self.device)
                    vs = self.train_set.getApp(1, vs_index)
                    vr = self.train_set.getApp(0, vr_index)

                    e1 = self.Ephi1(e, vs, vr, self.u)
                    vr1 = self.Vphi(e1, vs, vr, self.u)
                    candidates.append((e1, vs, vr1, vs_index, vr_index))
                    E_CON.append(e1)
                    V_CON.append(vs)
                    V_CON.append(vr1)

                E = self.train_set.aggregate(E_CON).view(1, -1)
                V = self.train_set.aggregate(V_CON).view(1, -1)
                u1 = self.Uphi(E, V, self.u)

                for iteration in candidates:
                    e1, vs, vr1, vs_index, vr_index = iteration
                    if ret[vs_index][vr_index] == tau_threshold:
                        continue

                    e2 = self.Ephi2(e1, vs, vr1, u1)
                    self.train_set.edges[vs_index][vr_index] = e1.data.view(-1)

                    tmp = F.softmax(e2)
                    tmp = tmp.cpu().data.numpy()[0]

                    t = line_con[self.cur][vs_index][-1]
                    # ret[vs_index][vr_index] = float(tmp[0]) * pow(decay, t-1)
                    if decay_tag[vs_index] > 0:
                        ret[vs_index][vr_index] = min(float(tmp[0]) * pow(decay, t - 1), 1.0)
                    else:
                        ret[vs_index][vr_index] = float(tmp[0])

                # for j in ret:
                #     print j
                # print ret
                results = self.hungarian.compute(ret)

                out = open(outFile, 'a')
                nxt = self.train_set.nxt
                for (i, j) in results:
                    # print (i,j)
                    if ret[i][j] >= tau_threshold:
                        continue
                    e1 = self.train_set.edges[i][j].view(1, -1).to(self.device)
                    vs = self.train_set.getApp(1, i)
                    vr = self.train_set.getApp(0, j)

                    vr1 = self.Vphi(e1, vs, vr, self.u)
                    self.train_set.detections[nxt][j][0] = vr1.data
                    e2 = self.Ephi2(e1, vs, vr1, u1)
                    self.train_set.edges[i][j] = e2.data.view(-1)

                    id = id_con[self.cur][i]
                    id_con[self.nxt][j] = id
                    attr1 = line_con[self.cur][i]
                    attr2 = line_con[self.nxt][j]
                    # print attrs
                    attr2[1] = str(id)
                    if attr1[-1] > 1:
                        # for the missing detections
                        self.linearModel(out, attr1, attr2)
                    line = ''
                    for attr in attr2[:-1]:
                        line += attr + ','
                    if show_recovering:
                        line += '0'
                    else:
                        line = line[:-1]
                    out.write(line + '\n')

                if u_update:
                    self.u = u1.data

                for i in range(n):
                    if id_con[self.nxt][i] == -1:
                        id_con[self.nxt][i] = id_step
                        attrs = line_con[self.nxt][i]
                        attrs[1] = str(id_step)
                        line = ''
                        for attr in attrs[:-1]:
                            line += attr + ','
                        if show_recovering:
                            line += '0'
                        else:
                            line = line[:-1]
                        out.write(line + '\n')
                        id_step += 1
                out.close()

                # For missing & Occlusion
                index = 0
                for (i, j) in results:
                    while i != index:
                        attrs = line_con[self.cur][index]
                        # print '*', attrs, '*'
                        if attrs[-1] + t_gap <= gap:
                            attrs[-1] += t_gap
                            line_con[self.nxt].append(attrs)
                            id_con[self.nxt].append(id_con[self.cur][index])
                            self.train_set.moveApp(index)
                        index += 1

                    if ret[i][j] >= tau_threshold:
                        attrs = line_con[self.cur][index]
                        # print '*', attrs, '*'
                        if attrs[-1] + t_gap <= gap:
                            attrs[-1] += t_gap
                            line_con[self.nxt].append(attrs)
                            id_con[self.nxt].append(id_con[self.cur][index])
                            self.train_set.moveApp(index)
                    index += 1
                while index < m:
                    attrs = line_con[self.cur][index]
                    # print '*', attrs, '*'
                    if attrs[-1] + t_gap <= gap:
                        attrs[-1] += t_gap
                        line_con[self.nxt].append(attrs)
                        id_con[self.nxt].append(id_con[self.cur][index])
                        self.train_set.moveApp(index)
                    index += 1

                line_con[self.cur] = []
                id_con[self.cur] = []
                # print 'Results',
                self.train_set.swapFC()
                self.swapFC()
                # print '^.^|', step, tail
            gtIn.close()


if __name__ == '__main__':
    try:
        # types = [['POI', 0.7]]
        types = [['DPM', -0.6], ['SDP', 0.5], ['FRCNN', 0.5]]
        for t in types:
            type, tau_conf_score = t
            head = time.time()

            f_dir = 'results/'
            if not os.path.exists(f_dir):
                os.mkdir(f_dir)

            for i in range(len(test_seqs)):
                if tt_tag:
                    seq_index = test_seqs[i]
                    seq_len = test_lengths[i]
                else:
                    seq_index = seqs[i]
                    seq_len = lengths[i]

                print('The sequence:', seq_index, '- The length of the training data:', seq_len)

                # s_dir = f_dir + '%02d/' % seq_index
                # if not os.path.exists(s_dir):
                #     os.mkdir(s_dir)
                #     print(s_dir, 'does not exist!')

                t_dir = f_dir
                # t_dir = s_dir + '%d/' % seq_len
                if not os.path.exists(t_dir):
                    os.mkdir(t_dir)
                    print(t_dir, 'does not exist!')

                if tt_tag:
                    seq_dir = 'MOT%d-%02d-%s' % (year, test_seqs[i], type)
                    sequence_dir = mot_dataset_dir + 'MOT%d/test/' % year + seq_dir
                    print(' ', sequence_dir)

                    start = time.time()
                    print('     Evaluating Graph Network...')
                    gn = GN(test_seqs[i], test_lengths[i], type)
                else:
                    seq_dir = 'MOT%d-%02d-%s' % (year, seqs[i], type)
                    sequence_dir = mot_dataset_dir + 'MOT%d/train/' % year + seq_dir
                    print(' ', sequence_dir)

                    start = time.time()
                    print('     Evaluating Graph Network...')
                    gn = GN(seqs[i], lengths[i])
                print('     Recover the number missing detections:', gn.missingCounter)
                print('     The number of sideConnections:', gn.sideConnection)
                print('Time consuming:', (time.time() - start) / 60.0)
            print('Time consuming:', (time.time() - head) / 60.0)
    except KeyboardInterrupt:
        print('Time consuming:', time.time() - start)
        print('')
        print('-' * 90)
        print('Existing from training early.')
