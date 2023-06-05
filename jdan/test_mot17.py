import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'

from tracker import SSTTracker, TrackerConfig, Track
# from sst_tracker import TrackSet as SSTTracker
from tracking_utils.evaluation import Evaluator

import cv2
from data.mot_data_reader import MOTDataReader
import numpy as np
from config.config import config
from outils.timer import Timer
import argparse

import torch

import motmetrics as mm

is_debug = True
# is_debug = False

# --type train --show_image False  --log_folder /data2/whd/workspace/MOT/SST/results/log_folder
# --type test --show_image False  --log_folder /data2/whd/workspace/MOT/SST/results/log_folder


parser = argparse.ArgumentParser(description='Single Shot Tracker Test')
parser.add_argument('--version', default='v1', help='current version')
parser.add_argument('--mot_root', default=config['mot_root'], help='MOT ROOT')
parser.add_argument('--type', default=config['type'], help='train/test')
parser.add_argument('--show_image', default=True, help='show image if true, or hidden')
parser.add_argument('--save_video', default=True, help='save video if true')
parser.add_argument('--log_folder', default=config['log_folder'], help='video saving or result saving folder')
parser.add_argument('--mot_version', default=17, help='mot version')

args = parser.parse_args()


def test(choice=None):
    if args.type == 'train':
        # dataset_index = [2, 4, 5, 9, 10, 11, 13]
        if is_debug:
            dataset_index = [2]
            dataset_detection_type = {'-SDP'}
        else:
            dataset_index = [2, 4, 5, 9, 10, 11, 13]
            dataset_detection_type = {'-DPM', '-FRCNN', '-SDP'}


    if args.type == 'test':
        # dataset_index = [3, 6, 7, 8, 12, 14]
        # dataset_index = [1, 3, 6, 7, 8, 12, 14]
        # dataset_detection_type = {'-FRCNN', '-SDP', '-DPM'}
        # for fix MOT17-06 image size
        dataset_index = [6]
        dataset_detection_type = {'-DPM', '-FRCNN', '-SDP'}

    dataset_image_folder_format = os.path.join(args.mot_root, args.type+'/MOT'+str(args.mot_version)+'-{:02}{}/img1')
    detection_file_name_format=os.path.join(args.mot_root, args.type+'/MOT'+str(args.mot_version)+'-{:02}{}/det/det.txt')

    if not os.path.exists(args.log_folder):
        os.mkdir(args.log_folder)

    save_folder = ''
    choice_str = ''
    if not choice is None:
        choice_str = TrackerConfig.get_configure_str(choice)
        save_folder = os.path.join(args.log_folder, choice_str)
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        # else:
        #     return

    saved_file_name_format = os.path.join(save_folder, 'MOT'+str(args.mot_version)+'-{:02}{}.txt')
    save_video_name_format = os.path.join(save_folder, 'MOT'+str(args.mot_version)+'-{:02}{}.avi')

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


            if first_run and args.save_video:
                vw = cv2.VideoWriter(save_video_name, cv2.VideoWriter_fourcc('M','J','P','G'), 10, (w, h))
                first_run = False

            det[:, [2,4]] /= float(w)
            det[:, [3,5]] /= float(h)
            timer.tic()
            with torch.no_grad():  # solve CUDA memory overflow
                # image_folder == '/data2/whd/workspace/MOT/FairMOT/src/data/MOT17/images/test/MOT17-03-FRCNN/img1' and i == 700
                image_org = tracker.update(img, args.show_image, i)  # SST tracker entrance
            timer.toc()
            print('{}:{}, {}, {}\r'.format(os.path.basename(saved_file_name), i, int(i*100/len(reader)), choice_str))
            if args.show_image and not image_org is None:
                cv2.imshow('res', image_org)
                cv2.waitKey(1)

            if args.save_video and not image_org is None:
                vw.write(image_org)

            # save result
            box_id = 0
            for t in tracker.tracks:
                n = t.nodes[-1]
                if t.age == 1:
                    b = n.get_box(tracker.frame_index-1, tracker.recorder)
                    result.append(
                        [i+1] + [t.id+1] + [b[0]*w] + [b[1]*h] + [b[2]*w] + [b[3]*h] + [-1] + [-1] + [-1] + [-1]  # recorder.all_boxes[-1] -> box_id
                    )
                    box_id += 1
        # save data
        np.savetxt(saved_file_name, np.array(result), fmt="%d,%d,%s,%s,%s,%s,%d,%d,%d,%d", delimiter=',')  # no result: ValueError: fmt has wrong number of % formats
        # np.savetxt(saved_file_name, np.array(result).astype(int), fmt='%i')
        print(result_str)

        data_type = 'mot'
        # evaluator = Evaluator(args.mot_root, seq, data_type)
        # accs.append(evaluator.eval_file(saved_file_name))  # src/data/MOT17/images/results/MOT15_val_all_dla34/MOT17-01-SDP.txt

    print('Tracking total time: ', timer.total_time)    # 200.08386325836182
    print('Tracking average time: ', timer.average_time)  # 0.33347310543060305 (Idea: 6.3 Hz)

    # if args.type == 'train':
    #     # get summary
    #     seqs_str = '''MOT17-02-SDP
    #                   MOT17-04-SDP
    #                   MOT17-05-SDP
    #                   MOT17-09-SDP
    #                   MOT17-10-SDP
    #                   MOT17-11-SDP
    #                   MOT17-13-SDP'''
    #     seqs = [seq.strip() for seq in seqs_str.split()]
    #     metrics = mm.metrics.motchallenge_metrics
    #     mh = mm.metrics.create()
    #     summary = Evaluator.get_summary(accs, seqs, metrics)
    #     strsummary = mm.io.render_summary(
    #         summary,
    #         formatters=mh.formatters,
    #         namemap=mm.io.motchallenge_metric_names
    #     )
    #     print(strsummary)

def test_eval():
    # test evaluator for tracking result
    # seqs_str = '''MOT17-02-SDP
    #                   MOT17-04-SDP
    #                   MOT17-05-SDP
    #                   MOT17-09-SDP
    #                   MOT17-10-SDP
    #                   MOT17-11-SDP
    #                   MOT17-13-SDP'''
    timer = Timer()
    timer.tic()
    seqs_str = '''MOT17-02-SDP'''
    seqs = [seq.strip() for seq in seqs_str.split()]
    result_dir = '/data2/whd/workspace/MOT/SST/results/log_folder/0_0_4_0_3_3'
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

'''
--type train --show_image False  --log_folder /data2/whd/workspace/MOT/SST/results/log_folder
'''
if __name__ == '__main__':
    all_choices = TrackerConfig.get_choices_age_node()
    iteration = 3

    # test_eval()

    # test()

    i = 0
    for age in range(1):
        for node in range(1):
            c = (0, 0, 4, 0, 3, 3)
            choice_str = TrackerConfig.get_configure_str(c)
            TrackerConfig.set_configure(c)
            print('=============================={}.{}=============================='.format(i, choice_str))
            test(c)
            i += 1

    # test_eval()

'''
solve affinity computation:
/data2/whd/workspace/MOT/SST/results/log_folder/0_0_4_0_3_3/MOT17-02-SDP.txt
Tracking total time:  189.31339025497437
Tracking average time:  0.3155223170916239
Evaluating MOT17-02-SDP
Evaluation time:  26.15590810775757
              IDF1   IDP   IDR  Rcll  Prcn GT MT PT ML  FP   FN IDs   FM  MOTA  MOTP
MOT17-02-SDP 56.8% 68.4% 48.6% 69.1% 97.1% 62 23 30  9 377 5745 232  628 65.8% 0.191
OVERALL      56.8% 68.4% 48.6% 69.1% 97.1% 62 23 30  9 377 5745 232  628 65.8% 0.191

test train dataset:
Tracking total time:  7931.741522073746
Tracking average time:  0.5008677394590645 


load sst pretrained final net
Tracking total time:  241.44399762153625
Tracking average time:  0.4024066627025604
Evaluating MOT17-02-SDP
Evaluation time:  25.313905239105225
             IDF1  IDP  IDR  Rcll  Prcn GT MT PT ML  FP   FN  IDs    FM  MOTA  MOTP
MOT17-02-SDP 6.1% 8.2% 4.9% 58.0% 96.4% 62 12 41  9 403 7813 8201  1649 11.6% 0.194
OVERALL      6.1% 8.2% 4.9% 58.0% 96.4% 62 12 41  9 403 7813 8201  1649 11.6% 0.194

Tracking total time:  478.55428314208984
Tracking average time:  0.7975904719034831
Evaluating MOT17-02-SDP
Evaluation time:  377.06544494628906
             IDF1  IDP  IDR  Rcll  Prcn GT MT PT ML  FP   FN  IDs    FM  MOTA  MOTP
MOT17-02-SDP 6.6% 8.5% 5.4% 61.3% 96.5% 62 17 36  9 419 7190 8988  1133 10.7% 0.192
OVERALL      6.6% 8.5% 5.4% 61.3% 96.5% 62 17 36  9 419 7190 8988  1133 10.7% 0.192


0.4
Tracking total time:  205.24776530265808
Tracking average time:  0.34207960883776345
Evaluating MOT17-02-SDP
Evaluation time:  872.7663218975067
             IDF1  IDP  IDR  Rcll  Prcn GT MT PT ML  FP   FN   IDs   FM  MOTA  MOTP
MOT17-02-SDP 0.4% 0.4% 0.3% 69.1% 97.2% 62 23 31  8 372 5740 12784  762 -1.7% 0.186
OVERALL      0.4% 0.4% 0.3% 69.1% 97.2% 62 23 31  8 372 5740 12784  762 -1.7% 0.186
'''

'''
              IDF1   IDP   IDR  Rcll  Prcn GT MT PT ML  FP   FN IDs   FM  MOTA  MOTP
MOT17-02-SDP 56.8% 68.4% 48.6% 69.1% 97.1% 62 23 30  9 377 5745 232  628 65.8% 0.191

mot --val_mot17 True --load_model ../models/all_dla34.pth --conf_thres 0.4
              IDF1   IDP   IDR  Rcll  Prcn  GT  MT  PT ML   FP    FN IDs    FM  MOTA  MOTP IDt IDa IDm
MOT17-02-SDP 64.0% 77.4% 54.6% 68.6% 97.4%  62  23  30  9  341  5831 182   656 65.8% 0.194  98  33  13
MOT17-04-SDP 83.7% 86.2% 81.4% 86.4% 91.4%  83  51  22 10 3866  6489  34   198 78.2% 0.171   8  20   2
MOT17-05-SDP 76.0% 82.9% 70.1% 81.2% 96.0% 133  63  59 11  237  1300  78   208 76.7% 0.199  83  25  40
MOT17-09-SDP 65.4% 71.6% 60.3% 81.2% 96.5%  26  19   7  0  157   999  52   104 77.3% 0.165  37  12   7
MOT17-10-SDP 65.0% 72.0% 59.2% 78.9% 96.0%  57  35  22  0  422  2707 146   401 74.5% 0.214  86  43  14
MOT17-11-SDP 85.7% 87.9% 83.6% 90.6% 95.3%  75  53  18  4  421   890  37   133 85.7% 0.157  24  18  13
MOT17-13-SDP 77.1% 82.0% 72.7% 84.0% 94.8% 110  75  28  7  535  1860  86   372 78.7% 0.206  72  26  35
OVERALL      76.8% 82.3% 72.0% 82.1% 93.9% 546 319 186 41 5979 20076 615  2072 76.3% 0.183 408 177 124
'''

'''
test mot17 on train dataset:
total_time:   3317.5044293403625
average_time: 0.20949131278986882
'''