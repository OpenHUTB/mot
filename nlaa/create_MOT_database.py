import argparse
import os
import numpy as np
import scipy.io as sio
import random
import cv2

from util.io_utils import *

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

# python3 create_MOT_database.py
# --data_root /data/whd/MOT/ --dataset MOT16 --output_dir ./MOT16_database/
# --data_dir /data/whd/MOT/MOT16 --info_dir /data/whd/MOT/MOT16/MOT16-evaluation/info/  --output_dir ./MOT_database/
# --data_dir /data/whd/MARS-v160809 --info_dir /data/whd/MARS-v160809/MARS-evaluation/info/  --output_dir ./MARS_database/

# 0001C1T0013F017.jpg -> 0001C1T0014F001.jpg  (654 image change to another person in '/data/whd/MARS-v160809/bbox_train/0001/0001C1T0008F022.jpg')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root',help='path/to/MARS/')
    parser.add_argument('--dataset', help='path/to/MARS/')
    parser.add_argument('--output_dir',help='path/to/save/database',default='./MARS_database')
    args = parser.parse_args()

    # Process Train Dataset
    pth = args.data_root + args.dataset + '/train/'
    # create MOT video Re-ID data set according MARS directory
    os.system('mkdir -p %s'%(args.output_dir))
    videos = os.listdir(pth)

    # # Crop train image
    # for vname in videos:
    #     os.system('mkdir -p %s' % os.path.join(args.output_dir, vname))
    #     frames_gt = read_txt_gtV2(pth + vname + '/gt/gt.txt')
    #     imgs_path = pth + vname + '/img1/'
    #     imgs = sorted(os.listdir(imgs_path))
    #     for frameid in range(len(frames_gt)):
    #         cur_img = cv2.imread(os.path.join(imgs_path, imgs[frameid]))
    #         gt_bboxes = frames_gt[str(frameid + 1)]
    #         for box_id in range(len(gt_bboxes)):
    #             cur_gt_box = gt_bboxes[box_id]
    #             person_path = os.path.join(args.output_dir, vname, cur_gt_box[0])
    #             os.system('mkdir -p %s' % (person_path))  # MOT16_database/MOT16-05/1/
    #             x1 = int(cur_gt_box[1])
    #             y1 = int(cur_gt_box[2])
    #             x2 = int(cur_gt_box[3])
    #             y2 = int(cur_gt_box[4])
    #             if x1 < 0:
    #                 x1 = 0
    #             if y1 < 0:
    #                 y1 = 0
    #             if x2 < 0:
    #                 x2 = 0
    #             if y2 <0:
    #                 y2 = 0
    #             if abs(x2-x1) > 5 and abs(y2-y1) > 5:
    #                 person_img = cur_img[y1: y2, x1: x2, :]
    #                 cv2.imwrite(os.path.join(person_path,
    #                                          ("%04d%02dF%04d.jpg") % (int(cur_gt_box[0]), int(vname[6:]), frameid)),
    #                             person_img)  # person_id, video_id, frame_id
    #             else:
    #                 print()
    #
    #         print(vname, ':', frameid)

    # Train image list
    train_imgs = []
    for vname in videos:
        video_dir = os.path.join(args.output_dir, vname)
        ids = sorted(os.listdir(video_dir))
        for id in ids:
            images = sorted(os.listdir(os.path.join(video_dir, id)))
            for image in images:
                train_imgs.append(os.path.abspath(os.path.join(video_dir, id, image)))
    train_imgs = np.array(train_imgs)
    np.savetxt(os.path.join(args.output_dir, 'train_path.txt'), train_imgs, fmt='%s', delimiter='\n')

    train_info = np.zeros((int(len(train_imgs) / 25), 4), dtype=int)
    tracklet_idx = 0
    img_idx = 0
    for vname in videos:
        video_dir = os.path.join(args.output_dir, vname)
        ids = sorted(os.listdir(video_dir))
        for id in ids:
            images = sorted(os.listdir(os.path.join(video_dir, id)))
            train_info[tracklet_idx, 0] = img_idx  # tracklet image index
            train_info[tracklet_idx, 1] = img_idx+len(images)-1  # front image end index
            train_info[tracklet_idx, 2] = int(id)  # person id
            train_info[tracklet_idx, 3] = int(vname[6:])  # camera id
            tracklet_idx += 1
            img_idx += len(images)
    np.save(os.path.join(args.output_dir, 'train_info.npy'), train_info[0:tracklet_idx, :])


    # ## Genarate train_info.npy (every tracklet(row) contains: tracklet start image index(in train_path.txt), end, person id, camera id
    # train_info = np.zeros((int(len(train_imgs)/25), 4))
    # tracklet_len = 50
    # tracklet_idx = 0
    # is_new_tracklet = True
    # img_idx = 0
    # for i in train_imgs:
    #     (filepath, tempfilename) = os.path.split(i)
    #     person_id = int(tempfilename[0:4])
    #     camera_id = int(tempfilename[4:6])
    #     if img_idx == 0:
    #         is_new_tracklet = True
    #         train_info[tracklet_idx, 0] = img_idx  # tracklet image index
    #         train_info[tracklet_idx, 2] = person_id  # person id
    #         train_info[tracklet_idx, 3] = camera_id  # camera id
    #         prev_person_id = person_id
    #         prev_camera_id = camera_id
    #         tracklet_idx += 1
    #         img_idx += 1
    #         continue
    #     if prev_person_id != person_id or prev_camera_id != camera_id:
    #         train_info[tracklet_idx-1, 1] = img_idx  # front image end index
    #         is_new_tracklet = True
    #         train_info[tracklet_idx, 0] = img_idx  # tracklet image index
    #         train_info[tracklet_idx, 2] = person_id  # person id
    #         train_info[tracklet_idx, 3] = camera_id  # camera id
    #         tracklet_idx += 1
    #     else:
    #         img_idx += 1
    #     print()
    #
    # np.save(os.path.join(args.output_dir, 'train_info.npy'), train_info)

    ## process matfile
    # train_info = sio.loadmat(os.path.join(args.info_dir,'tracks_train_info.mat'))['track_train_info']
    # test_info = sio.loadmat(os.path.join(args.info_dir,'tracks_test_info.mat'))['track_test_info']
    # query_IDX = sio.loadmat(os.path.join(args.info_dir,'query_IDX.mat'))['query_IDX']
    
    # start from 0 (matlab starts from 1)
    # train_info[:,0:2] = train_info[:,0:2]-1
    # test_info[:,0:2] = test_info[:,0:2]-1
    # query_IDX = query_IDX -1
    # np.save(os.path.join(args.output_dir,'train_info.npy'),train_info)
    # np.save(os.path.join(args.output_dir,'test_info.npy'),test_info)
    # np.save(os.path.join(args.output_dir,'query_IDX.npy'),query_IDX)
    


'''
0065 C1 T0002 F0016.jpg为例。
0065表示的行人的id，也就是 bbox_train文件夹中对应的 0065子文件夹名；
C1表示摄像头的id，说明这张图片是在第1个摄像头下拍摄的（一共有6个摄像头）；
T0002表示关于这个行人视频段中的第2个小段视频（tracklet）；
F0016表示在这张图片是在这个小段视频（tracklet）中的第16帧。在每个小段视频（tracklet）中，帧数从 F0001开始。
'''