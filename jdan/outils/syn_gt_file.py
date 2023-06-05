import os
import shutil

source_dir = '/data2/whd/workspace/MOT/FairMOT/src/data/MOT17/images/results/MOT15_val_all_dla34/'
target_root = '/data2/whd/workspace/MOT/SST/results/MOT17/train/'

seqs_str = '''MOT17-02-SDP
              MOT17-04-SDP
              MOT17-05-SDP
              MOT17-09-SDP
              MOT17-10-SDP
              MOT17-11-SDP
              MOT17-13-SDP'''
seqs = [seq.strip() for seq in seqs_str.split()]


for i in range(len(seqs)):
    source_file = os.path.join(source_dir, seqs[i]+'.txt')
    target_dir = os.path.join(target_root, seqs[i], 'gt')
    try:
        shutil.copy(source_file, target_dir)
        os.rename(os.path.join(target_dir, seqs[i]+'.txt'), os.path.join(target_dir, 'gt.txt'))
    except IOError as e:
        print("Unable to copy file. %s" % e)

# adding exception handling
