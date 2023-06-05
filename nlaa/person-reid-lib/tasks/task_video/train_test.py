#!usr/local/bin/python

import _init_paths
import os
import torch
import numpy as np
import random
from lib.utils.manager import Manager
from lib.utils.util import file_abs_path, ParseArgs
from lib.utils.meter import timer_lite
from solver import Solver
from lib.evaluation.eval_tools import compute_rank


def main():
    cur_dir = file_abs_path(__file__)
    manager = Manager(cur_dir, seed=None, mode='Train')
    logger = manager.logger
    ParseArgs(logger)
    if manager.seed is not None:
        random.seed(manager.seed)
        np.random.seed(manager.seed)
        torch.manual_seed(manager.seed)

    # ['iLIDS-VID', 'PRID-2011', 'LPW', 'MARS', 'VIPeR', 'Market1501', 'CUHK03', 'CUHK01', 'DukeMTMCreID', 'GRID', 'DukeMTMC-VideoReID']
    #       0            1         2      3        4          5           6         7             8           9             10

    manager.set_dataset(1)

    perf_box = {}
    repeat_times = 10
    for task_i in range(repeat_times):
        manager.split_id = int(task_i) 
        task = Solver(manager)
        train_test_time = timer_lite(task.run)
        perf_box[str(task_i)] = task.perf_box
        manager.store_performance(perf_box)

        logger.info('-----------Total time------------')
        logger.info('Split ID:' + str(task_i) + '  ' + str(train_test_time))
        logger.info('---------------------------------')

    compute_rank(perf_box, logger)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()


'''
2019-11-05 13:32:14,269 - solverbase.py[line:120] - INFO: Test Begin
2019-11-05 13:33:02,977 - eval_base.py[line:99] - INFO: mAP [81.637800]   Rank1 [73.333298] Rank5 [91.333298] Rank10 [96.666702] Rank20 [98.000000]
2019-11-05 13:33:03,274 - util.py[line:126] - INFO: Store data ---> /home/d/workspace/MOT/DMAN_MOT/person-reid-lib/tasks/task_video/output/result/cmc_2019-11-04_182653.json
2019-11-05 13:33:03,274 - train_test.py[line:39] - INFO: -----------Total time------------
2019-11-05 13:33:03,274 - train_test.py[line:40] - INFO: Split ID:9  1 hours, 55 mins, 59 sec
2019-11-05 13:33:03,274 - train_test.py[line:41] - INFO: ---------------------------------
2019-11-05 13:33:03,275 - eval_tools.py[line:21] - INFO: CMC and mAP for   10 times.
2019-11-05 13:33:03,275 - eval_tools.py[line:22] - INFO: Epoch      mAP     R1     R5      R10     R20  
2019-11-05 13:33:03,275 - eval_tools.py[line:24] - INFO:  200     77.92   68.40   89.13   94.40   97.40   
2019-11-05 13:33:03,275 - eval_tools.py[line:24] - INFO:  220     78.14   69.00   89.40   94.20   97.47   
2019-11-05 13:33:03,275 - eval_tools.py[line:24] - INFO:  240     78.84   69.93   90.13   94.60   98.07   
2019-11-05 13:33:03,275 - eval_tools.py[line:24] - INFO:  260     78.97   69.60   90.60   95.20   97.60   
2019-11-05 13:33:03,275 - eval_tools.py[line:24] - INFO:  280     78.28   68.53   90.87   95.13   97.20   
2019-11-05 13:33:03,275 - eval_tools.py[line:24] - INFO:  300     80.26   71.60   91.60   96.13   98.27   
2019-11-05 13:33:03,275 - eval_tools.py[line:24] - INFO:  320     79.51   70.13   91.27   95.27   98.20   
2019-11-05 13:33:03,275 - eval_tools.py[line:24] - INFO:  340     80.49   71.87   91.73   95.47   98.47   
2019-11-05 13:33:03,275 - eval_tools.py[line:24] - INFO:  360     78.95   69.53   91.27   94.93   97.53   
2019-11-05 13:33:03,275 - eval_tools.py[line:24] - INFO:  380     79.40   69.73   91.13   95.73   98.33   
2019-11-05 13:33:03,275 - eval_tools.py[line:24] - INFO:  400     81.07   72.47   91.53   95.73   98.33   
2019-11-05 13:33:03,275 - eval_tools.py[line:24] - INFO:  420     80.73   71.27   92.73   96.27   98.53   
2019-11-05 13:33:03,276 - eval_tools.py[line:24] - INFO:  440     80.47   71.07   92.47   96.13   98.53   
2019-11-05 13:33:03,276 - eval_tools.py[line:24] - INFO:  460     80.61   71.60   92.13   95.87   98.53   
2019-11-05 13:33:03,276 - eval_tools.py[line:24] - INFO:  480     80.66   71.73   92.20   96.60   98.80   
2019-11-05 13:33:03,276 - eval_tools.py[line:24] - INFO:  500     80.47   71.40   92.27   96.13   98.80   
2019-11-05 13:33:03,276 - eval_tools.py[line:24] - INFO:  520     80.48   71.93   92.00   96.20   98.07   
2019-11-05 13:33:03,276 - eval_tools.py[line:24] - INFO:  540     81.14   72.53   92.80   96.60   98.40   
2019-11-05 13:33:03,276 - eval_tools.py[line:24] - INFO:  560     80.21   70.93   91.53   95.93   98.73   
2019-11-05 13:33:03,276 - eval_tools.py[line:24] - INFO:  580     80.69   72.27   91.93   96.27   98.00   
2019-11-05 13:33:03,276 - eval_tools.py[line:24] - INFO:  600     81.43   72.87   92.53   95.93   97.80   
2019-11-05 13:33:03,276 - eval_tools.py[line:24] - INFO:  620     80.94   71.67   92.73   96.27   98.07   
2019-11-05 13:33:03,276 - eval_tools.py[line:24] - INFO:  640     80.33   71.60   91.87   96.13   98.07   
2019-11-05 13:33:03,276 - eval_tools.py[line:24] - INFO:  660     80.83   72.07   92.07   96.60   98.73   
2019-11-05 13:33:03,276 - eval_tools.py[line:24] - INFO:  680     81.00   71.60   92.80   96.40   98.87   
2019-11-05 13:33:03,276 - eval_tools.py[line:24] - INFO:  700     80.61   71.67   92.27   96.33   98.53   
2019-11-05 13:33:03,276 - eval_tools.py[line:24] - INFO:  720     80.62   71.47   92.47   96.87   98.87   
2019-11-05 13:33:03,276 - eval_tools.py[line:24] - INFO:  740     80.70   71.73   92.00   96.40   98.67   
2019-11-05 13:33:03,276 - eval_tools.py[line:24] - INFO:  760     80.20   71.33   90.73   95.13   98.47   
2019-11-05 13:33:03,276 - eval_tools.py[line:24] - INFO:  780     81.30   73.00   91.93   96.60   98.73   
2019-11-05 13:33:03,276 - eval_tools.py[line:24] - INFO:  800     81.14   72.20   92.00   96.73   98.40   
2019-11-05 13:33:03,276 - eval_tools.py[line:24] - INFO:  820     80.47   71.40   92.33   96.60   98.60   
2019-11-05 13:33:03,276 - eval_tools.py[line:24] - INFO:  840     81.54   72.87   92.67   96.93   98.67   
2019-11-05 13:33:03,276 - eval_tools.py[line:24] - INFO:  860     79.87   71.00   91.27   96.13   98.33   
2019-11-05 13:33:03,276 - eval_tools.py[line:24] - INFO:  880     79.81   70.80   92.07   96.47   98.73   
2019-11-05 13:33:03,276 - eval_tools.py[line:24] - INFO:  900     80.34   70.87   92.93   96.87   98.87 
'''