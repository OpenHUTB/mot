import os
import numpy as np


def get_gpu():
    # 获取每个 GPU 的剩余显存数，并存放到 tmp 文件中
    # -q, query, display GPU or Unit info
    # -d TYPE, display only selected information: memory, utilization, ecc, temperature, power, clock, compute, pids,
    #       performance, supported_clocks, page_retirement, accounting, encoder_stats
    # -A NUM, --after-context=NUM
    # nvidia-smi -q -d Memory | grep -A 4 GPU | grep Free | awk '{print $3}'
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_gpu = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    print("剩余显存最大的 GPU 编号是： {}", np.argmax(memory_gpu))  # 获取剩余显存最多的 GPU 的编号
    os.system('rm tmp')  # 删除临时生成的 tmp 文件
    return str(np.argmax(memory_gpu))